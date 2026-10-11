### 2.7 Token2Wav: turn speech codes into audible sound

Talker outputs integers, but a loudspeaker needs a rapidly sampled waveform. Token2Wav bridges that gap in two stages: first generate a time/frequency representation conditioned on the codes, then synthesize samples from it. This is a different job from recognizing the input recording or choosing the answer's words.

#### Keep three audio representations separate

1. Input audio uses 128-bin features and continuous encoder vectors.
2. Talker generates discrete codec IDs.
3. Token2Wav produces 80-bin mel features, then waveform samples.

The input and output mel dimensions differ. Their transforms and normalization also differ. Reusing the input Whisper features as output-vocoder conditioning would conflate separate interfaces.

#### What is a speech codec/tokenizer?

A learned audio tokenizer compresses acoustic information into discrete symbols. A generic vector-quantization explanation is:

$$h=f_{\rm enc}(x),\qquad
c=\arg\min_k\|h-e_k\|^2,\qquad \hat x=f_{\rm dec}(e_c).$$

In this background formula, x is a waveform segment, f_enc produces a continuous feature h, the codebook contains vectors e_k, and c selects the nearest one by squared Euclidean distance. The decoder f_dec reconstructs audio from the selected vector. A typical system learns an encoder, a table of representative vectors, and a decoder together; discretization gives a language-model-like predictor a finite set of symbols to choose from. Here e_k is a codebook vector. Reconstruction and codebook/commitment objectives are common ways to train such systems. **This is background intuition, not a disclosed exact architecture or objective for qwen-tts-tokenizer.** The Omni report names that tokenizer but does not fully specify its encoder and training recipe. This HF snapshot exposes codec-conditioned reconstruction rather than a complete raw-waveform tokenizer training implementation.

#### Flow matching: learn a velocity field

Read the archived [Flow Matching paper](sources/references/flow-matching-2210.02747v1.pdf) for the general framework. The intuition is to start with noise and repeatedly move it toward a plausible spectrogram, with the speech codes telling the network what to synthesize. A simple teaching path connects noise x_0 to a target mel sample x_1:

$$x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0,$$

$$\mathcal L_{\rm FM}=
\mathbb E_{t,x_0,x_1}\|v_\theta(x_t,t,c)-u_t\|^2.$$

t is interpolation time in [0,1], not audio time. c denotes codec and speaker conditioning. The network predicts a velocity with the same shape as the evolving mel tensor. Training samples random t and supervises the velocity. The subscript theta denotes the learned parameters, E means an average over training samples, and the squared norm sums prediction errors over mel/time entries. A velocity is a proposed change, not a waveform sample or codec ID.

For one scalar entry, take x_0=-1 and x_1=3. At t=0.25 the interpolated value is 0, and the target velocity is 4. A network predicting velocity 3 incurs squared error 1 on this entry. The real network predicts an entire 80-bin spectrogram's changes, conditioned on speech and speaker information.

This linear path/objective is a transparent example of conditional flow matching. The inference code alone does not establish Omni's exact training path, noise schedule, loss weighting, or complete data recipe.

During synthesis, solve:

$$\frac{dx_t}{dt}=v_\theta(x_t,t,c),\qquad x_0\sim\mathcal N(0,I).$$

The solver repeatedly predicts changes to the whole mel state. Flow matching is not the same operation as autoregressively choosing the next codec token.

#### Follow the DiT inputs

[[code:Qwen2_5OmniToken2WavDiTModel.sample]] creates noise, sizes the output to codec length times repeats, creates time points, and integrates the ODE.

[[code:DiTCodecEmbedding.forward]] embeds codec IDs and repeats them to the mel rate. repeats=2 means two mel frames per codec token. [[code:DiTInputEmbedding.forward]] concatenates noisy mel, encoded reference-mel information, codec embeddings, and speaker conditioning, then projects them into the Transformer.

The reference-mel path uses an ECAPA-style time-delay network: temporal convolutions, residual/multiscale features, channel attention, and attentive statistics pooling summarize speaker-related information. Channel attention weights useful feature channels, while attentive pooling weights time positions and summarizes their statistics. This turns a variable-length reference into speaker-related conditioning. The released inference path does not disclose the full training objective for this conditioning encoder. That conditioning encoder is part of the waveform path; it is not Thinker's Whisper-derived audio tower.

[[code:DiTDecoderLayer.forward]] uses timestep-conditioned normalization and gates:

$$\tilde h={\rm LN}(h)\odot(1+s(t))+b(t),\qquad
h_{\rm next}=h+g(t)\odot{\rm Attention}(\tilde h).$$

Here h is a hidden-state tensor, LN normalizes its channels, and the elementwise product symbol means multiplying corresponding entries. The functions s(t), b(t), and g(t) supply timestep-dependent scale, shift, and gate values. The gate controls how much the attention result changes the existing state. These s and b symbols are modulation values, not the later guidance scale. Additional modulation controls the MLP branch. The “DiT” name identifies a Transformer backbone that processes the evolving noisy representation with time conditioning; the sampler here follows a flow-matching ODE.

#### Guidance and numerical integration

The code combines conditional and null predictions:

$$v_{\rm guided}=v_{\rm cond}+s(v_{\rm cond}-v_{\rm null}).$$

This code's s is extra guidance beyond the conditional prediction, equivalent to a conventional null-plus-guidance expression with scale 1+s. The null path drops codec/audio/speaker conditions using the implementation's conventions.

[[code:RungeKutta4ODESolver]] implements RK4. For step size h:

$$k_1=v(t,x),\quad k_2=v(t+h/2,x+hk_1/2),$$
$$k_3=v(t+h/2,x+hk_2/2),\quad k_4=v(t+h,x+hk_3),$$
$$x_{\rm next}=x+\frac h6(k_1+2k_2+2k_3+k_4).$$

Each k is a velocity estimate. RK4 probes the start, two midpoints, and the end of a step, then uses their weighted average. The step h here is a change in ODE time, distinct from the hidden tensor h above and from audio duration. For the constant velocity 4 in the scalar example, a step h=0.1 moves the state by 0.4; ten such steps move -1 to 3.

The demonstration uses the known constant velocity from noise to a synthetic target. The tiny final error checks integration arithmetic. It does not train a neural network or prove that an untrained model can recover real speech.

The code also transforms the time grid with a sway coefficient, redistributing integration points between the start and end of the trajectory. More time points generally cost more evaluations; a default num_steps value is not a benchmark of audio quality.

#### Streaming masks: inspect each layer

[[code:Qwen2_5OmniToken2WavDiTModel._create_block_diff]] computes key-block minus query-block indices. [[code:DiTDecoderLayer.forward]] allows attention within configured lookback/lookahead limits.

The defaults are block_size=24 mel positions, lookback at layers 0 and 20, and lookahead at layer 10. Most layers attend within the current block. Those cross-block layers compose into an attention dependency envelope extending up to two blocks backward and one forward, matching the paper's four-block account. It is not a direct four-block mask in every layer.

For example, a query in block 3 can ultimately depend on blocks 1 through 4 after the relevant layers compose. The following mask demonstration makes that dependency visible. It distinguishes an individual layer’s allowed connections from the network’s combined receptive field.

Finite lookahead requires waiting for some future codec information. It is streaming-compatible with bounded delay, not strictly causal with zero lookahead. Attention-mask bounds alone also do not establish that the entire speaker/reference-conditioning path is causal.

#### BigVGAN: reconstruct samples from mel

The mel spectrogram still is not playable audio: it describes energy in bands, without explicitly giving every waveform sample. A vocoder learns to produce a suitable waveform. The [BigVGAN paper](sources/references/bigvgan-2206.04658v1.pdf) describes a convolutional generator trained with adversarial, feature-matching, and reconstruction-related losses. Its generator upsamples mel features and uses periodic activations with anti-alias filtering to model waveform structure. Periodic activations help represent repeating waveform patterns; anti-alias filtering reduces unwanted frequency artifacts introduced around those nonlinear operations. This is background from the original vocoder, not a claim that Omni releases an identical training recipe.

[[code:Qwen2_5OmniToken2WavBigVGANModel.forward]] normalizes mel values, runs transposed convolutions and residual blocks, applies periodic/anti-aliased activations, and outputs clamped waveform samples. Its upsample rates multiply to 240. With the model card's 24 kHz output, this corresponds nominally to 100 mel frames per second and, with repeats=2, 50 codec tokens per second. For 50 codec tokens, repetition gives 100 mel frames, and nominal upsampling gives 24,000 waveform samples—about one second of output. Boundary handling follows the actual decoder. These output rates differ from the input audio encoder's 25 representations per second.

[[code:Qwen2_5OmniToken2WavModel.forward]] connects the sampler and vocoder. Notice that a local variable named generated_waveform inside the DiT sampler actually holds an intermediate mel-state trajectory; trace shapes rather than trusting names.

#### Paper assignment and check

[[paper:2.3]] explains speech codes; [[paper:2.4]] and Figure 4 explain streaming reconstruction. Read the original component papers for the theory omitted by the short report.

**Check.** Are ODE time t, video timestamps, and autoregressive codec indices the same coordinate?

**Answer.** No. ODE time indexes denoising/transport progress; video timestamps index physical input time; codec indices index generated speech symbols.


**What the output losses mean:** the original vocoder’s discriminator learns to distinguish real and generated waveform segments; the generator learns to make generated segments difficult to distinguish. Feature-matching losses also compare internal discriminator activations, and reconstruction-related terms encourage acoustic agreement with a reference. These explain how a convolutional generator can learn realistic audio, while the report’s unreleased full training recipe remains a separate limitation.
