### 2.2 Audio encoder: turn sound into understanding vectors

#### Why preserve audio information?

Inputs can include speech, music, laughter, alarms, and mixtures. A transcript can discard information needed to answer questions about sound. Thinker's audio path therefore extracts continuous representations rather than first requiring text or speech-codec tokens.

The desired output is a list of vectors describing sound, including more than spoken words. Each vector has 2,048 entries so Thinker can insert it into its own sequence. The encoder's job is to turn a fast waveform into a shorter sequence while retaining useful information.

#### First explain Whisper's architecture and training

The original [Whisper paper](sources/references/whisper-2212.04356v1.pdf), especially the approach/model discussion on PDF pages 3–4, describes an encoder–decoder Transformer trained on large-scale weakly supervised audio/text pairs. Its encoder processes log-mel features through convolutions and self-attention. Its autoregressive text decoder cross-attends to encoder states: while choosing a text token, it can read the audio representations. “Teacher forcing” means supplying the correct previous text tokens during training instead of using the decoder’s own guesses. Predicting the next correct token supplies the learning signal. Special tokens identify tasks and languages; teacher-forced next-token prediction trains transcription and related supervised tasks.

Whisper's encoder is not its text decoder, and neither is merely a mel filter bank. Omni initializes an audio encoder from Whisper-large-v3 and adopts the audio-understanding approach of [Qwen2-Audio](sources/references/qwen2-audio-2407.10759v1.pdf). The encoder connects to Qwen's LLM, replacing the original Whisper decoder as the answering system.

The original Whisper paper predates large-v3. The precise 128-bin feature setting used here is established by the Omni paper/checkpoint, rather than inferred from that earlier paper.

#### Waveform to log-mel features

The paper specifies 16 kHz input, 25 ms windows, 10 ms hops, and 128 mel channels. These correspond to a 400-sample window and 160-sample hop.

An explanatory short-time Fourier transform and mel transform are:

$$S_{k,n}=\sum_{j=0}^{N-1}x[nH+j]w[j]e^{-2\pi i kj/N},
\qquad P_{k,n}=|S_{k,n}|^2,$$

$$M_{b,n}=\sum_kF_{b,k}P_{k,n},\qquad
Z_{b,n}=\log_{10}\max(M_{b,n},\epsilon).$$

x is the waveform, N=400 the window length, H=160 the hop, w the window, F the mel filter bank, and b a mel channel. The mel bank aggregates frequency energy with nonuniform spacing. The logarithm compresses its dynamic range.

Read the equations as a sequence of operations. For each small time window, S measures how much of each frequency is present. Squaring its magnitude gives energy P. A row of F collects energy from nearby frequencies into one mel band M. Taking the logarithm makes quiet and loud signals easier to represent on the same scale. The indices k and n mean frequency bin and time frame; j indexes samples inside a window, and i in the exponential is the imaginary unit. The small positive epsilon prevents taking the logarithm of zero.

For two seconds at 16,000 samples per second, the waveform contains 32,000 samples. A 160-sample hop moves forward by 0.01 seconds, giving about 200 frames. Each frame contains 128 numbers. The exact frame count depends on boundary padding and trimming, so the demonstration states those conventions rather than assuming every STFT returns exactly the same count.

These are teaching equations, not numbered Omni equations. The exact feature extractor also clips the log range and normalizes. Read [[code:WhisperFeatureExtractor._np_extract_fbank_features]] and its PyTorch counterpart. The NumPy demonstration below reproduces the important steps, with its boundary conventions stated.

#### Follow the audio encoder

[[code:Qwen2_5OmniAudioEncoder.__init__]] defines a first convolution mapping 128 channels to 1,280 with stride 1, then a second convolution with stride 2. It adds sinusoidal positions, applies 32 encoder layers, average-pools with kernel/stride 2, normalizes, and projects from 1,280 to 2,048.

The general convolution length is:

$$T_{\rm out}=\left\lfloor
\frac{T_{\rm in}+2p-d(k-1)-1}{s}+1\right\rfloor.$$

k is kernel size, p padding, d dilation, and s stride. This explains the integer expressions in [[code:Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths]].

For our 200-frame example, the first convolution preserves time length, the second reduces it to 100, and pooling reduces it to 50. Projection changes the width of each vector, not the number of vectors: (50,1280) becomes (50,2048). Convolutions learn local sound patterns; self-attention lets positions exchange information; pooling shortens the sequence; projection makes the result fit Thinker's interface.

The final representation stride is approximately 40 ms. That is not a 40 ms receptive field: each vector can encode surrounding information within its attention block.

#### Block attention is implemented through packing

In [[code:Qwen2_5OmniAudioEncoder.forward]], n_window=100 means input chunks contain n_window×2=200 mel frames, or two seconds. Read these operations:

1. Split valid audio into chunks, including a shorter final chunk.
2. Pad chunks for convolution and derive post-convolution masks.
3. Add local sinusoidal positions.
4. Remove padding and concatenate valid states.
5. Construct cumulative boundaries, cu_seqlens, for attention.
6. Reassemble each audio, pool, normalize, and project.

For example, cumulative boundaries [0,100,200] delimit two post-convolution blocks. They are not timestamps. In [[code:Qwen2_5OmniAudioAttention.forward]], inspect how the eager implementation creates a block-diagonal attention mask; compare the SDPA/FlashAttention alternatives.

Attention inside a two-second block is bidirectional. This supports block processing; it is not strictly causal sample-by-sample encoding. Thinker handles longer context after encoding.

The demonstration below starts with a simple mixture of 440 Hz and 880 Hz tones. Expect two bright frequency bands in the spectrogram. It teaches feature extraction, not Whisper's learned representations or recognition accuracy. No encoder weights are loaded.

#### Paper assignment and comprehension check

[[paper:2.2]] explains the representation; [[paper:2.4]] gives the blockwise motivation; [[paper:3]] describes initialization and training.

**Check.** Does this encoder produce Talker's speech tokens?

**Answer.** No. It produces continuous 2,048-dimensional understanding vectors. Talker predicts discrete output codec tokens.
