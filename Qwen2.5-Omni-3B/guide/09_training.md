### 2.8 Training: teach the components to work together

So far we have followed inference: what happens when a trained model receives a request. Training asks a different question: how do the encoders, reasoning model, and speech modules learn to work together? Initialization provides useful starting capabilities; alignment connects their representations; joint training teaches the combined tasks; post-training improves conversational behavior and speech stability.

#### Pretraining stages

[[paper:3]] describes:

1. Initialize the LLM from Qwen2.5, the audio encoder from Whisper-large-v3, and use the Qwen2.5-VL vision encoder. Keep the LLM fixed while first training adapters, then encoders, to align features with the language space.
2. Unfreeze parameters and train with mixed text, image, video, audio, and audiovisual data. The report cites additional 800B image/video-related tokens, 300B audio-related tokens, and 100B audiovisual-related tokens. Treat these as reported family training quantities, not an independently reproduced accounting for the 3B checkpoint.
3. Extend training sequence length from 8,192 to 32,768 using longer audio/video and other inputs.

Freezing the LLM initially gives the encoders a stable target space. Joint training then allows the system to adapt across modalities. Longer-context training addresses a different problem: learning to use extended sequences.

These motivations are interpretations of the training schedule. They do not establish missing optimizer settings, learning rates, exact loss weights, or curriculum proportions.

#### Post-training Thinker and Talker

[[paper:4.1]] and [[paper:4.2]] describe ChatML instruction data containing pure-text and multimodal conversations.

[[paper:4.3]] describes Talker continuation training, preference optimization, and speaker fine-tuning. Next-token supervision teaches speech continuation; preferences improve generation stability; speaker tuning specializes voices. The report mentions timbre disentanglement but does not give enough detail to implement its full objective.

#### Explain DPO, the paper's Equation (1)

For input x, preferred speech sequence y_w, rejected speech sequence y_l, policy P_theta, reference policy P_ref, and scale beta:

$$\mathcal L_{\rm DPO}
=-\mathbb E_{(x,y_w,y_l)}
\log\sigma\left[
\beta\left(
\log\frac{P_\theta(y_w\mid x)}{P_{\rm ref}(y_w\mid x)}
-\log\frac{P_\theta(y_l\mid x)}{P_{\rm ref}(y_l\mid x)}
\right)\right].$$

The sigmoid sigma turns a real-valued preference score into a value between zero and one; the negative logarithm rewards making the preferred outcome more likely in that comparison. beta controls the scale of the reference-relative score. The expectation averages over prompt/preferred/rejected examples.

This is **Equation (1) in Omni §4.3**, on PDF page 7. Sequence log-probabilities are sums of autoregressive token log-probabilities:

$$\log P_\theta(y\mid x)=\sum_j\log P_\theta(y_j\mid x,y_{<j}).$$

Define the reference-relative margin:

$$m=\left[\log P_\theta(y_w\mid x)-\log P_{\rm ref}(y_w\mid x)\right]
-\left[\log P_\theta(y_l\mid x)-\log P_{\rm ref}(y_l\mid x)\right].$$

Then the pair loss is softplus(-beta m). If policy equals reference, m=0 and the loss is log 2. Increasing the preferred sequence's relative log-probability lowers the loss. A fixed reference anchors the comparison. For example, with beta=0.1, margin m=0 gives about 0.693 loss; margin m=5 gives about 0.474. The preferred sequence has improved relative to the rejected sequence and to the fixed reference. The demonstration plots that relationship rather than optimizing a real Talker.

The report constructs preferences using word-error rate and punctuation/pause errors. Lower loss does not by itself prove naturalness or identify the exact reward/data implementation. The original [DPO paper](sources/references/dpo-2305.18290v1.pdf) explains the objective's connection to preference-based policy optimization.

#### Paper and implementation reading

Thinker's forward path exposes a language-model loss. This inference port does not include the complete pretraining curriculum, speech-tokenizer training, preference-data construction, or a DPO trainer.

[[code:Qwen2_5OmniTalkerForConditionalGeneration.forward]] sets loss to None. Read [[code:ForCausalLMLoss]] for the available text next-token loss, rather than mistaking it for a speech DPO trainer. Therefore DPO has a paper/theory mapping here and an explicit **release gap**, not a fabricated pointer to an inference function presented as its trainer.

**Check.** Can a released forward pass alone reproduce the reported training?

**Answer.** No. You also need data, preprocessing/labeling, initialization details, objectives, schedules, and optimization settings, some of which are not fully released.
