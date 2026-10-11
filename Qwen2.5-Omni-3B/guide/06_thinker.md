### 2.5 Thinker: reason over the shared sequence

Thinker is the point where the separate input paths become one reasoning context. The earlier components do not independently produce the final answer: they provide vectors that Thinker's language-model layers can compare. In our example, answering what the speaker meant may require attending to both the audio and the red bowl.

The checkpoint table below is read directly from config.json. Thinker's text backbone is 2,048 wide with 36 layers, 16 query heads, and 2 KV heads. Each head is 128 wide. Talker's input vectors also start at width 2,048, but its decoder is 896 wide. “3B” identifies the language-backbone scale; it does not mean all audio, vision, speech, and reconstruction modules together total exactly three billion parameters. The demonstration calculates the language-backbone count separately and draws the actual component boundaries.

#### Follow one forward pass

[[code:Qwen2_5OmniThinkerForConditionalGeneration.forward]] first derives valid audio lengths and packs audio features. It computes position IDs on prefill, or derives continuation positions from cache position and rope_deltas.

Next it obtains text embeddings shaped (B,L,2048). On prefill, the audio tower and vision encoder return feature matrices of the same final width. Boolean masks identify audio/image/video token locations, and masked_scatter replaces the corresponding embedding elements.

The central operation is conceptually:

$$e_i=\begin{cases}
E_{\rm text}(x_i),&\text{text/control position},\\
f_{\rm audio}(X)_j,&\text{audio placeholder},\\
f_{\rm vision}(I)_j,&\text{visual placeholder}.
\end{cases}$$

The cases say which vector occupies position i. E_text is the learned text embedding table; x_i is a text/control token ID. The functions f_audio and f_vision denote the corresponding encoders, and j selects the matching media vector. B is batch size and L is the expanded input sequence length. All three cases must have the same final width of 2,048 to form one tensor.

The feature index j follows the processor's ordering. This is input fusion into a shared sequence; it is not a separate decoder cross-attention layer to every modality.

The function then calls the text Transformer, applies a language head, and optionally computes loss. The head returns logits: one raw score per possible output token. Softmax converts scores into probabilities before token selection or loss calculation. During single-token cached decoding it avoids repeatedly encoding the original media.

#### Attention shapes and grouped-query attention

[[code:Qwen2_5OmniAttention.forward]] maps hidden states into queries, keys, and values. For this Thinker:

- Queries: (B,16,L,128).
- Keys and values before repetition: (B,2,L,128).
- Each KV head is shared by 8 query heads.

After TMRoPE and cache update, repeat_kv makes keys/values compatible with the 16 query heads. The attention computation is:

$$A={\rm softmax}\left(\frac{Q'K'^\top}{\sqrt{128}}+M\right),\qquad O=AV.$$

Q' and K' are queries and keys after position rotation; V contains the values to be combined. Each entry of Q'K'^T scores one query/key pair. Dividing by the square root of head width keeps scores from growing simply because there are more channels. Softmax turns the permitted scores into weights that sum to one; AV forms a weighted average of value vectors. The mask M prevents reading positions that should not be available.

For intuition, suppose two permitted positions have scores 0 and log(3). Softmax gives weights 1/4 and 3/4. If their scalar values are 2 and 10, the attention output is 0.25×2+0.75×10=8. Real heads perform this operation on vectors, and the model learns the query/key/value projections.

M masks future positions and padding. Eager attention uses an additive mask with large negative masked values; boolean masks in PyTorch SDPA have different conventions, so inspect the backend rather than copying a mask indiscriminately.

The output is combined across heads and projected back to width 2,048. [[code:Qwen2_5OmniDecoderLayer.forward]] wraps attention and an MLP with residual paths and normalization.

Grouped-query attention reduces the KV cache compared with storing 16 independent KV heads. An approximate cache element count for B sequences of length L over N layers is:

$$2BNLH_{\rm KV}d_{\rm head}.$$

H_KV is the number of key/value heads, d_head is each head’s width, and N is the number of layers. B and L again mean batch size and cached sequence length. The leading 2 represents K and V. This excludes weights, activations, and other modules; it is not a complete GPU-memory estimate.

#### Prefill versus decoding

Prefill processes the expanded prompt. Decoding processes a new token while reusing prior K/V states. [[code:Qwen2_5OmniThinkerForConditionalGeneration.prepare_inputs_for_generation]] slices inputs and stops resending media features after prefill. The underlying GenerationMixin handles the autoregressive loop and sampling.

TMRoPE coordinates can differ from a flattened token counter, so continuation uses rope_deltas. Dropping that offset while retaining the KV cache can create inconsistent positions.

#### Training and next-token loss

With prompt/context X and answer Y:

$$p_\theta(Y\mid X)=\prod_{j=1}^{|Y|}
p_\theta(y_j\mid X,y_{<j}),\qquad
\mathcal L_{\rm CE}=-\sum_{j\in\mathcal S}\log p_\theta(y_j\mid X,y_{<j}).$$

Here theta denotes the model parameters; Y is the answer token sequence and X the input context. The product expresses generating one token after another. The negative sum of log-probabilities is the cross-entropy objective: it penalizes assigning low probability to the correct next token. For a single target token, probability 0.8 gives loss about 0.223, while probability 0.2 gives about 1.609. During training, gradient updates encourage the correct token to receive more probability.

The sum above shows the objective before reduction. The HF utility normally averages valid token losses; when num_items_in_batch is supplied, it divides the summed loss by that count. The supervised-position set S excludes labels marked -100. Teacher forcing supplies previous ground-truth tokens. The loss implementation shifts predictions and labels for next-token alignment. The precise dataset labeling policy is a training-data decision, not something to infer merely from the model forward pass.

[[code:Qwen2_5OmniThinkerForConditionalGeneration.forward]] delegates to the configured loss function; [[code:ForCausalLMLoss]] shows the shift and how -100 labels are ignored. Greedy decoding chooses argmax logits; sampling draws from a transformed distribution. Neither changes the trained architecture.

The attention demonstration compares processing a complete synthetic sequence with processing its continuation using cached keys/values. The outputs should agree up to numerical precision. This illustrates why caching can avoid repeating old computations while preserving the causal attention result; it does not measure the pretrained model's answers or runtime.

#### Paper assignment

[[paper:2.1]] identifies the decoder and encoders. [[paper:2.3]] explains text generation. The paper assumes standard language-model machinery; the attention and caching details are most directly learned from the source.

**Check.** Are multimodal features re-encoded once for every generated word?

**Answer.** Not in this cached path. They are inserted during prefill; decoding reuses the resulting cache.
