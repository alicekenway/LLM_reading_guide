### 2.4 TMRoPE: connect position with space and time

#### Why token order is insufficient

At one video frame per second and four frames per second, the same frame-index difference represents different durations. Audio produces approximately 25 final representations per second. A common time coordinate can relate the streams despite their different sampling rates.

There are two operations: interleave features in time chunks, and assign coordinates reflecting time/space. Neither operation replaces the other.

To understand the position mechanism, begin with ordinary text positions, then add the extra coordinates. The goal is not to attach a timestamp to the final answer. It is to change how attention compares input features, so relative location and time can affect what attends to what.

#### Start with ordinary RoPE

For a channel pair and position p:

$$R(p\omega)=
\begin{pmatrix}\cos(p\omega)&-\sin(p\omega)\\
\sin(p\omega)&\cos(p\omega)\end{pmatrix},
\quad q'_p=R(p\omega)q_p,\quad k'_r=R(r\omega)k_r.$$

Therefore:

$$ (q'_p)^\top k'_r=q_p^\top R((r-p)\omega)k_r. $$

Here q_p and k_r are a query and key channel pair at positions p and r. The frequency omega determines how quickly their angles change as position advances. R is a two-dimensional rotation matrix, and the transpose symbol means exchanging matrix rows and columns. Queries and keys use the same rotation rule; subtracting their angles makes their relative positions appear in their dot product.

As a small example, choose omega=pi/2. A query vector (1,0) at position 1 becomes (0,1); a key (1,0) at position 2 becomes (-1,0). Their dot product becomes zero even though the original vectors were identical. The rotation has supplied positional information without changing either vector's length. Real attention heads use many such pairs with different frequencies.

The dot product depends on relative position through an angle difference. Rotations also preserve vector norm. Different channel pairs use frequencies such as theta raised to -2j/d_head.

HF pairs first-half and second-half channels using rotate_half. That is equivalent to two-dimensional rotations after a permutation. Comparing it with an adjacent-pair implementation requires matching the layout.

#### Split rotary channels across three coordinates

Each token gets (t,h,w) coordinates:

$$ {\rm TMRoPE}(q;t,h,w)=R_t(t)q_t\oplus R_h(h)q_h\oplus R_w(w)q_w. $$

The subscripts t, h, and w mean temporal, height, and width channel sections. The vector q is split into those sections; each section rotates using its own coordinate. The symbol ⊕ means combining channel sections. This is a conceptual teaching equation; the exact channel layout is in the source.

Thinker's mrope_section is [16,24,24], summing to half its 128-dimensional head. In [[code:apply_multimodal_rotary_pos_emb]], Python list repetition by 2 gives [16,24,24,16,24,24]. It does not independently double each entry. Temporal/height/width cosine and sine values are selected section by section.

[[code:Qwen2_5OmniRotaryEmbedding.forward]] produces cosine/sine tensors shaped (3,B,L,d_head). The application function selects sections and broadcasts them over attention heads, then rotates queries and keys.

#### Modality conventions

| Modality | Coordinate convention |
|---|---|
| Text | (p,p,p), reducing to ordinary RoPE |
| Audio | The same increasing coordinate on all axes, about 40 ms per step |
| Image | Constant temporal coordinate; varying merged-grid height/width |
| Video | Time coordinates derived from seconds; varying spatial coordinates |

With 25 position IDs per second, relative video time s becomes approximately floor(25s). One-second grid spacing gives 0,25,50,…; two-second spacing gives 0,50,100,… . Integer conversion and sequence offsets follow the code.

For a tiny 2×2 merged image grid, the coordinates can be (0,0,0), (0,0,1), (0,1,0), and (0,1,1), before sequence offsets. All four share an image time, but retain distinct row/column addresses. A video grid at 2 seconds can use temporal ID 50 on the same 25-ID-per-second scale as audio. This makes physical time comparable even though that frame contributes several spatial tokens.

“Absolute time” here means clip-relative physical time with a sequence offset, not wall-clock time. A subsequent segment starts after the preceding maximum coordinate. It does not necessarily start after the number of preceding flattened tokens.

#### Interleaving and position construction

[[code:Qwen2_5OmniProcessor.replace_multimodal_special_tokens]] places visual placeholders before audio placeholders within each two-second chunk. [[code:Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index]] assigns positions in that same order.

Read its branches in increasing complexity: text only; standalone audio; image; video without audio; video with audio. Then examine rope_deltas, which makes position bookkeeping compatible with cached generation.

Look closely at start/end markers, removed padding, max()+1 offsets, and helper calls. These are necessary to implement the conceptual formula correctly.

A system using variable frame rates must preserve timing through media preparation. Position encoding cannot recover timestamps that a loader discarded.

#### Paper assignment and comprehension check

[[paper:2.2]] and Figure 3 are the main assignment. Compute a tiny image/video grid, then return to the figure. The paper explains TMRoPE mostly in prose and diagrams; the equations here make that account explicit.

**Check.** Must a text coordinate after 64 image tokens advance by 64?

**Answer.** No. The code advances from the maximum spatial/temporal coordinate plus one, including control-token bookkeeping.


The demonstration checks that equal coordinates on all three axes recover text RoPE and that rotations preserve vector norms. Its timeline then shows the intended audio/video arrangement; the note after the demonstration separates a historical implementation issue from that intended design.

#### Implementation note: a discrepancy in this historical snapshot

The processor's chunk helper receives a one-dimensional time-index array. The model's similarly named helper receives a tensor shaped (3,N), but loops while i is less than len(token_indices), which is 3 rather than N. This is a real snapshot-specific implementation issue, not the intended TMRoPE algorithm.

The included weight-free check executes the archived function bodies. For 100 audio positions and chunk size 50, the processor returns two chunks, while the model helper returns one. On a synthetic four-second audiovisual sequence, the first tokens of two visual grids receive temporal coordinates [2,48], whereas the intended chunk ordering gives [2,52]. A visual grid with only two tokens can also trigger an IndexError.

These observations are limited to the pinned historical commit. The archived upstream files remain byte-identical; this guide does not silently patch them or assert that every later Transformers release has the same behavior. The small correct-coordinate examples below teach the paper's design. The source-check results separately expose the observed implementation discrepancy.

Read [[code:Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_chunked_index]] beside [[code:Qwen2_5OmniProcessor.get_chunked_index]], then inspect the saved check results. This is why comparing shapes and execution with the paper is valuable.

