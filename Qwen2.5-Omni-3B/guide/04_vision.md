### 2.3 Vision encoder: turn images and frames into vectors

The audio encoder turns time windows into vectors. The vision encoder does the corresponding job for small regions of frames. A whole image collapsed into one vector would make fine spatial relationships difficult to preserve; patch tokens keep separate locations available until learned merging reduces their number.

For the red bowl, the useful information may include shape, color, nearby objects, and a pointing hand. The architecture provides representations from which those relationships can be learned. Patchification alone does not recognize a bowl.

#### Structure and training lineage

The encoder is inherited from [Qwen2.5-VL](sources/references/qwen2.5-vl-2502.13923v1.pdf). It is a Vision Transformer with temporal patchification, rotary positions, mostly spatial-window attention, selected full-attention layers, and a patch merger.

Each Transformer block applies normalized self-attention and an MLP with residual connections. For language alignment, image/text targets can backpropagate a language-model token loss into visual features. Omni describes first fixing the LLM while training adapters/encoders, then joint multimodal training. This does not establish a new Omni-specific classification or contrastive objective. Consult the VL paper's architecture/training sections for its own recipe, and Omni §3 for the disclosed adaptation.

#### Patch embedding is a 3-D convolution

[[code:Qwen2_5_VisionPatchEmbed.forward]] reshapes patch vectors into (N_patches,3,2,14,14), then applies a convolution with kernel and stride (2,14,14).

For already valid frame dimensions:

$$T_g=T_{\rm frames}/2,\quad H_g=H/14,\quad W_g=W/14,
\qquad N_p=T_gH_gW_g.$$

T_frames is the number of prepared frames; H and W are their pixel height and width. T_g, H_g, and W_g count patch-grid positions, and N_p is the total number of raw patches. Each patch starts as 3×2×14×14=1,176 numbers: three color channels across two frames and a 14×14 region. A learned linear operation maps that region to a 1,280-dimensional vector.

The processor first resizes/pads to valid multiples. It also arranges patches so 2×2 neighborhoods are contiguous for merging. An arbitrary reshape of row-major RGB pixels does not necessarily reproduce that ordering.

An image is duplicated into two identical frames, yielding one temporal patch. Video uses pairs of actual frames, allowing temporal patches to represent motion. Follow the processor's temporal padding for odd frame counts.

#### Window and full attention

[[code:Qwen2_5OmniVisionEncoder.get_window_index]] creates a patch order grouped by windows. [[code:Qwen2_5OmniVisionEncoder.forward]] applies that order to both states and rotary positions. Its layer loop chooses window or full-attention cumulative boundaries.

Most layers attend within spatial windows. Layers identified by fullatt_block_indexes attend across one temporal patch's spatial grid. The boundaries are built per temporal grid step; “full” does not mean unrestricted attention across an entire long video.

Dense attention on N tokens involves N² relationships. For n_w windows with W tokens each, window attention involves roughly n_w W² relationships. For example, 256 tokens give 65,536 possible query/key pairs. Four windows of 64 tokens give 16,384 pairs. This illustrates the difference in attention relationships; projection, MLP, and data movement still cost work, so it is not a measured fourfold speedup.

#### Four patch vectors become one

[[code:Qwen2_5OmniPatchMerger.forward]] normalizes patch vectors, concatenates each 2×2 neighborhood, and applies an MLP:

$$v_{\rm merged}=W_2\,{\rm GELU}\!\left(
W_1[{\rm Norm}(v_1);\ldots;{\rm Norm}(v_4)]+b_1\right)+b_2.$$

Read the merger from inside out. Normalize each input vector, join four vectors, apply a learned matrix W_1 and bias b_1, pass through the GELU activation, then apply W_2 and b_2. For 1,280-wide inputs, concatenation has width 5,120; the final output has width 2,048. The learned matrices can retain useful combinations instead of weighting every patch equally.

The semicolon means concatenation. Output width is 2,048, matching Thinker. This is learned merging, not average pooling.

After merging, the inverse permutation restores grid order. This is essential because the processor and Thinker's placeholder replacement expect the original order.

Return to the 224×224 image: one temporal grid and 16×16 spatial patches give 256 raw patch vectors. Merging 2×2 groups leaves an 8×8 grid, or 64 vectors. The demonstration colors these groups so you can see which four patches become neighbors. It illustrates geometry and ordering, without applying a trained vision encoder.

#### Paper assignment

[[paper:2.2]] gives the inherited encoder and image duplication; [[paper:2.4]] describes patch size and merging. The source supplies windowing and ordering details beyond the brief paper account.

**Check.** Does duplicating a 224×224 image produce 128 Thinker visual tokens?

**Answer.** No. Two frames form one temporal patch. The 16×16 spatial patches merge to 8×8 vectors: 64 tokens.
