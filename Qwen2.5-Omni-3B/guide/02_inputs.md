### 2.1 Input preparation: reserve the right places for media

#### Problem and mechanism

Before looking inside a neural network, ask what reaches it. In our cooking example, the user supplies a sentence, frames, and sound. The language model cannot consume an image file or a WAV filename directly. The processor makes a coordinated package: token locations that say where each modality belongs, plus the numerical media data that will fill those locations.

Think of placeholders as reserved seats. Reserving one seat for an image that later produces 64 vectors would leave 63 vectors with nowhere to go. The processor therefore needs to know the encoder's output count before Thinker assembles its input.

A user supplies a conversation with media, whereas a Transformer consumes a sequence of vectors. The processor must produce two things that agree: a token sequence with modality placeholders, and feature tensors whose encoded vectors replace those placeholders.

A placeholder identifies a position in the language-model sequence; it is not the raw image or waveform. If an image produces 64 merged vectors, the input needs 64 image-token locations.

#### Follow the interfaces

The chat template turns roles and media markers into text. Modality processors compute audio features and image/video patches. The Omni processor expands markers, tokenizes the expanded text, and combines the tensors.

| Field | Meaning | Typical shape |
|---|---|---|
| input_ids | Text/control/modality IDs | (B,L) |
| attention_mask | Valid language-model positions | (B,L) |
| input_features | Padded audio log-mel features | (N_audio,128,T_pad) |
| feature_attention_mask | Valid audio feature frames | (N_audio,T_pad) |
| pixel_values | Flattened image patches | (N_raw_patches,3×2×14×14) |
| image_grid_thw | Pre-merging image patch grids | (N_images,3) |
| pixel_values_videos | Flattened video patches | Same patch-vector convention |
| video_grid_thw | Video patch grids | (N_videos,3) |
| video_second_per_grid | Seconds per temporal grid step | One value per video |

B is the conversation batch size; L is the expanded sequence length. The number of media inputs need not equal B.

In [[code:Qwen2_5OmniProcessor.__call__]], follow audio extraction and mask renaming, then image/video grids, then placeholder expansion and tokenization. Video spacing is computed as temporal patch size divided by frames per second.

[[code:Qwen2_5OmniProcessor.replace_multimodal_special_tokens]] expands each marker to its encoded feature count. In the video-with-audio branch, both modalities are expanded in a common chunk order. The same audio-in-video setting must be used in media preparation, processing, and generation.

#### Shapes must agree before attention

For an already resized 224×224 image, patch size 14 gives a 16×16 spatial grid. Duplicating the image into two frames and using temporal patch size 2 yields **one temporal grid step**. Merging 2×2 patches gives:

$$N_{\rm visual}=\frac{T_gH_gW_g}{m^2}
=\frac{1\cdot16\cdot16}{4}=64.$$

Here T_g, H_g, and W_g are the temporal, height, and width lengths of the patch grid; m=2 is the spatial merge factor. N_visual counts vectors, not individual pixels. There are 256 raw patches in this example, but four adjacent patches become each final vector.

The processor inserts 64 image placeholders; the encoder returns 64 vectors of Thinker width.

For audio, valid mel-frame length T_m determines:

$$T_c=\left\lfloor\frac{T_m-1}{2}\right\rfloor+1,\qquad
T_a=\left\lfloor\frac{T_c-2}{2}\right\rfloor+1.$$

The floor symbol means round down to an integer. T_m is the number of valid spectrogram frames, not the padded tensor length. T_c is the stride-2 convolution output length, and T_a the length after pooling. For 200 frames, these are 100 and 50. Boundary rounding matters; do not replace the integer expressions with T_m/4 everywhere.

The following demonstration varies audio frame counts to show the rounding, then checks the image count. It is useful to predict the result for 199 versus 200 frames before running it. Matching counts is the first correctness condition; matching the order of the vectors will matter in the next components.

#### Paper and source assignment

[[paper:2.2]] describes representations; [[paper:4.1]] gives the conversation format. Its video examples are conceptual data examples; executable prompts must use the actual archived chat template.

[[code:ProcessorMixin.apply_chat_template]] is the inherited template boundary. [[code:Qwen2VLImageProcessor._preprocess]] arranges pixels into patches. A convenience media-loading helper is not the neural encoder.

**Check.** Why are the audio and language-model masks different?

**Answer.** They mask different sequences: spectrogram frames versus expanded language-model positions.
