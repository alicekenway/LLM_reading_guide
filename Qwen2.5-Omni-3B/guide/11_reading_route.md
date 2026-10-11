### 2.9 Complete walkthrough: connect the stages

Return to the cooking example from Part 1. To make the counts easy to follow, take one already resized 224×224 frame showing the red bowl, two seconds of audio, and a text question. This is a simplified image-plus-audio request; a true video repeats the visual grids over time. The walkthrough explains data flow, not an answer produced by downloaded weights.

**Prepare the request.** The chat template supplies roles and modality markers. The processor turns the frame into a patch grid (1,16,16) and the recording into 200 valid mel frames. It expands the image marker to 64 positions and the audio marker to 50 positions. The text question and control markers add their own positions. Read [[code:Qwen2_5OmniProcessor.__call__]] and its placeholder-expansion helper to connect the user-facing message with those counts.

**Encode the media.** Vision starts with 256 raw patch vectors, attends over them, and merges them into 64 vectors of width 2,048. Audio applies convolutions, block attention, pooling, and projection to return 50 vectors of the same width. The numbers 64 and 50 are lengths; 2,048 is the width of every vector. This distinction explains why the two modalities can occupy the same language-model sequence despite having different rates.

| Boundary | Representation for this example | What changes here? |
|---|---|---|
| Prepared image | 256 patches, each with 1,176 pixel values | Pixels become patch inputs |
| Vision output | (64,2048) | Attention and learned merging produce model vectors |
| Prepared audio | (128,200), before any padding | Samples become time/frequency features |
| Audio output | (50,2048) | Encoding and downsampling produce model vectors |
| Thinker input | (1,L,2048) | 114 media locations are filled, alongside text/control positions |
| Thinker output | Text IDs plus selected hidden states | The shared context becomes an answer and conditioning |
| Talker continuation | Width 2048 → 896; 8448 codec logits per step | Conditioning and prior speech code predict another code |
| Token2Wav | Codec IDs → 80-bin mel → 24 kHz samples | Discrete symbols become playable sound |

**Assign positions and reason.** Thinker inserts the media vectors into the reserved positions. Position construction gives image locations spatial coordinates and audio positions an increasing time coordinate. With video, the processor additionally interleaves visual/audio chunks and the position code uses physical time. Read [[code:Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_rope_index]], then the modality-replacement operations in [[code:Qwen2_5OmniThinkerForConditionalGeneration.forward]]. You should now recognize how the representation and coordinate paths meet.

Thinker's prefill processes the assembled context. Its next-token logits begin the answer; later steps reuse the KV cache. An illustrative answer might be “Put the chopped vegetables in the red bowl; the timer is ringing.” That sentence is here to connect the responsibilities, not to claim a pretrained inference result. Actual words are represented by tokenizer IDs, which need not correspond one-to-one with words.

**Generate speech.** In [[code:Qwen2_5OmniForConditionalGeneration.generate]], trace the selected Thinker hidden states and embedding-layer information into Talker conditioning. Follow the speaker and codec boundary tokens, then the Talker call. Talker predicts speech codes rather than waveform samples. The wrapper removes terminal/prefix codes before calling Token2Wav. Read [[code:Qwen2_5OmniToken2WavModel.forward]] to see the final sampler-to-vocoder connection.

For an output segment of 50 ordinary codec tokens, the nominal reconstruction rates give 100 mel frames and about 24,000 samples, or one second at 24 kHz. Input and output durations are not required to match: a two-second question can have a longer spoken answer. These nominal rates do not replace the actual padding, stopping, and slicing rules in the implementation.

**Revisit the paper.** [[paper:2.1]] and Figure 2 provide the architecture map; [[paper:2.2]] covers representations and positions; [[paper:2.3]] covers generation; [[paper:2.4]] covers blockwise operation. On this second reading, ask where each arrow in the paper corresponds to a tensor boundary in the table above. The [paper/code map](paper_code_map.csv) provides the complete reference list.

#### Check your understanding

- **What changes if the prepared image is 448×448?** There are (448/14)²=1,024 raw spatial patches and 256 merged vectors, assuming one temporal grid step. The vector width remains 2,048.
- **Why retain both text information and contextual states?** Exact linguistic identity and contextual meaning provide different information for pronunciation and prosody.
- **Why can two-second audio blocks support longer conversations?** They provide local representations that Thinker's longer sequence can combine. The encoder's local attention and the language model's context length are different boundaries.
- **Does one-block lookahead mean no delay?** No. That future context must be available before the dependency can be resolved.
- **Where is the full DPO trainer?** It is not in this archived inference implementation. The paper supplies the objective; the guide marks the release gap.
- **What if audio/video positions disagree?** Inspect timestamps, processor ordering, placeholder counts, and position construction. Part 2.4 separately documents the historical helper discrepancy.

#### Optional full-model inference

You have enough information to read the model without running it. The following cells are disabled by default and were not used to validate pretrained outputs. They require a separate CUDA environment, the pinned Transformers revision, compatible PyTorch/torchvision, accelerate, librosa, soundfile, and qwen-omni-utils. Use the setup commands in [README.md](README.md).

The archived model card reports a BF16 estimate of 18.38 GB for its 15-second-video setting and notes that practical memory usage can be higher. Treat this as a source-reported measurement, not a guarantee. Resolution, sequence length, cache, backend, and speech output all affect memory. No model weights were downloaded to write this guide.

Provide a local image, audio, or short video path before enabling these cells. Use the same audio-in-video setting across preparation and generation. The speech wrapper supports a single item. Decode only response tokens after the prompt and save audio at 24 kHz. Prefer short audiovisual examples while studying the historical position-helper issue; moving to another release requires new source references and compatibility checks.
