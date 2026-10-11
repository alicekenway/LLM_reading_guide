def audio_lengths(valid_mel_frames):
    after_conv = (np.asarray(valid_mel_frames) - 1) // 2 + 1
    after_pool = (after_conv - 2) // 2 + 1
    return after_conv, after_pool

frames = np.array([4,5,7,8,199,200,201,400])
conv_lengths, token_lengths = audio_lengths(frames)
display(Markdown("| Valid mel frames | After stride-2 conv | Final audio tokens |\n"
                 "|---:|---:|---:|\n" +
                 "\n".join(f"| {a} | {b} | {c} |" for a,b,c in zip(frames,conv_lengths,token_lengths))))
assert token_lengths.tolist() == [1,1,2,2,50,50,50,100]
results["audio_tokens_2s"] = int(audio_lengths(200)[1])
grid = np.array([1,16,16])
results["image_tokens_224"] = int(grid.prod() // 4)
print("224×224 image: grid",grid.tolist(),"→",results["image_tokens_224"],"merged tokens")
display(Markdown("**Read the original chat template:** [tokenizer_config.json](sources/checkpoint/tokenizer_config.json), field `chat_template`. Inspect the modality markers and role delimiters in your editor."))
