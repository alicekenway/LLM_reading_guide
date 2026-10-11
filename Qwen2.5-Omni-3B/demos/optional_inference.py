if RUN_FULL_INFERENCE:
    # Install optional dependencies first; see README. This cell downloads weights.
    socket.socket.connect = _original_connect
    import torch
    import soundfile as sf
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    if not torch.cuda.is_available():
        raise RuntimeError("These optional full-model examples require a suitably sized CUDA GPU.")
    model_id="Qwen/Qwen2.5-Omni-3B"
    revision=manifest["checkpoint_revision"]
    processor=Qwen2_5OmniProcessor.from_pretrained(model_id,revision=revision)
    model=Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,revision=revision,torch_dtype=torch.bfloat16,
        device_map="auto",attn_implementation="sdpa")
    def infer(messages,return_audio=False,use_audio_in_video=False):
        prompt=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        audios,images,videos=process_mm_info(messages,use_audio_in_video=use_audio_in_video)
        inputs=processor(text=prompt,audio=audios,images=images,videos=videos,
                         return_tensors="pt",padding=True,use_audio_in_video=use_audio_in_video)
        inputs=inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            output=model.generate(**inputs,return_audio=return_audio,
                                  use_audio_in_video=use_audio_in_video,thinker_max_new_tokens=128)
        ids,waveform=(output if return_audio else (output,None))
        response=processor.batch_decode(ids[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        print(response[0])
        if waveform is not None:
            sf.write("optional_response.wav",waveform.reshape(-1).cpu().numpy(),24000)
            print("Saved optional_response.wav at 24 kHz.")
        return response,waveform
    infer([{"role":"user","content":[{"type":"text","text":"Explain why the sky appears blue in two sentences."}]}])
else:
    print("Optional full-model inference skipped: RUN_FULL_INFERENCE=False.")
