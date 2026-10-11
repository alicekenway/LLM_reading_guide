if RUN_FULL_INFERENCE:
    IMAGE_PATH = ""  # Set local paths before running.
    AUDIO_PATH = ""
    VIDEO_PATH = ""  # Prefer ≤2 s for this historical audiovisual snapshot.
    for kind,path,prompt in [
        ("image",IMAGE_PATH,"Describe the image."),
        ("audio",AUDIO_PATH,"Describe the sounds you hear."),
        ("video",VIDEO_PATH,"Describe the visible actions and the sounds.")]:
        if not path:
            print("Skipped",kind,"example: no local media path supplied.")
            continue
        assert Path(path).is_file(),path
        messages=[{"role":"user","content":[{"type":kind,kind:path},{"type":"text","text":prompt}]}]
        infer(messages,return_audio=False,use_audio_in_video=(kind=="video"))
    # Speech output is independent of supplying a media path.
    infer([{"role":"user","content":[{"type":"text","text":"Say hello and introduce yourself briefly."}]}],
          return_audio=True)
else:
    print("Optional image/audio/video and speech-output examples skipped.")
