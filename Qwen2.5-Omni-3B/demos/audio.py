sample_rate = 16000
duration = 2.0
t = np.arange(int(sample_rate*duration)) / sample_rate
wave = 0.4*np.sin(2*np.pi*440*t) + 0.2*np.sin(2*np.pi*880*t)
n_fft, hop = 400,160
centered = np.pad(wave,(n_fft//2,n_fft//2),mode="reflect")
windows = np.lib.stride_tricks.sliding_window_view(centered,n_fft)[::hop]
window = np.hanning(n_fft+1)[:-1]  # periodic Hann
power = np.abs(np.fft.rfft(windows*window,axis=-1))**2
power = power[:-1].T  # Whisper drops the last centered frame

def hz_to_mel(hz):
    hz=np.asarray(hz,dtype=float)
    return np.where(hz<1000,hz/(200/3),15+np.log(np.maximum(hz,1e-12)/1000)/(np.log(6.4)/27))

def mel_to_hz(mel):
    mel=np.asarray(mel,dtype=float)
    return np.where(mel<15,mel*(200/3),1000*np.exp((mel-15)*np.log(6.4)/27))

frequencies=np.fft.rfftfreq(n_fft,1/sample_rate)
edges=mel_to_hz(np.linspace(0,hz_to_mel(sample_rate/2),130))
lower=(frequencies[:,None]-edges[:-2])/(edges[1:-1]-edges[:-2])
upper=(edges[2:]-frequencies[:,None])/(edges[2:]-edges[1:-1])
filters=np.maximum(0,np.minimum(lower,upper))* (2/(edges[2:]-edges[:-2]))
mel=filters.T@power
log_mel=np.log10(np.maximum(mel,1e-10))
features=(np.maximum(log_mel,log_mel.max()-8)+4)/4
assert features.shape==(128,200)
print("Synthetic signal:",wave.shape,"→ features:",features.shape,"→ 50 audio tokens")
fig,axes=plt.subplots(2,1,figsize=(10,6),layout="constrained")
axes[0].plot(t[:800],wave[:800],color=COLORS["audio"])
axes[0].set(title="Synthetic 440 Hz + 880 Hz waveform (first 50 ms)",xlabel="Time (s)",ylabel="Amplitude")
im=axes[1].imshow(features,origin="lower",aspect="auto",extent=(0,2,0,128),cmap="magma")
axes[1].set(title="128-bin normalized log-mel features",xlabel="Time (s)",ylabel="Mel bin")
fig.colorbar(im,ax=axes[1],label="Normalized log energy")
finish_figure(fig,"audio_features")
display(Markdown("The peaks reflect the two synthetic tones. This is a feature calculation, not a speech-recognition result."))
