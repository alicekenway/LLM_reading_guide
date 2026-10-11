def rotate_half_np(x):
    half=x.shape[-1]//2
    return np.concatenate([-x[...,half:],x[...,:half]],axis=-1)

def rotary_np(q,position_ids,sections=(16,24,24),theta=1e6):
    # q: (sequence, head_dim); IDs: (3, sequence), split-half HF layout.
    dim=q.shape[-1]
    inv=theta**(-np.arange(0,dim,2)/dim)
    frequencies=position_ids[...,None]*inv[None,None,:]
    angles=np.concatenate([frequencies,frequencies],axis=-1)
    cuts=np.cumsum(sections*2)[:-1]
    pieces=np.split(angles,cuts,axis=-1)
    selected=np.concatenate([piece[i%3] for i,piece in enumerate(pieces)],axis=-1)
    return q*np.cos(selected)+rotate_half_np(q)*np.sin(selected)

q=rng.normal(size=(6,128))
text_ids=np.tile(np.arange(6),(3,1))
rotated=rotary_np(q,text_ids)
inv=1e6**(-np.arange(0,128,2)/128)
ordinary_angles=np.concatenate([np.arange(6)[:,None]*inv]*2,axis=-1)
ordinary=q*np.cos(ordinary_angles)+rotate_half_np(q)*np.sin(ordinary_angles)
results["tmrope_text_matches_rope"]=bool(np.allclose(rotated,ordinary))
results["rope_norm_error"]=float(np.max(np.abs(np.linalg.norm(q,axis=-1)-np.linalg.norm(rotated,axis=-1))))
assert results["tmrope_text_matches_rope"] and results["rope_norm_error"]<1e-10
tiny_grid=np.array([[0,0,0,0],[0,0,1,1],[0,1,0,1]])
print("2×2 merged image coordinates, rows t/h/w:\n",tiny_grid)
audio_times=np.arange(100)/25
video_times=np.array([0,1,2,3.0])
fig,axes=plt.subplots(2,1,figsize=(10,5),layout="constrained")
axes[0].scatter(audio_times,np.zeros_like(audio_times),s=10,color=COLORS["audio"],label="Audio: 25 features/s")
axes[0].scatter(video_times,np.ones_like(video_times),marker="s",s=80,color=COLORS["vision"],label="Illustrative video grid: 1/s")
axes[0].axvline(2,color="#555",ls="--")
axes[0].set(yticks=[0,1],yticklabels=["Audio","Video"],xlabel="Physical input time (s)",
            title="A shared time axis: temporal ID ≈ floor(25 × seconds)")
axes[0].legend(loc="upper right")
segments=[("Video 0–2 s",4,COLORS["vision"]),("Audio 0–2 s",50,COLORS["audio"]),
          ("Video 2–4 s",4,COLORS["vision"]),("Audio 2–4 s",50,COLORS["audio"])]
offset=0
for label,width,color in segments:
    axes[1].barh(0,width,left=offset,color=color,height=.5)
    axes[1].text(offset+width/2,.37,label,ha="center",fontsize=9,rotation=25)
    offset+=width
axes[1].set(ylim=(-.7,1),yticks=[],xlabel="Flattened feature-token index (illustrative counts)",
            title="Intended processor sequence: video first, then audio, in each 2 s chunk")
finish_figure(fig,"time_alignment")
hf_checks=json.loads((ROOT/"hf_mechanism_checks.json").read_text())
display(Markdown("**Actual archived-HF check:** intended video coordinates " +
                 str(hf_checks["audiovisual_expected_temporal_positions"]) + "; observed " +
                 str(hf_checks["audiovisual_actual_temporal_positions"]) +
                 ". See the implementation note below; the timeline illustrates intended design."))
