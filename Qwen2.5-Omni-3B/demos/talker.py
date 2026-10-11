# Scaled dimensions make the addition/projection easy to inspect.
contextual=rng.normal(size=(1,3,8))
text_information=rng.normal(size=(1,3,8))
codec_embedding=rng.normal(size=(1,3,8))
conditioning=contextual+text_information
projection=rng.normal(size=(8,4))
talker_input=(conditioning+codec_embedding)@projection
print("Scaled demonstration:",conditioning.shape,"+ codec",codec_embedding.shape,"→",talker_input.shape)
print("Actual continuation dimensions: (B,1,2048) → projection → (B,1,896) → 8448 codec logits.")
print("The random vectors illustrate the operation; they do not generate speech.")
fig,ax=plt.subplots(figsize=(10,2.8))
ax.axis("off");ax.set(xlim=(0,10),ylim=(0,2.5))
for x,title,color in [(0.3,"1. Thinker.generate\nreturns text + states",COLORS["text"]),
                       (3.7,"2. Talker.generate\nreturns codec tokens",COLORS["speech"]),
                       (7.1,"3. Token2Wav\nreturns waveform",COLORS["audio"])]:
    ax.add_patch(FancyBboxPatch((x,.7),2.5,1.1,boxstyle="round,pad=.12",facecolor="white",edgecolor=color,lw=1.6))
    ax.text(x+1.25,1.25,title,ha="center",va="center")
for x in [2.9,6.3]:
    ax.annotate("",xy=(x+.65,1.25),xytext=(x,1.25),arrowprops=dict(arrowstyle="->",lw=1.5))
ax.set_title("Actual call order in the pinned HF wrapper",pad=14)
finish_figure(fig,"wrapper_order")
