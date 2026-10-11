text_cfg = config["thinker_config"]["text_config"]
audio_cfg = config["thinker_config"]["audio_config"]
vision_cfg = config["thinker_config"]["vision_config"]
talk_cfg = config["talker_config"]
display(Markdown(
    "| Component | Width | Layers | Q / KV heads |\n|---|---:|---:|---:|\n"
    f"| Thinker text | {text_cfg['hidden_size']} | {text_cfg['num_hidden_layers']} | "
    f"{text_cfg['num_attention_heads']} / {text_cfg['num_key_value_heads']} |\n"
    f"| Audio encoder | {audio_cfg['d_model']} → {audio_cfg['output_dim']} | "
    f"{audio_cfg['encoder_layers']} | {audio_cfg['encoder_attention_heads']} / full |\n"
    f"| Vision encoder | {vision_cfg['hidden_size']} → {vision_cfg['out_hidden_size']} | "
    f"{vision_cfg['depth']} | {vision_cfg['num_heads']} / full |\n"
    f"| Talker | {talk_cfg['embedding_size']} → {talk_cfg['hidden_size']} | "
    f"{talk_cfg['num_hidden_layers']} | {talk_cfg['num_attention_heads']} / {talk_cfg['num_key_value_heads']} |"))
d = text_cfg["hidden_size"]
f = text_cfg["intermediate_size"]
kv_width = text_cfg["num_key_value_heads"] * (d // text_cfg["num_attention_heads"])
per_layer = 2*d*d + 2*d*kv_width + 3*d*f + 2*d + d + 2*kv_width
backbone_parameters = text_cfg["num_hidden_layers"] * per_layer + d + 2*text_cfg["vocab_size"]*d
print(f"Backbone parameter calculation (untied input/output embeddings): {backbone_parameters:,}")
print("This excludes audio/vision encoders, Talker, and Token2Wav.")

fig, ax = plt.subplots(figsize=(12,7))
ax.set(xlim=(0,12), ylim=(0,8))
ax.axis("off")
def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",
                              facecolor="white",edgecolor=color,linewidth=1.7))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=10)
def arrow(a,b,label=None):
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",color="#5b6574",lw=1.5))
    if label:
        ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+0.16,label,ha="center",fontsize=9)
box(.3,6.2,2.0,.8,"Waveform\n16 kHz",COLORS["audio"])
box(3.3,6.2,2.5,.8,"128-bin mel\nAudio encoder: 1,280 → 2,048",COLORS["audio"])
box(.3,4.8,2.0,.8,"Image / video\nRGB frames",COLORS["vision"])
box(3.3,4.8,2.5,.8,"Patchify → ViT → merger\n1,280 → 2,048",COLORS["vision"])
box(.3,3.3,2.0,.8,"Text\nBPE tokens",COLORS["text"])
box(3.3,3.3,2.5,.8,"Text embeddings\n2,048",COLORS["text"])
box(7.1,4.6,2.1,1.3,"Thinker\n36-layer decoder\nTMRoPE\nshared sequence",COLORS["text"])
box(9.8,4.8,1.8,.8,"Text IDs",COLORS["text"])
box(7.1,2.4,2.1,1.0,"Talker\n2,048 → 896\n8,448 codec logits",COLORS["speech"])
box(7.1,.5,2.1,1.0,"Flow-matching DiT\ncodec → 80-bin mel",COLORS["speech"])
box(9.8,.5,1.8,1.0,"BigVGAN\n24 kHz speech",COLORS["speech"])
for a,b in [((2.4,6.6),(3.15,6.6)),((2.4,5.2),(3.15,5.2)),((2.4,3.7),(3.15,3.7)),
            ((5.95,6.6),(7.0,5.65)),((5.95,5.2),(7.0,5.2)),((5.95,3.7),(7.0,4.85)),
            ((9.35,5.2),(9.65,5.2)),((8.15,4.45),(8.15,3.55)),
            ((8.15,2.25),(8.15,1.65)),((9.35,1),(9.65,1))]:
    arrow(a,b)
ax.text(8.5,3.95,"hidden states + text information",fontsize=9)
ax.set_title("Qwen2.5-Omni-3B: representations and model boundaries",pad=12)
finish_figure(fig,"architecture")
