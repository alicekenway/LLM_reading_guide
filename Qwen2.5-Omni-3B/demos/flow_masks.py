blocks=np.arange(8)
diff=blocks[None,:]-blocks[:,None]
local=diff==0
back=(diff>=-1)&(diff<=0)
ahead=(diff>=0)&(diff<=1)
def compose(a,b):
    return (a.astype(int)@b.astype(int))>0
reach=compose(back,compose(ahead,back))
assert reach[3].nonzero()[0].tolist()==[1,2,3,4]
fig,axes=plt.subplots(1,4,figsize=(12,3.6),layout="constrained")
for ax,mask,title in zip(axes,[local,back,ahead,reach],["Most layers: current","Layers 0,20: lookback","Layer 10: lookahead","Composed: -2 to +1"]):
    ax.imshow(mask,cmap="Blues",vmin=0,vmax=1)
    ax.set(title=title,xlabel="Key block",ylabel="Query block",xticks=[0,3,7],yticks=[0,3,7])
finish_figure(fig,"dit_receptive_field")
display(Markdown("These are block-level attention dependencies. Other layers preserve within-block access; conditioning is separate."))
