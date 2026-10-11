length,dim=12,8
Q,K,V=[rng.normal(size=(length,dim)) for _ in range(3)]
def softmax(x):
    ex=np.exp(x-x.max(axis=-1,keepdims=True))
    return ex/ex.sum(axis=-1,keepdims=True)
causal=np.tri(length,dtype=bool)
scores=Q@K.T/np.sqrt(dim)
full=softmax(np.where(causal,scores,-np.inf))@V
cached=[]
for i in range(length):
    cached.append(softmax(Q[i:i+1]@K[:i+1].T/np.sqrt(dim))@V[:i+1])
cached=np.concatenate(cached,axis=0)
results["cached_attention_error"]=float(np.max(np.abs(full-cached)))
assert results["cached_attention_error"]<1e-10
block_ids=np.arange(length)//4
blockwise=block_ids[:,None]==block_ids[None,:]
fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
for ax,mask,title in zip(axes,[blockwise,causal],["Encoder: bidirectional within each block","Thinker: causal sequence attention"]):
    ax.imshow(mask,origin="upper",cmap="Blues",vmin=0,vmax=1)
    ax.set(title=title,xlabel="Key position",ylabel="Query position",xticks=[0,4,8,11],yticks=[0,4,8,11])
finish_figure(fig,"attention_masks")
print("Synthetic cached/full attention max error:",results["cached_attention_error"])
print("Mask illustration uses 4 positions/block, not the real audio block length.")
