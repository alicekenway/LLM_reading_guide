noise=rng.normal(size=(80,24))
time_axis=np.linspace(0,1,24)
mel_axis=np.arange(80)[:,None]
target=np.exp(-((mel_axis-(22+6*np.sin(2*np.pi*time_axis)))/7)**2)*2
velocity=target-noise
state=noise.copy()
trajectory=[state.copy()]
# Oracle linear-path velocity: no neural network is trained in this demonstration.
for _ in range(10):
    h=.1
    k1=velocity;k2=velocity;k3=velocity;k4=velocity
    state=state+h*(k1+2*k2+2*k3+k4)/6
    trajectory.append(state.copy())
results["flow_endpoint_error"]=float(np.max(np.abs(state-target)))
assert results["flow_endpoint_error"]<1e-10
fig,axes=plt.subplots(1,3,figsize=(11,4),layout="constrained")
for ax,array,title in zip(axes,[trajectory[0],trajectory[5],trajectory[-1]],["t = 0: Gaussian noise","t = 0.5: transport midpoint","t = 1: synthetic mel target"]):
    ax.imshow(array,origin="lower",aspect="auto",cmap="magma",vmin=-2,vmax=2)
    ax.set(title=title,xlabel="Audio frame",ylabel="Mel bin")
finish_figure(fig,"flow_matching")
print("Oracle-path RK4 endpoint error:",results["flow_endpoint_error"])

