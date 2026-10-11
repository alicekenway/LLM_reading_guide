beta=.1
margins=np.linspace(-20,20,201)
losses=np.logaddexp(0,-beta*margins)
results["baseline_dpo_loss"]=float(np.logaddexp(0,0))
results["preferred_dpo_loss"]=float(np.logaddexp(0,-beta*5))
assert results["preferred_dpo_loss"]<results["baseline_dpo_loss"]
fig,ax=plt.subplots(figsize=(8,4.2),layout="constrained")
ax.plot(margins,losses,color=COLORS["speech"],lw=2)
ax.axvline(0,color="#666",ls="--",lw=1)
ax.scatter([0,5],[results["baseline_dpo_loss"],results["preferred_dpo_loss"]],color=COLORS["speech"])
ax.set(title="DPO pair loss for β = 0.1",xlabel="Preferred minus rejected reference-relative log-probability margin",
       ylabel="Loss = softplus(−β × margin)")
finish_figure(fig,"dpo_loss")
print("Equal-to-reference policy:",results["baseline_dpo_loss"])
print("Preferred margin +5:",results["preferred_dpo_loss"])
