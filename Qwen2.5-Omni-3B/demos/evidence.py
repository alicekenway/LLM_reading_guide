from bs4 import BeautifulSoup
card=(ROOT/"sources/checkpoint/README.md").read_text()
soup=BeautifulSoup(card,"html.parser")
card_rows=[]
for tr in soup.find_all("tr"):
    cells=[td.get_text(" ",strip=True) for td in tr.find_all("td")]
    if len(cells)==2 and cells[0] in ["Qwen2.5-Omni-3B","Qwen2.5-Omni-7B"] and "%" in cells[1]:
        card_rows.append(cells)
scores={name:float(value.split("|")[-1].strip().strip("%")) for name,value in card_rows}
assert scores=={"Qwen2.5-Omni-3B":52.19,"Qwen2.5-Omni-7B":56.13}
fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
names=list(scores)
bars=ax.barh(["3B model card","7B paper / model card"],[scores[n] for n in names],
             color=[COLORS["vision"],"#b7c8d5"],edgecolor="#386d94",height=.55)
ax.bar_label(bars,labels=[f"{scores[n]:.2f}%" for n in names],padding=5)
ax.set(xlim=(0,65),xlabel="Reported OmniBench accuracy (%)",
       title="Reported results, not a reproduced evaluation")
ax.invert_yaxis()
finish_figure(fig,"omnibench")
results["omnibench_model_card"]=scores
