"""Render the plain-language architecture used in Part 1, without model weights."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def render():
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    ax.set(xlim=(0, 11), ylim=(0, 5))
    ax.axis('off')
    boxes = [(.2,2.4,2.1,1.4,'Text, images,\naudio and video','#386d94'),
             (3.0,2.4,2.2,1.4,'Thinker\nUnderstand and answer','#687b36'),
             (6.0,2.4,2.0,1.4,'Talker\nChoose speech codes','#9c547c'),
             (8.7,2.4,2.0,1.4,'Token2Wav\nMake audible speech','#b45c24')]
    for x,y,w,h,text,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',
                                   facecolor='white',edgecolor=color,linewidth=1.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#253047')
    for start,end in [(2.4,2.9),(5.3,5.9),(8.1,8.6)]:
        ax.annotate('',xy=(end,3.1),xytext=(start,3.1),
                    arrowprops=dict(arrowstyle='->',lw=1.6,color='#5b6574'))
    ax.annotate('Text answer',xy=(4.1,3.95),xytext=(4.1,4.65),ha='center',fontsize=12,
                arrowprops=dict(arrowstyle='<-',color='#687b36'))
    ax.text(1.25,1.65,'Encoders turn media\ninto useful vectors',ha='center',fontsize=11)
    ax.text(4.1,.65,'TMRoPE + time chunks\nconnect what happened together',ha='center',fontsize=11)
    ax.text(5.6,1.65,'Text information\n+ contextual states',ha='center',fontsize=10)
    ax.text(9.7,1.65,'Spectrogram\nthen waveform',ha='center',fontsize=11)
    ax.set_title('The division of work: understand → speak → reconstruct sound',fontsize=14,pad=15)
    (ROOT/'figures').mkdir(exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(ROOT/'figures'/f'idea_overview.{extension}',bbox_inches='tight',dpi=140)
    plt.close(fig)

if __name__=='__main__':
    render()
