from pathlib import Path
import json
import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, Markdown, Image

ROOT = Path.cwd()
assert (ROOT / "sources/manifest.json").exists(), "Run from the guide directory."
RUN_FULL_INFERENCE = False
_original_connect = socket.socket.connect

def _offline_connect(self, address):
    if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError("Default study cells run offline; full inference is explicitly optional.")
    return _original_connect(self, address)

if not RUN_FULL_INFERENCE:
    socket.socket.connect = _offline_connect

plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#253047", "text.color": "#253047"})
COLORS = dict(audio="#b45c24", vision="#386d94", text="#687b36", speech="#9c547c")
rng = np.random.default_rng(7)
results = {}
index = json.loads((ROOT / "sources/code_index.json").read_text())
config = json.loads((ROOT / "sources/checkpoint/config.json").read_text())
manifest = json.loads((ROOT / "sources/manifest.json").read_text())
(ROOT / "figures").mkdir(exist_ok=True)

def finish_figure(fig, name):
    fig.savefig(ROOT / "figures" / (name + ".png"), bbox_inches="tight")
    fig.savefig(ROOT / "figures" / (name + ".svg"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

print("CPU demonstrations; no pretrained weights; external network disabled.")
print("Paper:", manifest["paper_revision"])
print("Transformers:", manifest["transformers_revision"])
