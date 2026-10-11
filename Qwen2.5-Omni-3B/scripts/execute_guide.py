"""Execute the notebook with a project-local kernel."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT/'.runtime'
runtime.mkdir(exist_ok=True)
kernel_dir=runtime/'kernels/qwen-reading'
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({
    'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Qwen reading guide','language':'python',
    'env':{'MPLCONFIGDIR':str(runtime/'matplotlib'),'IPYTHONDIR':str(runtime/'ipython')}
}))
os.environ['JUPYTER_PATH']=str(runtime)
os.environ['JUPYTER_RUNTIME_DIR']=str(runtime/'jupyter')
os.environ['IPYTHONDIR']=str(runtime/'ipython')
os.environ['MPLCONFIGDIR']=str(runtime/'matplotlib')
nb=nbformat.read(ROOT/'reading_guide.ipynb',as_version=4)
client=NotebookClient(nb,timeout=180,kernel_name='qwen-reading',
                      resources={'metadata':{'path':str(ROOT)}})
client.execute()
nbformat.write(nb,ROOT/'reading_guide.ipynb')
print('Executed reading_guide.ipynb.')
