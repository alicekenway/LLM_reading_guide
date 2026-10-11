"""Build the study notebook and exact paper/code map from authored chapters."""
from pathlib import Path
import csv
import json
import re
import nbformat as nbf
from render_overview import render

ROOT = Path(__file__).resolve().parents[1]
PAPER_PAGES = {'1': '2–3', '2.1':'3', '2.2':'4', '2.3':'4–5', '2.4':'5',
               '3':'6','4.1':'6','4.2':'6','4.3':'7','5.1':'7–12','5.2':'12–13'}
CHAPTER_SECTIONS = {1:'1;2.1',2:'2.2;4.1',3:'2.2;2.4;3',4:'2.2;2.4',
                    5:'2.2',6:'2.1;2.3',7:'2.3',8:'2.4',9:'3;4.3',10:'5.1;5.2',11:'2.1;2.2;2.3;2.4'}
index=json.loads((ROOT/'sources/code_index.json').read_text())
cells=[]
rows={}
render()

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

def demo(name):
    code((ROOT/'demos'/f'{name}.py').read_text())

def resolve(text, chapter):
    def paper(match):
        section=match.group(1)
        pages=PAPER_PAGES[section]
        page=pages.split('–')[0]
        return f'[Paper §{section}, PDF pp. {pages}](sources/paper/qwen2.5-omni-2503.20215v1.pdf#page={page}).'
    def source(match):
        name=match.group(1)
        item=index[name]
        rows[name]=dict(chapter={2:'2.1',3:'2.2',4:'2.3',5:'2.4',6:'2.5',7:'2.6',8:'2.7',9:'2.8',11:'2.9'}.get(chapter,str(chapter)),paper_section=CHAPTER_SECTIONS[chapter],
            paper_pages=';'.join(PAPER_PAGES.get(s,'3–5') for s in CHAPTER_SECTIONS[chapter].split(';')),
            paper_equation_or_figure=({5:'Figure 3',8:'Figure 4'}.get(chapter,'')),
            code_symbol=name,local_path=item['local_path'],start_line=item['start_line'],
            end_line=item['end_line'],upstream_url=item['upstream_url'],status='released',
            notes='Source symbol verified by AST; section association is a reading assignment.')
        local=item['local_path']
        return f'[{name}]({local}) (lines {item["start_line"]}–{item["end_line"]})'
    return re.sub(r'\[\[code:(.*?)\]\]',source,re.sub(r'\[\[paper:(.*?)\]\]',paper,text))

# Reading references belong to the explanation, never to isolated link cells.
ADDITIONAL_READINGS = {
    2:['Qwen2_5OmniProcessor.__call__','Qwen2_5OmniProcessor.replace_multimodal_special_tokens'],
    3:['Qwen2_5OmniAudioEncoder.__init__','Qwen2_5OmniAudioEncoder.forward'],
    4:['Qwen2_5OmniVisionEncoder.forward','Qwen2_5OmniPatchMerger.forward'],
    5:['apply_multimodal_rotary_pos_emb','Qwen2_5OmniPreTrainedModelForConditionalGeneration.get_chunked_index'],
    6:['Qwen2_5OmniForConditionalGeneration.__init__','Qwen2_5OmniAttention.forward','ForCausalLMLoss'],
    7:['Qwen2_5OmniForConditionalGeneration.generate','Qwen2_5OmniTalkerForConditionalGeneration.forward'],
    8:['Qwen2_5OmniToken2WavDiTModel.sample','DiTDecoderLayer.forward','Qwen2_5OmniToken2WavBigVGANModel.forward'],
    9:['Qwen2_5OmniTalkerForConditionalGeneration.forward']}


def chapter_text(filename, chapter):
    text=(ROOT/'guide'/filename).read_text()
    missing=[symbol for symbol in ADDITIONAL_READINGS.get(chapter,[]) if '[[code:'+symbol+']]' not in text]
    if missing:
        text+='\n\n**Additional implementation reading:** '+'; '.join('[[code:'+symbol+']]' for symbol in missing)+'.\n'
    paths = sorted({index[symbol]['local_path'] for symbol in re.findall(r'\[\[code:(.*?)\]\]',text)})
    text+='\n\n**Original files for this section:**\n\n'+'\n'.join(f'- [{path}]({path})' for path in paths)+'\n'
    return resolve(text,chapter)


def paper_figure(filename, number):
    return (f'\n\n![Original paper Figure {number}](sources/paper/tex/figures/{filename})\n\n'
            f'Figure {number}, Qwen Team, *Qwen2.5-Omni Technical Report* v1, reproduced under CC BY 4.0. '
            'Use the explanation above to interpret the original diagram.\n')

md("# Qwen2.5-Omni-3B: understand the idea, then follow the implementation\n\n"
   "This notebook has two parts. **[Part 1](#part-1)** explains the problem, earlier methods, the proposed solution, "
   "and the results. **[Part 2](#part-2)** teaches the components, formulas, and original implementation step by step. "
   "Basic ML, attention, LLM, and PyTorch knowledge is assumed. You can read all saved outputs without running cells.\n\n"
   "References open local PDFs or original Python files. For code, use the named symbol and line range in your editor; "
   "in VS Code, Ctrl+G jumps to a line. The [paper/code map](paper_code_map.csv) collects those locations.")
ideas=resolve((ROOT/'guide/01_ideas.md').read_text(),10)
md('<a id="part-1"></a>\n\n'+ideas)
md('<a id="part-2"></a>\n\n## Part 2 — Understand each component\n\n'
   'We now follow the data: prepare inputs, encode sound and vision, attach useful positions, generate an answer, '
   'and turn it into speech. Training comes after the mechanisms it trains. The final walkthrough connects the stages.\n\n'
   '**Notation and execution.** A tensor shape lists its axis lengths; for example, (B,L,d) means B conversations, '
   'L token positions, and d numbers per vector. Keep sequence length separate from vector width. '
   'The formulas are explanatory derivations unless explicitly identified as numbered paper equations. '
   'Default demonstrations run on CPU, offline, with synthetic inputs and no model weights. Run the setup below '
   'once before the demonstrations; full GPU inference stays disabled.\n\n'
   '**Sources.** The [paper PDF](sources/paper/qwen2.5-omni-2503.20215v1.pdf) is arXiv:2503.20215v1. '
   'Transformers is pinned to cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d; checkpoint metadata is pinned to '
   'f75b40e3da2003cdd6e1829b1f420ca70797c34e. These are the historical sources, not moving branches. '
   'See the [manifest](sources/manifest.json) for provenance and checksums.\n\n'
   '**Component route:** [Inputs](#component-1) · [Audio](#component-2) · [Vision](#component-3) · '
   '[TMRoPE](#component-4) · [Thinker](#component-5) · [Talker](#component-6) · '
   '[Token2Wav](#component-7) · [Training](#component-8) · [Complete walkthrough](#component-9)')
demo('setup')
components=[(2,'02_inputs.md','inputs'),(3,'03_audio.md','audio'),(4,'04_vision.md','vision'),
            (5,'05_tmrope.md','tmrope'),(6,'06_thinker.md','attention'),(7,'07_talker.md','talker'),
            (8,'08_waveform.md',None),(9,'09_training.md','dpo'),(11,'11_reading_route.md',None)]
for number,(chapter,filename,demonstration) in enumerate(components,1):
    text=chapter_text(filename,chapter)
    text='<a id="component-'+str(number)+'"></a>\n\n'+text
    if chapter==5:
        marker='#### Implementation note: a discrepancy in this historical snapshot'
        main,note=text.split(marker,1)
        md(main+paper_figure('ATMRoPE.png',3))
        demo('tmrope')
        md(marker+note)
    elif chapter==6:
        intro,body=text.split('#### Follow one forward pass',1)
        md(intro)
        demo('orientation')
        md('#### Follow one forward pass'+body)
        demo('attention')
    elif chapter==8:
        main,rest=text.split('#### Streaming masks: inspect each layer',1)
        streaming,vocoder=rest.split('#### BigVGAN: reconstruct samples from mel',1)
        md(main)
        demo('flow_transport')
        md('#### Streaming masks: inspect each layer'+streaming+paper_figure('attentionmask.png',4))
        demo('flow_masks')
        md('#### BigVGAN: reconstruct samples from mel'+vocoder)
    else:
        md(text)
        if demonstration:
            demo(demonstration)
    if chapter==11:
        demo('optional_inference')
        demo('optional_media')
md('### Reproduction notes\n\nThe final cell saves the numerical checks from the demonstrations. '
   'These verify stated mathematical properties, not pretrained model quality. '
   'The separate [original-function checks](hf_mechanism_checks.json) execute selected archived HF bodies '
   'on CPU with PyTorch 2.3.1 and no weights. Full pretrained inference was not run. '
   'See [README.md](README.md) for setup/rebuild commands and [VALIDATION.md](VALIDATION.md) for validation limits.')
demo('save_checks')
nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3 (reading guide)','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},
    'guide_sources':{'paper':'2503.20215v1','transformers':'cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d',
                     'checkpoint':'f75b40e3da2003cdd6e1829b1f420ca70797c34e'}})
nbf.validate(nb)
nbf.write(nb,ROOT/'reading_guide.ipynb')

rows['DPO training']=dict(chapter='2.8',paper_section='4.3',paper_pages='7',paper_equation_or_figure='Equation (1)',
    code_symbol='DPO training',local_path='',start_line='',end_line='',upstream_url='',
    status='not_released',notes='No complete DPO trainer in this archived inference implementation.')
rows['speech tokenizer training']=dict(chapter='2.7',paper_section='2.3',paper_pages='5',paper_equation_or_figure='',
    code_symbol='speech tokenizer training',local_path='',start_line='',end_line='',upstream_url='',
    status='not_released',notes='The exact codec encoder architecture and training recipe are not fully specified.')
with (ROOT/'paper_code_map.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(next(iter(rows.values())).keys()))
    writer.writeheader()
    writer.writerows(rows.values())
print(f'Built {len(cells)} cells and {len(rows)} paper/code entries.')
