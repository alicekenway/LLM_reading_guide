"""Validate the delivered notebook, archived sources, and reading references."""
from pathlib import Path
import ast
import hashlib
import json
import csv
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def validate():
    notebook = ROOT / 'reading_guide.ipynb'
    assert notebook.exists(), 'The reading notebook has not been built.'
    import nbformat
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    assert code_cells and all(c.execution_count is not None for c in code_cells), 'Unexecuted code cell'
    errors = [o for c in code_cells for o in c.get('outputs', []) if o.output_type == 'error']
    assert not errors, errors
    manifest = json.loads((ROOT / 'sources/manifest.json').read_text())
    archived = [s for s in manifest['sources'] if s.get('status') == 'archived']
    for source in archived:
        data = (ROOT / source['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source['sha256'], source['path']
    assert (ROOT / 'sources/paper/qwen2.5-omni-2503.20215v1.pdf').read_bytes().startswith(b'%PDF')
    rows = list(csv.DictReader((ROOT / 'paper_code_map.csv').open()))
    assert len(rows) >= 25, 'Missing mechanism references'
    for row in rows:
        if row['status'] == 'not_released':
            assert not row['local_path'] and row['notes']
            continue
        text = (ROOT / row['local_path']).read_text()
        tree = ast.parse(text)
        line = int(row['start_line'])
        assert any(n.lineno == line for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))), row
        assert row['upstream_url'].startswith('https://github.com/huggingface/transformers/blob/' + manifest['transformers_revision'])
    body = '\n'.join(c.source for c in nb.cells)
    headings = re.findall(r'^## (.+)$', body, re.MULTILINE)
    assert headings == ['Part 1 — Understand the idea', 'Part 2 — Understand each component'], headings
    assert len(nb.cells) <= 45, 'The guide has become fragmented again'
    part_two = next(i for i,c in enumerate(nb.cells) if '## Part 2 —' in c.source)
    assert all(c.cell_type == 'markdown' for c in nb.cells[:part_two]), 'Part 1 must read without code cells'
    ideas = '\n'.join(c.source for c in nb.cells[:part_two])
    assert re.findall(r'^### (\d)\. ', ideas, re.MULTILINE) == ['1','2','3','4']
    assert re.findall(r'^### 2\.(\d) ', body, re.MULTILINE) == list('123456789')
    assert '### Read the original implementation' not in body
    for c in nb.cells:
        if c.cell_type == 'markdown':
            prose = re.sub(r'<a[^>]*></a>', '', c.source)
            prose = re.sub(r'^#+ .*$', '', prose, flags=re.MULTILINE).strip()
            assert prose, 'Heading-only Markdown cell'
    assert not re.search(r'\[\[code:|\[\[paper:', body), 'Unresolved source reference'
    results = json.loads((ROOT / 'demo_results.json').read_text())
    assert results['audio_tokens_2s'] == 50
    assert results['image_tokens_224'] == 64
    assert results['tmrope_text_matches_rope']
    assert results['rope_norm_error'] < 1e-10
    assert results['flow_endpoint_error'] < 1e-10
    assert results['preferred_dpo_loss'] < results['baseline_dpo_loss']
    assert results['cached_attention_error'] < 1e-10
    # Part 1's static chart must still agree with the pinned model card.
    from bs4 import BeautifulSoup
    card = BeautifulSoup((ROOT/'sources/checkpoint/README.md').read_text(),'html.parser')
    scores = {}
    for tr in card.find_all('tr'):
        row = [td.get_text(' ',strip=True) for td in tr.find_all('td')]
        if len(row)==2 and row[0] in ('Qwen2.5-Omni-3B','Qwen2.5-Omni-7B') and '%' in row[1]:
            scores[row[0]] = float(row[1].split('|')[-1].strip().strip('%'))
    assert scores == {'Qwen2.5-Omni-3B':52.19,'Qwen2.5-Omni-7B':56.13}
    chart = (ROOT/'figures/omnibench.svg').read_text()
    assert all(f'{score:.2f}%' in chart for score in scores.values())
    images = [o for c in code_cells for o in c.get('outputs', []) if o.output_type in ('display_data', 'execute_result') and 'image/png' in o.get('data', {})]
    assert len(images) >= 7, 'Expected rendered demonstrations'
    assert 'show_code(' not in body and 'sources/browse/' not in body
    assert not (ROOT / 'reading_guide.html').exists()
    figures = set()
    for cell in nb.cells:
        if cell.cell_type != 'markdown':
            continue
        for target in re.findall(r'\]\(([^)]+)\)', cell.source):
            if target.startswith(('https://', 'http://', '#')):
                continue
            path = unquote(target.split('#')[0])
            assert (ROOT / path).exists(), target
            assert not path.endswith('.html'), target
        figures.update(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', cell.source))
    for source in archived:
        if 'blob_sha' in source:
            data = (ROOT/source['path']).read_bytes()
            blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert blob == source['blob_sha'], source['path']
    print(f'PASS: two parts, {len(nb.cells)} cells, {len(code_cells)} executed code cells, '
          f'{len(images)+len(figures)} figures ({len(images)} outputs, {len(figures)} inline), '
          f'{len(archived)} archived sources, {len(rows)} paper/code references.')

if __name__ == '__main__':
    validate()
