"""Index archived code and render paper pages; no network access."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import fitz

ROOT = Path(__file__).resolve().parents[1]
TF_REV = 'cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d'

def main():
    manifest_path = ROOT / 'sources/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = {r['path']: r for r in manifest['sources']}
    index = {}
    for path in (ROOT / 'sources/transformers').rglob('*.py'):
        source = path.read_text()
        relative = str(path.relative_to(ROOT))
        upstream = str(path.relative_to(ROOT / 'sources/transformers'))
        tree = ast.parse(source)
        def walk(nodes, prefix=''):
            for n in nodes:
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + n.name
                    index[name] = dict(local_path=relative, start_line=n.lineno, end_line=n.end_lineno,
                        upstream_url=f'https://github.com/huggingface/transformers/blob/{TF_REV}/{upstream}#L{n.lineno}-L{n.end_lineno}')
                    walk(n.body, name + '.')
        if not path.name.startswith('modular_'):
            walk(tree.body)
    for path in (ROOT / 'sources').rglob('*'):
        if not path.is_file() or path == manifest_path or 'browse' in path.parts:
            continue
        relative = str(path.relative_to(ROOT))
        if relative not in records:
            # Derived artifacts preserve provenance through their parent archive.
            if relative.startswith('sources/paper/tex/'):
                records[relative] = dict(path=relative, url='https://arxiv.org/src/2503.20215v1',
                    revision='2503.20215v1', derived_from='sources/paper/qwen2.5-omni-2503.20215v1-source.tar')
            elif relative.startswith('sources/paper/page-'):
                records[relative] = dict(path=relative, url='https://arxiv.org/pdf/2503.20215v1',
                    revision='2503.20215v1', derived_from='sources/paper/qwen2.5-omni-2503.20215v1.pdf')
            else:
                continue
        data = path.read_bytes()
        record = records[relative]
        record.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), status='archived')
        record.setdefault('retrieved_utc', datetime.now(timezone.utc).isoformat())
        if 'blob_sha' in record:
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            record['local_git_blob_sha'] = actual
            record['upstream_byte_identical'] = actual == record['blob_sha']
    manifest['sources'] = list(records.values())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'sources/code_index.json').write_text(json.dumps(index, indent=2) + '\n')
    document = fitz.open(ROOT / 'sources/paper/qwen2.5-omni-2503.20215v1.pdf')
    output = ROOT / '.runtime/paper-preview'
    output.mkdir(parents=True, exist_ok=True)
    for page in [2,3,4,5,6]:
        document[page].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(output / f'page-{page+1}.png')
    print(f'Indexed {len(index)} symbols; rendered architecture/training PDF pages.')

if __name__ == '__main__':
    main()
