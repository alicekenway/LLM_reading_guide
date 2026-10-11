"""Archive official source inputs; no model weights are downloaded."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HF_REV = 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
TF_REV = 'cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d'

def download(item):
    path, url, revision = item
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'paper-code-reading-guide/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    target.write_bytes(data)
    return dict(path=path, url=url, revision=revision,
                retrieved_utc=datetime.now(timezone.utc).isoformat(),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), status='archived')

def main():
    jobs = [
        ('sources/paper/qwen2.5-omni-2503.20215v1.pdf', 'https://arxiv.org/pdf/2503.20215v1', '2503.20215v1'),
        ('sources/paper/qwen2.5-omni-2503.20215v1-source.tar', 'https://arxiv.org/src/2503.20215v1', '2503.20215v1'),
    ]
    for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'generation_config.json', 'README.md', 'LICENSE']:
        jobs.append((f'sources/checkpoint/{name}', f'https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/{HF_REV}/{name}', HF_REV))
    for name, ident in [('whisper','2212.04356'), ('qwen2-audio','2407.10759'), ('qwen2.5-vl','2502.13923'), ('flow-matching','2210.02747'), ('bigvgan','2206.04658'), ('dpo','2305.18290')]:
        jobs.append((f'sources/references/{name}-{ident}v1.pdf', f'https://arxiv.org/pdf/{ident}v1', f'{ident}v1'))
    manifest_path = ROOT / 'sources/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'sources': []}
    sources = {s['path']: s for s in manifest['sources']}
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(download, item): item for item in jobs}
        for future in as_completed(pending):
            item = pending[future]
            try:
                record = future.result()
                print('Archived', record['path'], record['bytes'], flush=True)
            except Exception as exc:
                record = dict(path=item[0], url=item[1], revision=item[2], status='unavailable', error=str(exc))
                print('Unavailable', item[0], str(exc), flush=True)
            sources[item[0]] = record
    manifest.update(model='Qwen/Qwen2.5-Omni-3B', checkpoint_revision=HF_REV,
                    transformers_revision=TF_REV, paper_revision='2503.20215v1',
                    sources=list(sources.values()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
