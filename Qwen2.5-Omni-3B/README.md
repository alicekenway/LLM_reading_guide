# Qwen2.5-Omni-3B reading guide

Read [the executed Jupyter notebook](reading_guide.ipynb). It is the sole reading guide.
The guide assumes basic ML/LLM/PyTorch knowledge and explains multimodal and speech components in detail.

## Included

- Two connected parts: idea understanding, then detailed component teaching.
- Part 1 follows four questions: the problem, existing methods and limitations, the proposed method, and the results. It can be read without running code.
- Part 2 follows inputs → audio → vision → TMRoPE → Thinker → Talker → Token2Wav, then training and a complete walkthrough. Intuition, defined formulas, worked examples, and paper/code references stay together.
- Equations with symbols/dimensions, worked examples, comprehension checks, and exact paper/code references.
- CPU demonstrations with saved outputs and figures; optional full inference cells are disabled.
- [Paper PDF](sources/paper/qwen2.5-omni-2503.20215v1.pdf), source material for figure provenance, six original component papers, and byte-identical relevant HF source files.
- [Paper/code map](paper_code_map.csv), [source manifest](sources/manifest.json), and numerical/source check results.
- Direct links to original local Python files, with symbols and line ranges. Implementation excerpts and chat templates are not copied into notebook outputs.

## Reading in WSL

Open this folder in VS Code connected to WSL using `code .`, then open `reading_guide.ipynb`. The saved outputs can be read without running cells. To run demonstrations, select `.venv/bin/python` as the kernel.

For implementation reading, follow a notebook link to the original Python file, or open the listed path in the VS Code Explorer. Use Ctrl+G to jump to the listed line number and read the named symbol. Relative file links may behave differently across notebook frontends; the visible path and line range provide an editor-independent fallback. Paper links use the local PDF and specify the relevant PDF pages.

## Source versions and important distinctions

Paper: **arXiv:2503.20215v1**, 20 PDF pages, mainly 7B evaluation.

Transformers: **cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d**, resolving the model-card-recommended historical v4.51.3-Qwen2.5-Omni-preview reference.

Checkpoint metadata: **f75b40e3da2003cdd6e1829b1f420ca70797c34e**.

The notebook distinguishes the paper's design from this historical implementation. In particular:

- The HF wrapper calls Thinker, then Talker, then Token2Wav; it does not expose simultaneous chunk delivery.
- A weight-free reproduction identifies a multi-chunk audiovisual position-order discrepancy in this snapshot's model chunk helper. It loops over len(3,N)=3 instead of N; very small visual grids can also raise IndexError. See Part 2.4 and [the exact check results](hf_mechanism_checks.json). Upstream files are preserved unchanged.
- Talker's inference forward pass does not implement a training loss or DPO trainer.
- Speech-tokenizer training and several training-recipe details are not fully specified/released.
- Paper benchmark numbers are not automatically 3B numbers.

## Rebuild and validate

Use Python 3.12 and an isolated environment:

    python -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python scripts/prepare_sources.py
    .venv/bin/python scripts/build_guide.py
    .venv/bin/python scripts/execute_guide.py
    .venv/bin/python scripts/validate_guide.py

The delivered source archive is sufficient for those commands; default notebook execution needs neither downloads nor pretrained weights. Jupyter does require permission to open local loopback sockets for its kernel.

The default notebook uses NumPy/Matplotlib. Source-level checks additionally need PyTorch:

    python scripts/check_hf_mechanisms.py

Run that command in a PyTorch environment. It was checked on CPU with PyTorch 2.3.1+cu121 and NumPy 1.26.4, loading no weights. Selected original function bodies are extracted by AST; only type annotations/decorators are removed for isolated execution.

To refresh the version-pinned downloads, with network access:

    .venv/bin/python scripts/download_sources.py

This downloader retrieves metadata and papers. The included Transformers archive is already pinned and checksum-verified; the downloader does not fetch model weights.

## Optional full-model GPU examples

Full inference was **not executed** while writing this guide. Default validation covers educational calculations and selected upstream functions, not pretrained-model outputs.

Create a separate environment to avoid changing the notebook environment:

    python -m venv .venv-hf
    .venv-hf/bin/python -m pip install 'numpy<2' 'torch>=2.3' torchvision accelerate librosa soundfile qwen-omni-utils ipykernel matplotlib beautifulsoup4
    .venv-hf/bin/python -m pip install 'git+https://github.com/huggingface/transformers@cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d'
    .venv-hf/bin/python -m ipykernel install --prefix .venv-hf --name qwen-omni-full --display-name 'Qwen Omni full inference'

Choose a CUDA-compatible PyTorch/torchvision pair for your hardware. Video/audio decoding also requires ffmpeg. The pinned model card discusses memory requirements; its estimates depend on inputs and backend and are not guarantees.

Select that kernel, set RUN_FULL_INFERENCE to True in the setup cell, and supply local media paths in the optional cells. Those cells download model weights. Use a single-item batch for speech output. Prefer short audiovisual clips while studying this historical snapshot. The speech example writes optional_response.wav at 24 kHz.

## Reuse for another paper

The reusable [writing plan](../READING_GUIDE_PLAN.md) and [archived skill](../skills/paper-code-reading-guide/SKILL.md) live in the parent directory requested.

Tell Codex:

> Use ~/Dev/project/LLM_reading_guide/skills/paper-code-reading-guide/SKILL.md for &lt;paper URL or model ID&gt;.

The archived skill is portable and has not been globally installed. Its default preferences match this guide. A general transfer review is recorded in [VALIDATION.md](VALIDATION.md).

## Licenses

The archived Transformers files retain their Apache 2.0 license and copyright notices. The checkpoint uses its separately archived Qwen Research license. The main paper and reproduced figures are attributed to the Qwen Team and distributed under the paper's CC BY 4.0 notice. Other component papers retain their original notices. 
