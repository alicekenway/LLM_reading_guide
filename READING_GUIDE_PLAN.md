# Reusable plan: read a model paper through its implementation

## Purpose and reader

Produce an English Jupyter study guide that enables a reader with basic ML, LLM, attention, and PyTorch knowledge to understand the paper and independently follow the actual model code. Explain unfamiliar component architectures and training methods rather than treating names such as Whisper, codecs, diffusion, or flow matching as explanations.

## Inputs and defaults

Accept a paper URL/arXiv ID and a model or repository identifier. Discover the official counterpart when only one is supplied. Ask only about choices that materially change the guide and cannot be discovered. Default to an English, implementation-focused guide with CPU demonstrations and disabled optional full inference. Do not download model weights for the default guide.

## Work sequence

1. Inspect the target workspace and any user instructions. Identify the authoritative paper, checkpoint, source implementation, and relevant original component papers. Resolve mutable references to immutable commits and versioned paper IDs. Check local hardware and dependencies.
2. Archive the paper PDF and available source for figure provenance; source files, configs, chat templates, relevant inherited utilities, and licenses. Record paths, immutable URLs, revisions, retrieval timestamps, and SHA-256 hashes. Mark failures accurately. Preserve upstream files rather than editing them.
3. Read the full paper and trace the real input-processing, forward, generation, and output-decoding code. Build a paper-to-code map with exact PDF pages, sections, equations/figures, symbols, local line ranges, and pinned upstream links. Trace inherited functions when they matter.
4. Write exactly two main parts. Part 1 is a connected, plain-language explanation with four sections: the problem, existing methods and their limitations, the proposed method, and the results. Include a concrete running example, a simple responsibility diagram, and a small evidence table. Keep setup, code assignments, detailed dimensions, and derivations out of Part 1. Part 2 follows the actual components in data-flow order, then training and a complete walkthrough. Each component teaches purpose, inputs/outputs, intuition, formulas with defined symbols/units/dimensions, one worked example, and precise PDF/original-code reading locations. Put setup at the start of Part 2 and optional inference/reproduction notes at its end.
5. Explain all core equations: symbols, dimensions, units, conditioning, training/inference roles, and code operations. Distinguish numbered paper equations, equations from component sources, and explanatory derivations. Explain architecture AND training for unfamiliar inherited components. Do not invent unavailable training recipes.
6. Add architecture/call-flow diagrams and bounded, deterministic CPU examples. Demonstrations must illustrate real mechanisms, identify simplifications, and test meaningful invariants. Where feasible, compare with weight-free executions of actual upstream functions. Optional full inference must be gated and include dependencies, modality inputs, resource considerations, output interpretation, and honest execution status.
7. Explain the training stages and evaluation evidence. Identify which results apply to the model family versus the exact checkpoint. Separate reported design, released implementation, derived intuition, and unavailable details. Compare wrapper execution order with streaming or concurrency claims.
8. Execute the default notebook in a clean kernel without network or model weights. Save notebook outputs, inspect figures and equations at reading size, validate all references and checksums, and record remaining limitations.

## Deliverables

Within `<reading-guide-root>/<model-name>/`: `reading_guide.ipynb`, `sources/manifest.json`, versioned sources, `paper_code_map.csv`, figures, pinned notebook requirements, build/validation commands, and a concise README. Keep scripts only where they improve reproducibility. Adapt chapter organization to each paper; do not force Qwen-specific components onto unrelated models.

## Acceptance criteria

- The two-part structure separates idea understanding from detailed component teaching; Part 1 answers its four questions in order and needs no code execution.
- Explanations use connected, easy-to-understand prose. Introduce intuition before equations and define jargon on first use. Keep the technical depth; remove fragmentation rather than necessary explanation.
- Group explanations into substantial Markdown cells. Split around useful demonstrations/figures, not every small heading or link. Do not create isolated heading or reading-link cells, or repeated boilerplate prompts to read/run code.

- A reader can trace one representative input through preprocessing, every major model boundary, generation, and decoding with shapes and code references.
- Every central contribution has a problem statement, explained mechanism/formula, paper passage, and verified code location or an explicit release gap.
- All default cells run in order; saved figures and outputs are visible; optional inference skips safely.
- References, line ranges, source revisions, computations, and diagrams agree with the actual archived files.
- Prior-method limitations and benchmark conclusions remain proportional to the evidence.
- Source licensing and missing validation are visible. Full training reproduction is outside the default scope.

## Reuse

Invoke the archived skill directly:

> Use `~/Dev/project/LLM_reading_guide/skills/paper-code-reading-guide/SKILL.md` for `<paper URL or model ID>`.

The skill is archived here; global installation is a separate optional action.

## Reading format preference

Use the Jupyter notebook as the sole reading guide. Do not create HTML exports, HTML paper copies, or source-browser pages. Reference papers as PDFs and use pictures inside the notebook. Link to the unmodified original local source files and list exact symbols and line ranges for editor navigation. Do not paste long implementation excerpts or chat templates into the notebook. Keep short runnable pedagogical demonstrations, explanatory formulas, and saved outputs.
