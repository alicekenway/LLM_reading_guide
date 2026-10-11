# Guide contract and authoring outline

## Chapter pattern

For each important mechanism, provide:

1. The concrete problem and a motivating example.
2. What earlier approaches do and the specific limitation addressed.
3. The mechanism with a diagram where useful.
4. Core equations, defined symbols, units, tensor dimensions, and training/inference distinction.
5. A worked numerical or shape example with explicit simplifications.
6. A reading assignment: exact paper section, PDF page, equation/figure and what to look for.
7. Code: pinned file/symbol/line range, explanation of operations without long copied excerpts, and caller/callee connections.
8. A comprehension check followed by an answer.

## Required two-part notebook outline

- Brief title and reading navigation.
- **Part 1 — Understand the idea:** four connected sections covering the problem, existing methods/limitations, proposed method, and results. Use one concrete running example, a simple architecture/responsibility diagram, and a small evidence table. Do not interrupt this part with setup or implementation cells.
- **Part 2 — Understand the components:** setup, component sections following data flow, training objectives, one complete walkthrough, then optional inference and reproduction notes. Adapt the component list to the actual model rather than copying Qwen-specific modules.
- Teach each component through purpose, inputs/outputs, intuition, defined formulas, a worked example, and attached exact PDF/original-code references. Explain training versus inference and missing release details.
- Use connected prose and substantial Markdown cells; split only around a useful demonstration or figure. Avoid heading-only/link-only cells and repeated “read the implementation” or “run the demonstration” boilerplate. Explain jargon before relying on it, without removing important technical detail.

## Accuracy and portability checklist

- Distinguish paper claims, component-source claims, verified code behavior, and pedagogical derivations.
- Preserve original equation numbering only for original equations; never invent numbers or release details.
- Specify PDF page numbers separately from printed page numbers if they differ.
- Source references must resolve to archived symbols, exact line ranges, and immutable URLs; link directly to original local source files and list symbols and line ranges for editor navigation.
- Explain unfamiliar components' architecture and learning objective. A named model is not an explanation.
- Input placeholders must match feature counts. Trace dimensions through each model boundary.
- Check wrapper order against claims of streaming, concurrency, causality, and lookahead.
- Scope benchmark results to the evaluated variant and settings; motivation alone is not causal evidence.
- Default cells must be deterministic, CPU runnable, network independent, and weight free.
- Optional inference must skip cleanly and state when it was not executed.
- Use the notebook as the sole guide, PDFs for papers, and original files for code; do not generate HTML copies or source browsers.
- Save rendered outputs and inspect their appearance; execution alone does not prove visual quality.
- Validate manifest hashes, notebook format, reference links, and numerical invariants.

## Transfer review

Before archiving a new skill revision, apply the contract to an unrelated model with a paper/code mismatch or missing training implementation. Confirm that the workflow identifies the gap, adapts its chapters, and avoids copying architecture or benchmark assumptions from the previous model. Use an independent evaluation when available; otherwise record a manual transfer review without claiming an agent test.
