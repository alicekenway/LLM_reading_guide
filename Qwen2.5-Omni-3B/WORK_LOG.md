# Implementation record

- User approved the conversational plan and requested implementation.
- Archived the official paper PDF, HTML, source, figures, checkpoint metadata, component papers, and HF implementation at immutable revisions.
- Preserved upstream source bytes, checked against the connector-provided Git blob hashes.
- Authored eleven chapters and deterministic weight-free demonstrations.
- Reproduced the historical chunk-helper discrepancy using original function bodies, including both long-input position disagreement and a tiny-grid IndexError. Kept the archive unchanged.
- Used the requested workspace for the study artifact; no application branch, repository mutation, merge, or publication was needed.
- Prepared the reusable shared plan and skill in temporary staging, then saved them in the explicitly requested parent directory.
- Validation is performed with notebook execution, numeric invariants, exact source references/checksums, upstream-function checks, and visual inspection. Full pretrained inference remains optional and unrun.
- Review uses an author self-review because no independent subagent tool is available; no independent agent test is claimed.
- Final checks passed: 128 notebook cells, 36 executed code cells, 13 rendered figures, 63 source checksums, 37 paper/code references, HTML structure/local links, original-function CPU checks, and skill structure.
- Chromium stalled before loading the HTML; browser equation/layout verification remains incomplete and is explicitly recorded in VALIDATION.md and html_validation.json. Saved figures were inspected directly.

## Notebook-only revision

- User requested a notebook-only guide, PDF paper references, and original code files rather than browser-based code reading.
- Removed HTML guide/export tooling, paper HTML, source browsers, and MathJax assets; cleaned the source manifest.
- Replaced implementation excerpt cells and the chat-template dump with original-file reading assignments, symbols, and line ranges. Preserved all original HF source bytes, demonstrations, and paper figures.
- Updated the archived parent plan and skill so future guides follow the same reading preference.
- Final notebook-only validation passed: 128 cells, 17 executed code cells, 13 figures, 60 archived-source checksums, 37 paper/code references, and all local notebook reading links. Original HF source Git blob matches remain intact; archived skill structural validation passed.

## Two-part teaching revision

- User approved the two-part plan and selected step-by-step teaching depth.
- Rewrote Part 1 as a connected account of the problem, existing approaches, method, and results, using a recurring cooking-video example and a new plain-language diagram.
- Expanded the component explanations with intuition, definitions, arithmetic/shape examples, and operation-level code reading. Grouped related prose rather than splitting every heading or reference.
- Moved setup into Part 2, attached diagrams/demonstrations to their explanations, and separated intended TMRoPE behavior from the historical implementation note.
- Checked the original PDF and corrected exact benchmark-table pages. Preserved the paper/3B distinction and all original HF source bytes.
- Updated the reusable parent plan, skill, and contract with the two-part structure and teaching preferences.
- Final validation passed: two main parts, 31 cells, 14 executed code cells, 13 figures (9 outputs and 4 inline images), 60 archived sources with original Git blob checks, and 37 exact paper/code entries. Skill structure validation passed. Final prose-only edits retained outputs only after confirming code-cell sources were unchanged. Full GPU inference remains unrun.
