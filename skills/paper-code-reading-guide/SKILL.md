---
name: paper-code-reading-guide
description: Use when a user wants to study a machine-learning model through its paper and source implementation, with an explanatory Jupyter reading guide, equations, exact code references, diagrams, and runnable demonstrations.
---

# Paper-to-code reading guide

Create a local study package that teaches a reader to understand the model and follow its actual implementation. Accept a paper or model identifier as the short invocation; discover its official counterpart.

Read [the guide contract](references/guide-contract.md) before authoring. Follow the user's requested destination and language. Defaults: English; basic ML, attention, LLM, and PyTorch prerequisites; detailed explanations of unfamiliar component structure and training; CPU demonstrations without pretrained weights; optional full inference disabled.

## Workflow

1. Inspect the workspace and identify official sources. Resolve code/checkpoint commits and versioned paper IDs. Archive the paper, relevant code, configs, templates, utilities, and licenses with a checksum manifest. Do not download weights by default.
2. Read the paper and trace preprocessing, forward, generation, and decoding. Build an exact paper/code map before writing explanations. Follow inherited utilities when necessary.
3. Use two main parts. Part 1 answers, in order: what problem is addressed; what existing methods do and where they struggle; what method is proposed; what results are reported. Make it a connected plain-language explanation with a running example, simple diagram, and small evidence table. Keep code/setup and detailed formulas out of Part 1. Part 2 follows components in data-flow order, then training and a complete walkthrough. Teach each component step by step: intuition before formulas, defined terms/symbols/units/dimensions, worked examples, and attached PDF/original-code reading assignments. Explain prior-method limitations fairly.
   Group related explanation into substantial Markdown cells; split around useful demonstrations or figures, never every heading or reference. Avoid isolated reading-link cells and repeated read/run boilerplate. Put setup at the beginning of Part 2 and optional inference/reproduction notes at the end. Preserve depth while using easy-to-understand language.
4. Explain unfamiliar named components through architecture, training objective, inputs/outputs, and adaptations in this model. Distinguish component-paper facts from the target model's disclosed recipe.
5. Add deterministic CPU demonstrations and rendered diagrams. Label pedagogical simplifications; compare against actual weight-free source functions when practical. Keep optional heavyweight inference behind a false-by-default flag.
6. Execute the notebook in order, retain outputs, inspect presentation, and validate citations, source lines, shapes, numerical invariants, and checksums. Report any unavailable execution or source honestly.

## Handoff

Deliver the notebook, PDF papers, unmodified original source files, manifest, paper/code map, figures, environment requirements, and exact reproduction commands. Use the notebook as the sole guide; do not create HTML exports, HTML paper copies, or source-browser pages. Link to local original code with symbols and exact line ranges instead of pasting long implementation excerpts or chat templates. Keep runnable educational demonstrations and figures in the notebook. Explain material gaps between the paper and released implementation, including staged generation wrappers, absent training code, and checkpoint-specific differences.

Use the available PDF and notebook tooling if helpful, but keep this skill usable without any particular plugin. Do not publish, globally install the skill, download weights, or modify unrelated repositories unless the user requests it.
