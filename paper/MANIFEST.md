# Paper artifact manifest

- Status: advisor-authored storyline draft; no experiment result
- Repository baseline: `945972fa936b12bc91a8850edfbbd97f9cce3fbb`
- Artifact commit: `PENDING`
- Draft PR head: `PENDING`
- TeX entry: `paper/main.tex`
- Bibliography: `paper/references.bib` (intentionally empty; source disclosure in paper)
- Evidence ledger: `paper/EVIDENCE_LEDGER.md`
- PDF: `paper/bidkv.pdf`
- Full build transcript: `paper/tectonic.log`
- Build command: `/home/shuhao/.conda/envs/neuromem/bin/tectonic -X compile main.tex --outdir build --keep-logs`
- Build exit code: `0`
- PDF pages: `3`
- Visual inspection: all three pages rendered with `pdftoppm -png -r 110` and inspected; no clipping, overlap, unreadable table, blank page, or anomalous column gap
- Tectonic diagnostics: no undefined references, missing citations, or overfull boxes; full transcript retains non-fatal underfull diagnostics in narrow two-column prose
- Highest evidence: host contract
- Missing evidence: native pressure/release receipt, serving correctness, matched distributions, surrogate calibration, SLO boundary
- Host tests: `377 passed, 19 skipped in 1.18s` with `PYTHONPATH=src`; skips are optional framework/runtime cases
- Installed native-selector check: `1 skipped` because a compatible vLLM runtime is not present; not reported as serving validation

## Artifact SHA256

- `paper/main.tex`: `1595809a877de922f547f00514a71dda585a6546da8bbc8ff42493cc46fc949a`
- `paper/references.bib`: `cc743f32348837e720ac8792be8b5657bd9995bcb2086ecc5bd5af0a6efe9de4`
- `paper/EVIDENCE_LEDGER.md`: `9589054f13825bcc0ba4f4da9a796e4d2cc76ffe9c3cb86e81c3278750e529e3`
- `paper/bidkv.pdf`: `c21658249eedc2fb1b732d8a92883db1ba9ffc77f4c2c12ee3e29a5eb735dd9a`
- `paper/tectonic.log`: `3adc5877e802608be057adb143c4571eb4cc92ccfe06a5a6da752f7736135d28`

The final metadata-only commit binds this manifest to the preceding artifact
commit; the Draft PR body records both exact remote hashes.
