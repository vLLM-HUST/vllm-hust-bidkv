# Paper artifact manifest

- Status: advisor-authored storyline draft; no experiment result
- Repository baseline: `945972fa936b12bc91a8850edfbbd97f9cce3fbb`
- Artifact commit: `a7edee16056a36795f6c400fa71fc808f4e2b703`
- Draft PR head: metadata-only successor; exact remote head is recorded in the Draft PR body
- TeX entry: `paper/main.tex`
- Bibliography: `paper/references.bib` (2 verified primary-source entries, both cited)
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

- `paper/main.tex`: `cc912f235b347fc457b2e16857170cb1acd4d6573ada0636f9ca2cbe6d92efb2`
- `paper/references.bib`: `0c7d63cfbb523a2a86a3768a2b9e1cb0322da7b33bb89b48bb87844b2f40f20a`
- `paper/EVIDENCE_LEDGER.md`: `9589054f13825bcc0ba4f4da9a796e4d2cc76ffe9c3cb86e81c3278750e529e3`
- `paper/bidkv.pdf`: `9f2601b3154cf55615a4a5a6956f0a7c4597504bbd8a0da7a76645dc90c22069`
- `paper/tectonic.log`: `3b42947ef731b94b6cb2db3da31a6b47f8f385103966217c3840092ebf11d7fc`

The final metadata-only commit binds this manifest to the preceding artifact
commit; the Draft PR body records both exact remote hashes.
