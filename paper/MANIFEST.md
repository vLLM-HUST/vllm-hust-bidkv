# Paper artifact manifest

- Status: advisor-authored evidence sync; no new experiment or result
- Repository baseline: `a0cba97d9abdc99908e46616db622f0e0099127f`
- Artifact commit: `c4d943a5f7a553719464c10deb6402508078be27`
- Draft PR head: metadata-only successor; exact remote head is recorded in the Draft PR body
- TeX entry: `paper/main.tex`
- Bibliography: `paper/references.bib` (2 verified primary-source entries, both cited)
- Evidence ledger: `paper/EVIDENCE_LEDGER.md`
- PDF: `paper/bidkv.pdf`
- Full build transcript: `paper/tectonic.log`
- Build command: `/home/shuhao/.codex-audit/research-demand-tools-20260909/tectonic -X compile main.tex --outdir build --keep-logs`
- Build exit code: `0`
- PDF pages: `3`
- Visual inspection: all three pages rendered with `pdftoppm -png -r 110` and inspected; no clipping, overlap, unreadable table, blank page, or anomalous column gap
- Tectonic diagnostics: no undefined references, missing citations, or overfull horizontal boxes; full transcript retains non-fatal underfull diagnostics and one 1.72pt overfull vertical box in narrow two-column prose
- Highest evidence: controlled real-device qualification on Qwen3.8-27B, Ascend TP4, `FULL_DECODE_ONLY`
- Supported result boundary: five functional cells; the repeated interactive cell is negative for the current formula; the ascending-mixed cell is inconclusive because one repeat had zero policy calls
- Missing evidence: checked-in raw-result custody, cross-model/topology/device transfer, consistent treatment exposure in the ascending-mixed cell, surrogate calibration, and a beneficial regime over the strongest deployable baseline
- Source-suite receipt reported by the versioned evidence note: `385 passed, 25 skipped`; this paper did not rerun the student environment

## Artifact SHA256

- `paper/main.tex`: `b6374f49c048c1c7b98a58c0585da0003e5586db1fa631a8a41ddf6fc2820935`
- `paper/references.bib`: `0c7d63cfbb523a2a86a3768a2b9e1cb0322da7b33bb89b48bb87844b2f40f20a`
- `paper/EVIDENCE_LEDGER.md`: `2f1ecd4c3732600adf525871a04a60bce25d183f5f5797bc7a8bc01a528afc0a`
- `paper/bidkv.pdf`: `f66ee0aadcc04c2066a7c371517613cacd5c335f024d12d671cdbcf01b35d4aa`
- `paper/tectonic.log`: `e0e7409ac9f6441eb88ad27b767d6ceb8d0ea583d9a280955245b5ab531567e3`

The final metadata-only commit binds this manifest to the preceding artifact
commit; the Draft PR body records both exact remote hashes.
