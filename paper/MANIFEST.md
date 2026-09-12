# Post-acceptance BidKV maintenance-note manifest

- Artifact identity: post-acceptance Ascend TP4 maintenance qualification;
  this is not the accepted SC 2026 paper, a replacement evaluation, or a
  retraction of the original result
- Repository baseline before PR #13: `a0cba97d9abdc99908e46616db622f0e0099127f`
- Previous PR #13 metadata head: `f6e69ffe7bd71b3502312b02fc138f00fe62e68e`
- Current artifact commit: `d3de629c091a2592962f2fcd299a2174dcb09eeb`
- Draft PR head: metadata-only successor; exact remote head is recorded in the Draft PR body
- Repository navigation: `README.md`
- TeX entry: `paper/main.tex`
- Bibliography: `paper/references.bib` (2 verified primary-source entries, both cited)
- Evidence ledger: `paper/EVIDENCE_LEDGER.md`
- PDF: `paper/bidkv.pdf`
- Full build transcript: `paper/tectonic.log`
- Build command: `/home/shuhao/.codex-audit/research-demand-tools-20260909/tectonic -X compile main.tex --outdir build --keep-logs`
- Build exit code: `0`
- PDF pages: `3`
- Visual inspection: all three pages rendered with `pdftoppm -png -r 110` and inspected; no clipping, overlap, unreadable table, blank page, or anomalous column gap
- Tectonic diagnostics: no undefined references, missing citations, or overfull horizontal boxes; full transcript retains non-fatal underfull diagnostics
- Highest maintenance evidence: controlled real-device qualification on Qwen3.8-27B, Ascend TP4, `FULL_DECODE_ONLY`
- Supported maintenance boundary: five functional cells; the repeated interactive cell is negative for the maintained formula/configuration; the ascending-mixed cell is inconclusive because one repeat had zero policy calls
- Accepted-result boundary: these later cells do not replace, negate, or reopen the accepted SC 2026 artifact and its original evidence
- Optional follow-up only: cross-model/topology/device transfer, consistent treatment exposure, surrogate calibration, and a new bounded winner regime
- Source-suite receipt: `385 passed, 25 skipped`; the 112 review independently reran this suite, while this proportional documentation/TeX edit did not rerun the student environment

## Artifact SHA256

- `README.md`: `a4ffa740be0d8fab7425feec358a2dc07c1339889eb2121a66279e611c273c73`
- `paper/main.tex`: `2d078e8cc6075c79ed0a9252c467a8e55823aa26854d2922cf24adff5a655264`
- `paper/references.bib`: `0c7d63cfbb523a2a86a3768a2b9e1cb0322da7b33bb89b48bb87844b2f40f20a`
- `paper/EVIDENCE_LEDGER.md`: `5149bbba5ab22bc3c924078ac5a46f8c42680f030d65ae995413d50bb42af354`
- `paper/bidkv.pdf`: `ecffcb71e6aef42d433ee4072848e3452229d27a1cda49cc7980ea366f8c1473`
- `paper/tectonic.log`: `41cd13c2f62d1ef91b7c4e94c53ac4812a196ceb5bd03668bb90e08e03e16a7f`

The final metadata-only commit binds this manifest to the preceding artifact
commit. The Draft PR records both exact remote hashes.
