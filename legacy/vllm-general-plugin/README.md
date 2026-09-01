# BidKV legacy vLLM adapter

This archived experiment-only distribution registers BidKV through
`vllm.general_plugins`. It is intentionally separate from the main `bidkv`
wheel and must not be installed in a typed Extension Manager serving
environment.

Install it only to reproduce the historical `BIDKV_STRATEGY` experiments from
the matching BidKV source revision.
