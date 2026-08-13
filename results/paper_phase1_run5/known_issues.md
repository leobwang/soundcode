# Paper Phase 1 — known issues

Appended to by `scripts/paper_phase1.py` when an arm is skipped, partially run, or encounters an infrastructure failure.

## Arm 9 (rocode_upstream_cpp): SKIPPED

The upstream ROCODE codebase (`reproduce/algo/rocode/upstream/`) supports only Python HumanEval and MBPP. Its `PA_tools.py` runs Python `exec()` on a candidate snippet and parses `SyntaxError` / `AssertionError`; there is no C++ program analyzer in the upstream or any way to compile/run C++ test cases through its interface. Porting PA to C++ (g++ syntax errors + dynamic test execution) is a multi-day effort and is out of scope for the paper deadline.

The headline comparison (arms 3 vs 4, 5 vs 6) does not depend on this arm — those are 'sync_naive' (our own ROCODE-faithful sync scheduler) vs 'async_naive' (SoundCode). The sync_naive arm is the algorithmic equivalent of ROCODE: it blocks on the program analyzer at every statement boundary, applies a rollback on error, and restarts. The only difference is the language (C++/Rust vs Python) and the engine (vLLM vs HuggingFace transformers).

