# Examples

Small, self-contained scripts that show how to drive diapason. None of them
need a real model or real audio files: the ASR example uses the deterministic
dummy adapter, and the SED/AQA suites score predictions that are already stored
in their manifests.

| Script | What it shows |
| --- | --- |
| `run_asr.py` | Loading a suite, running one ASR task, printing scores |
| `run_suite.py` | Running every bundled smoke suite and writing reports |
| `compare_adapters.py` | Scoring the same task with two adapter configurations |

The `data/` directory holds the JSONL manifests the bundled configs in
`../configs` point at. Each line is one evaluation example; see
`docs/usage.md` for the full manifest schema.

Run an example from the repository root:

```bash
python examples/run_asr.py
```
