# Refusal contrast set

Built by `scripts/build_refusal_data.py` (seed 0). Fit and test splits are disjoint.

| file | class | source | n |
|---|---|---|--:|
| harmful.jsonl | harmful (fit) | AdvBench (Zou et al. 2023) | 128 |
| harmless.jsonl | harmless (fit) | Alpaca (Taori et al. 2023) | 128 |
| harmful_test.jsonl | harmful (held-out) | AdvBench | 100 |
| harmless_test.jsonl | harmless (held-out) | Alpaca | 80 |

The direction is diff-of-means on the fit split; the behavioural refusal rate is measured on the held-out split. AdvBench is MIT-licensed; Alpaca is CC BY-NC 4.0 (research use).
