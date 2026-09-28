# edgefront - banking77

500 examples, 77 labels. legacy-datasets/banking77 test split, 500 examples, seed 0

## Verdict

**jev leads at 79.0% accuracy**

- no hosted/local pair was run, so no substitution verdict
- recommended: **jev** (substitution tolerance 3.0 pts)

## Results

| backend | accuracy | macro-F1 | ECE | p50 ms | p90 ms | p99 ms | $/1M calls | offline | errors |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| jev | 0.790 | 0.741 | 0.103 | 381 | 703 | 2417 | 57.4 | no | 0 |

On the accuracy/latency frontier: jev.

## How these numbers were produced

- first 3 calls per backend discarded as warmup; latency is the distribution over the rest, never a bare mean
- backends run sequentially so they cannot contend for CPU
- a failed call counts as wrong, not as missing
- `jev` cost basis: 1366 input tok/call x $0.042/Mtok, output free

## Environment

- Windows-11-10.0.22631-SP0 / Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- python 3.13.0, run 2026-09-28T16:35:16Z UTC
- edgefront commit `8c61efe`
- hosted latency depends on network position; rerun locally before trusting it for your own deployment
