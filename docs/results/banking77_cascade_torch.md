# edgefront - banking77

300 examples, 77 labels. artifacts/predictions/distilbert-mnli-torch-cuda.predictions.json + artifacts/predictions/jev.predictions.json

## Verdict

**cascade@0.85 leads at 79.0% accuracy**

- no hosted/local pair was run, so no substitution verdict
- recommended: **cascade@0.85** (substitution tolerance 3.0 pts)

## Results

| backend | accuracy | macro-F1 | ECE | p50 ms | p90 ms | p99 ms | $/1M calls | offline | errors |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| cascade@0.00 | 0.193 | 0.187 | 0.128 | 67.3 | 91.9 | 142 | 1.81 | no | 0 |
| cascade@0.05 | 0.193 | 0.187 | 0.128 | 67.3 | 91.9 | 142 | 1.81 | no | 0 |
| cascade@0.10 | 0.247 | 0.240 | 0.129 | 68.4 | 129 | 564 | 5.63 | no | 0 |
| cascade@0.15 | 0.303 | 0.307 | 0.135 | 69.0 | 437 | 955 | 10.4 | no | 0 |
| cascade@0.20 | 0.373 | 0.396 | 0.144 | 71.5 | 461 | 1094 | 16.5 | no | 0 |
| cascade@0.25 | 0.450 | 0.463 | 0.165 | 84.4 | 502 | 1522 | 24.8 | no | 0 |
| cascade@0.30 | 0.543 | 0.538 | 0.170 | 408 | 571 | 2069 | 33.9 | no | 0 |
| cascade@0.35 | 0.603 | 0.589 | 0.164 | 421 | 591 | 2069 | 39.7 | no | 0 |
| cascade@0.40 | 0.653 | 0.632 | 0.155 | 429 | 680 | 2182 | 43.9 | no | 0 |
| cascade@0.45 | 0.667 | 0.643 | 0.162 | 433 | 732 | 2182 | 46.6 | no | 0 |
| cascade@0.50 | 0.703 | 0.679 | 0.156 | 440 | 745 | 2480 | 50.6 | no | 0 |
| cascade@0.55 | 0.720 | 0.691 | 0.149 | 444 | 761 | 2480 | 52.1 | no | 0 |
| cascade@0.60 | 0.743 | 0.705 | 0.138 | 448 | 765 | 2480 | 54.4 | no | 0 |
| cascade@0.65 | 0.757 | 0.717 | 0.128 | 450 | 770 | 2480 | 55.4 | no | 0 |
| cascade@0.70 | 0.767 | 0.723 | 0.121 | 453 | 770 | 2480 | 56.1 | no | 0 |
| cascade@0.75 | 0.773 | 0.731 | 0.116 | 454 | 770 | 2480 | 57.1 | no | 0 |
| cascade@0.80 | 0.787 | 0.741 | 0.104 | 455 | 776 | 2480 | 57.8 | no | 0 |
| cascade@0.85 | 0.790 | 0.741 | 0.102 | 456 | 787 | 2480 | 58.8 | no | 0 |
| cascade@0.90 | 0.790 | 0.741 | 0.103 | 457 | 787 | 2480 | 59.0 | no | 0 |
| cascade@0.95 | 0.790 | 0.741 | 0.103 | 458 | 787 | 2480 | 59.2 | no | 0 |
| cascade@1.00 | 0.790 | 0.741 | 0.103 | 458 | 787 | 2480 | 59.2 | no | 0 |
| cascade@1.01 | 0.790 | 0.741 | 0.103 | 458 | 787 | 2480 | 59.2 | no | 0 |

On the accuracy/latency frontier: cascade@0.00, cascade@0.05, cascade@0.10, cascade@0.15, cascade@0.20, cascade@0.25, cascade@0.30, cascade@0.35, cascade@0.40, cascade@0.45, cascade@0.50, cascade@0.55, cascade@0.60, cascade@0.65, cascade@0.70, cascade@0.75, cascade@0.80, cascade@0.85.

## How these numbers were produced

- first 3 calls per backend discarded as warmup; latency is the distribution over the rest, never a bare mean
- backends run sequentially so they cannot contend for CPU
- a failed call counts as wrong, not as missing
- `cascade@0.00` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 0.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.05` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 0.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.10` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 6.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.15` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 15.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.20` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 25.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.25` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 40.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.30` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 56.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.35` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 66.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.40` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 73.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.45` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 78.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.50` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 85.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.55` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 87.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.60` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 91.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.65` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 93.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.70` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 94.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.75` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 96.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.80` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 97.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.85` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 99.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.90` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 99.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.95` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 100.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@1.00` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 100.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@1.01` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 100.0% of calls (1366 input tok/call x $0.042/Mtok, output free)

## Environment

- Windows-11-10.0.22631-SP0 / Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- python 3.13.0, run 2026-09-28T16:47:18Z UTC
- edgefront commit `8c61efe`
- hosted latency depends on network position; rerun locally before trusting it for your own deployment
