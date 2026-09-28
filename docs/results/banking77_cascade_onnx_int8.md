# edgefront - banking77

300 examples, 77 labels. artifacts/predictions/distilbert-mnli-onnx-int8-cpu.predictions.json + artifacts/predictions/jev.predictions.json

## Verdict

**cascade@0.85 leads at 79.0% accuracy**

- no hosted/local pair was run, so no substitution verdict
- recommended: **cascade@0.85** (substitution tolerance 3.0 pts)

## Results

| backend | accuracy | macro-F1 | ECE | p50 ms | p90 ms | p99 ms | $/1M calls | offline | errors |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| cascade@0.00 | 0.167 | 0.168 | 0.126 | 810 | 1177 | 1849 | 21.8 | no | 0 |
| cascade@0.05 | 0.177 | 0.177 | 0.129 | 813 | 1179 | 1864 | 22.5 | no | 0 |
| cascade@0.10 | 0.250 | 0.271 | 0.141 | 846 | 1302 | 2164 | 29.2 | no | 0 |
| cascade@0.15 | 0.343 | 0.365 | 0.137 | 877 | 1442 | 2482 | 35.5 | no | 0 |
| cascade@0.20 | 0.410 | 0.436 | 0.144 | 924 | 1565 | 2482 | 41.6 | no | 0 |
| cascade@0.25 | 0.493 | 0.513 | 0.162 | 1060 | 1637 | 2482 | 51.0 | no | 0 |
| cascade@0.30 | 0.587 | 0.590 | 0.146 | 1126 | 1651 | 2482 | 57.7 | no | 0 |
| cascade@0.35 | 0.650 | 0.635 | 0.144 | 1150 | 1693 | 2841 | 63.8 | no | 0 |
| cascade@0.40 | 0.687 | 0.657 | 0.144 | 1174 | 1785 | 2841 | 67.8 | no | 0 |
| cascade@0.45 | 0.713 | 0.689 | 0.139 | 1201 | 1821 | 2841 | 70.3 | no | 0 |
| cascade@0.50 | 0.737 | 0.711 | 0.131 | 1210 | 1844 | 3106 | 72.2 | no | 0 |
| cascade@0.55 | 0.747 | 0.722 | 0.128 | 1215 | 1850 | 3208 | 73.8 | no | 0 |
| cascade@0.60 | 0.753 | 0.730 | 0.125 | 1218 | 1850 | 3208 | 74.5 | no | 0 |
| cascade@0.65 | 0.773 | 0.741 | 0.110 | 1230 | 1850 | 3208 | 76.1 | no | 0 |
| cascade@0.70 | 0.777 | 0.743 | 0.110 | 1231 | 1850 | 3208 | 76.8 | no | 0 |
| cascade@0.75 | 0.777 | 0.736 | 0.112 | 1232 | 1866 | 3208 | 77.4 | no | 0 |
| cascade@0.80 | 0.787 | 0.741 | 0.106 | 1240 | 1866 | 3208 | 78.6 | no | 0 |
| cascade@0.85 | 0.790 | 0.742 | 0.103 | 1241 | 1866 | 3208 | 78.7 | no | 0 |
| cascade@0.90 | 0.790 | 0.741 | 0.103 | 1241 | 1866 | 3208 | 78.9 | no | 0 |
| cascade@0.95 | 0.790 | 0.741 | 0.103 | 1241 | 1866 | 3208 | 79.1 | no | 0 |
| cascade@1.00 | 0.790 | 0.741 | 0.103 | 1241 | 1866 | 3208 | 79.1 | no | 0 |
| cascade@1.01 | 0.790 | 0.741 | 0.103 | 1241 | 1866 | 3208 | 79.1 | no | 0 |

On the accuracy/latency frontier: cascade@0.00, cascade@0.05, cascade@0.10, cascade@0.15, cascade@0.20, cascade@0.25, cascade@0.30, cascade@0.35, cascade@0.40, cascade@0.45, cascade@0.50, cascade@0.55, cascade@0.60, cascade@0.65, cascade@0.70, cascade@0.80, cascade@0.85, cascade@0.90, cascade@0.95, cascade@1.00, cascade@1.01.

## How these numbers were produced

- first 3 calls per backend discarded as warmup; latency is the distribution over the rest, never a bare mean
- backends run sequentially so they cannot contend for CPU
- a failed call counts as wrong, not as missing
- `cascade@0.00` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 0.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.05` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 1.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.10` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 13.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.15` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 24.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.20` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 34.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.25` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 51.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.30` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 62.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.35` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 73.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.40` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 80.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.45` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 84.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.50` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 88.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.55` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 90.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.60` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 92.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.65` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 94.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.70` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 96.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.75` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 97.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.80` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 99.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.85` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 99.3% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.90` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 99.7% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@0.95` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 100.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@1.00` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 100.0% of calls (1366 input tok/call x $0.042/Mtok, output free)
- `cascade@1.01` cost basis: local always runs (amortised $600 over 3y at 25% utilisation, plus 45W at $0.12/kWh) plus hosted on 100.0% of calls (1366 input tok/call x $0.042/Mtok, output free)

## Environment

- Windows-11-10.0.22631-SP0 / Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- python 3.13.0, run 2026-09-28T16:47:15Z UTC
- edgefront commit `8c61efe`
- hosted latency depends on network position; rerun locally before trusting it for your own deployment
