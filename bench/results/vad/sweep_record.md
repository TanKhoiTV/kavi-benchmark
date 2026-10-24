# VAD sweep record (Stage 5)

- command: `python -m bench.vad_stage5 --allow-provisional --publish-provisional`
- git commit: `9b8455d41f3239ea629a2a9c36ab965d375c970d`
- manifest: `eval_data/vad_manifest_v2.json` sha256 `706099670f5558a1ef2ecf435b2da1984d7fefa57b740e92e6537608ed737669`
- manifest seed: 42 (regenerated from the seed: yes)
- source bounds: `bench/vad_source_bounds.json` sha256 `9498658667b9ec412ee3feec8b92b16bc9b240da3f25b4c1eacb47cccefa27ab`
- collar: 0.15 s from `bench/vad_collar.json` sha256 `n/a` PROVISIONAL
- grid: thresholds [0.01, 0.02, 0.05, 0.1, 0.2], timeouts [200, 350, 500, 750, 1000] ms
- minimum items per cell: 20
- python 3.12.13, numpy 1.26.4
- runtime: 36.6 s

## Verification

- rows: 1000 (expected 1000 = 5 thresholds x 5 timeouts x 40 condition/SNR/pause cells)
- missing cells: 0
- cells below the minimum item count: 0
- failed runs: 0
- result: PASS
