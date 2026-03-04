# ConTSG-Bench Specification (Single Source of Truth)

This document is the canonical reference for benchmark scope. Public-facing docs should
reference this file instead of duplicating counts.

Last updated: 2026-03-03

## Scope Summary

- Benchmark datasets: **10**
- Generation models in benchmark suite: **11**
- Leaderboard metrics: **15**
  - Fidelity: 7
  - Adherence: 4
  - Utility: 4

## Benchmark Dataset IDs (10)

1. `synth-m`
2. `synth-u`
3. `ettm1`
4. `weather_concept`
5. `weather_morphology`
6. `telecomts_segment`
7. `istanbul_traffic`
8. `airquality_beijing`
9. `ptbxl_concept`
10. `ptbxl_morphology`

## Generation Model IDs (11)

### Text-conditioned

- `verbalts`
- `t2s`
- `bridge`
- `diffusets`
- `text2motion`
- `retrieval`

### Attribute-conditioned

- `timeweaver`
- `wavestitch`
- `tedit`

### Label-conditioned

- `timevqvae`
- `ttscgan`

## Leaderboard Metric IDs (15)

### Fidelity (7)

- `acd`
- `sd`
- `kd`
- `mdd`
- `fid`
- `prdc_f1.precision`
- `prdc_f1.recall`

### Adherence (4)

- `jftsd`
- `joint_prdc_f1.precision`
- `joint_prdc_f1.recall`
- `cttp`

### Utility (4)

- `dtw`
- `crps`
- `ed`
- `wape`

## Ranking Policy Notes

- Overall ranking uses Fidelity + Adherence groups.
- Utility metrics are reported but excluded from overall ranking.
