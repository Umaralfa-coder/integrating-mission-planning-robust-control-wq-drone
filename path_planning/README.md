# Path planning

DE optimiser (`FastGeneticAIOptimizer`) for the track heading and spacing of the area-based coverage planner.

**Dependency:** the `covplan` package (Ramkumar Sudha et al., 2024) providing `covplan.lines` and
`covplan.coverage_path_planner.Field`. Install it from its original repository and make sure
`python -c "import covplan"` works before running anything here.

| Path | Content |
|---|---|
| `run_aoi8_de.py`, `run_aoi8_multi.py` | DE runs on AOI8 |
| `fixed_coverage.py`, `optimize_script_v2.py` | coverage evaluation and optimiser |
| `ablation/` | length-only fitness (`abl.sh`) and heading-only at fixed spacing (`hd.sh`), with the result JSON files |
| `analysis/` | Pareto weight sweep and term contributions |

Seed 42; weights 0.35 / 0.25 / 0.25 / 0.15; spacing bounds [0.5, 4] m.
