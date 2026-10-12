# Integrating Mission Planning and Robust Control for Water Quality Monitoring with an Aquatic Surface Drone

Code, data products and supplementary material accompanying the manuscript
(CERI Digital Systems, IMT Nord Europe; Univ. Lille).

## Repository layout

| Folder | Content |
|---|---|
| `path_planning/` | Coverage path planning with the DE/AI optimiser (heading and spacing) on top of covplan; AOI8 rings and runs |
| `path_planning/ablation/` | Ablation scripts and result files (length-only fitness; heading-only vs heading + spacing) |
| `control/` | NR-PID controller and simulation (**to add**) |
| `water_quality/` | Raw-log parser and Aug 6, 2026 time-series plot; GP water-quality estimation (**GP code to add**) |
| `figures/` | Final paper figures (PDF) |
| `supplementary/` | Supplementary document (PDF) |
| `video/` | Link to the mission video (see `video/README.md`) |

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Path planning**: `python path_planning/run_aoi8_de.py` (seed 42; weights 0.35/0.25/0.25/0.15 for length, turns, overlap, coverage; spacing bound [0.5, 4] m).
2. **Ablations**: `bash path_planning/ablation/hd.sh` (heading-only at fixed spacing) and `abl.sh` (length-only fitness). Results are stored as JSON next to the scripts.
3. **Water quality time series**: set the raw log path in `water_quality/parse6.py`, then `python parse6.py && python plot6.py`.

Note: some scripts still contain absolute paths from the authors' machine; edit them before running.

## Data

Field data were collected on the Scarpe River (France). Raw sensor logs follow the format
`DDMMYY;hhmmss.ss;lat;lon;alt;$PVHY2,...;1,Temp,DO%,DOmgL,...;2,T,pH,ORP,...;3,T,Cond,...`.

## Citation

If you use this work, please cite the manuscript (reference to be added after acceptance).

## License

MIT, see `LICENSE`.
