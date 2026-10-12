# Control: NR-PID, NLPID ablation and PID baseline

| File | Content |
|---|---|
| `nr_pid.py` | `NRPID` (proposed), `NLPID` (no sliding-mode term), `PID` (fixed-gain baseline), thrust allocation |
| `params_field_trials.py` | Parameters of paper Table "NR-PID control parameters used in field trials" + implementation safeguards |
| `test_nrpid.py` | Smoke test on a toy plant (not the GOOSE model): `python test_nrpid.py` |

Usage (ROS 2 node side): set `desired_heading`, `desired_speed` (from the LOS guidance),
`current_heading`, `current_speed`, `current_sway`, then call `update()` to get the thruster forces `(F1, F2)`.

Notes
- Two scalar channels (heading -> tau_r, speed -> tau_u); position tracking is done by the LOS guidance layer.
- Heading error is `psi - psi_d` (opposite sign to the paper); see `heading_sign`.
- Safeguards not in the analysed law: filtered/clamped derivative, integral clamp, lower gain bounds, torque and thruster limits.
