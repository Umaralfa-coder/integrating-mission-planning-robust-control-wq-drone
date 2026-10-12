"""Smoke test on a toy yaw/surge plant (NOT the GOOSE model): checks that the
controllers run, stay bounded and converge. Run: python test_nrpid.py"""
import math
from nr_pid import PID, NRPID, NLPID

def run(ctrl, T=60.0, dt=0.05):
    psi = r = u = 0.0
    ctrl.desired_heading, ctrl.desired_speed = 1.0, 1.5
    hist = []
    for n in range(int(T / dt)):
        ctrl.current_heading, ctrl.current_speed = psi, u
        F1, F2 = ctrl.update(dt)
        tau_u, tau_r = F1 + F2, (F1 - F2) * ctrl.b
        r += (-tau_r / 8.0 - 2.0 * r) * dt      # sign chosen so +tau_r reduces psi
        psi += r * dt
        u += (tau_u / 30.0 - 0.6 * u) * dt
        hist.append((abs(psi - 1.0), abs(u - 1.5)))
    return hist[-1], max(h[0] for h in hist)

if __name__ == "__main__":
    for C in (NRPID, NLPID):
        print(C.__name__, run(C(30.0, 0.0, 0.2) if False else C(30.0, -18.0, 0.2)))
    pid = PID(30.0, -18.0, 0.2); pid.set_coef([8.0, 0.2, 6.0, 40.0, 2.0, 2.0])
    print("PID", run(pid))
