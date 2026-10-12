"""NR-PID (nonlinear robust PID) and baseline controllers for the GOOSE SAD.

Implements, per scalar channel (heading psi -> tau_r, speed u -> tau_u):

    tau = K_P(.) e + K_D(.) e_dot + K_I(.) int(e) + Theta(t) tanh(sigma / Phi)
    sigma = e_dot + lambda e

with the nonlinear gain shape (paper Eqs. Kp/Kd/Ki_nonlinear), the Lyapunov
adaptation laws of the base gains (Eqs. adapt_Kp..Ki) with projection, and the
three-case projection law for Theta (Eq. adapt_Theta).

Classes
-------
PID    fixed-gain baseline (heading PID, speed PI-D + feedforward)
NRPID  proposed controller (full adaptation + sliding-mode term)
NLPID  ablation: same as NRPID with the sliding-mode term removed (Theta = 0)

Conventions
-----------
* Heading error is e = sawtooth(psi - psi_d) (sign opposite to the paper's
  e = psi_d - psi); the thrust allocation F2 = tau_u/2 - tau_r/(2b) accounts
  for it.  `heading_sign` flips it if the vehicle frame is reversed.
* Speed error uses the total speed sqrt(u^2 + v^2).
* Position tracking (cross-track) is handled by the LOS guidance layer, which
  supplies psi_d and u_d.
"""
import math
import time
from math import atan, exp, tan

from params_field_trials import HEADING, SPEED, IMPL


def sawtooth(x: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return 2 * atan(tan(x / 2))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Baseline fixed-gain PID
# ---------------------------------------------------------------------------
class PID:
    def __init__(self, m11: float, Xu: float, b: float):
        self.m11, self.Xu, self.b = m11, Xu, b
        self.desired_heading = self.current_heading = 0.0
        self.heading_error = self.last_heading_error = self.heading_error_integ = 0.0
        self.desired_speed = self.current_speed = self.current_sway = 0.0
        self.speed_error = self.last_speed_error = self.speed_error_integ = 0.0
        self.kp = self.ki = self.kd = 0.0
        self.kp_speed, self.ki_speed, self.kd_speed = 10.0, 5.0, 1.0
        self.F1 = self.F2 = 0.0
        self.last_t = None
        self.tau_u = self.tau_r = 0.0

    def set_coef(self, coefs):
        """coefs = [kp, ki, kd] or [kp, ki, kd, kp_speed, ki_speed, kd_speed]."""
        if len(coefs) >= 3:
            self.kp, self.ki, self.kd = coefs[0:3]
        if len(coefs) >= 6:
            self.kp_speed, self.ki_speed, self.kd_speed = coefs[3:6]

    def update(self, dt=None):
        if dt is None:
            t = time.time()
            dt = 0.0 if self.last_t is None else t - self.last_t
            self.last_t = t
        tau_u = self.compute_tau_u(dt)
        tau_r = self.compute_tau_r(dt)
        F1, F2 = thrust_allocation(tau_u, tau_r, self.b)
        lim = IMPL["thruster_limit"]
        self.F1, self.F2 = _clamp(F1, -lim, lim), _clamp(F2, -lim, lim)
        return self.F1, self.F2

    def compute_tau_u(self, dt):
        if dt <= 0:
            return self.tau_u
        self.speed_error = self.desired_speed - math.hypot(self.current_speed, self.current_sway)
        self.speed_error_integ = _clamp(self.speed_error_integ + self.speed_error * dt, -10.0, 10.0)
        tau_ff = (self.desired_speed * self.Xu) / (exp((self.Xu / self.m11) * 100) - 1)
        tau_d = self.kd_speed * (self.speed_error - self.last_speed_error) / dt
        self.last_speed_error = self.speed_error
        self.tau_u = (tau_ff + self.kp_speed * self.speed_error
                      + self.ki_speed * self.speed_error_integ + tau_d)
        return self.tau_u

    def compute_tau_r(self, dt):
        if dt <= 0:
            return self.tau_r
        self.heading_error = sawtooth(self.current_heading - self.desired_heading)
        self.heading_error_integ += self.heading_error * dt
        self.tau_r = (self.kp * self.heading_error + self.ki * self.heading_error_integ
                      + self.kd * (self.heading_error - self.last_heading_error) / dt)
        self.last_heading_error = self.heading_error
        return self.tau_r


def thrust_allocation(tau_u, tau_r, b):
    """F2 = tau_u/2 - tau_r/(2b),  F1 = tau_u - F2   (paper Eq. thrust_allocation)."""
    F2 = tau_u / 2 - tau_r / (2 * b)
    return tau_u - F2, F2


# ---------------------------------------------------------------------------
# One NR-PID scalar channel
# ---------------------------------------------------------------------------
class _Channel:
    def __init__(self, P, name):
        self.P = P
        self.name = name
        self.k = dict(p=P["base_gains"]["p"], i=P["base_gains"]["i"], d=P["base_gains"]["d"])
        self.Theta = P["theta0"]
        self.integ = 0.0
        self.last_e = 0.0
        self.e_dot = 0.0
        self.sigma = 0.0
        self.Kp = self.Kd = self.Ki = 0.0

    @staticmethod
    def nl_gain(x, k, mu, eps):
        """K(.) = k |x|^(mu-1) if |x| > eps else k eps^(mu-1)."""
        a = abs(x)
        return k * (a ** (mu - 1) if a > eps else eps ** (mu - 1))

    def _adapt(self, key, signal, dt):
        P = self.P
        if self.k[key] >= P["k_max"][key]:           # projection (paper Eq. projection)
            return
        if abs(signal) < 1e-9:
            return
        kdot = P["gamma"][key] * self.sigma * signal * abs(signal) ** (P["alpha"][key] - 1.0)
        lo = IMPL["k_min"][self.name][key]
        self.k[key] = _clamp(self.k[key] + kdot * dt, lo, P["k_max"][key])

    def _theta_dot(self):
        P = self.P
        if self.Theta > P["theta_min"] + 1e-12:
            s = abs(self.sigma) - P["eps_theta"]
            return P["Gamma"] * abs(self.sigma) * (1.0 if s > 0 else -1.0 if s < 0 else 0.0)
        if abs(self.Theta - P["theta_min"]) <= 1e-12:
            return P["gamma_theta"]
        return P["theta_min"]

    def step(self, e, dt, smc=True):
        P = self.P
        # integral with anti-windup
        lim = IMPL["integral_limit"][self.name]
        self.integ = _clamp(self.integ + e * dt, -lim, lim)
        # filtered, clamped derivative
        raw = (e - self.last_e) / dt
        c = IMPL["deriv_clamp"][self.name]
        a = IMPL["deriv_filter_alpha"][self.name]
        self.e_dot = a * _clamp(raw, -c, c) + (1 - a) * self.e_dot
        self.last_e = e
        # sliding variable and adaptation
        self.sigma = self.e_dot + P["lam"] * e
        self._adapt("p", e, dt)
        self._adapt("d", self.e_dot, dt)
        self._adapt("i", self.integ, dt)
        # nonlinear gains
        self.Kp = self.nl_gain(e, self.k["p"], P["mu"]["p"], P["eps"]["p"])
        self.Kd = self.nl_gain(self.e_dot, self.k["d"], P["mu"]["d"], P["eps"]["d"])
        self.Ki = self.nl_gain(self.integ, self.k["i"], P["mu"]["i"], P["eps"]["i"])
        # sliding-mode gain (three-case projection) and smooth saturation
        if smc:
            self.Theta = _clamp(self.Theta + self._theta_dot() * dt, P["theta_min"], P["theta_max"])
            smc_term = self.Theta * math.tanh(self.sigma / P["Phi"])
        else:
            smc_term = 0.0
        tau = self.Kp * e + self.Kd * self.e_dot + self.Ki * self.integ + smc_term
        self.smc_term = smc_term
        return _clamp(tau, -IMPL["tau_limit"][self.name], IMPL["tau_limit"][self.name])


# ---------------------------------------------------------------------------
# Proposed controller
# ---------------------------------------------------------------------------
class NRPID:
    smc = True

    def __init__(self, m11, Xu, b, heading=HEADING, speed=SPEED, heading_sign=1.0):
        self.m11, self.Xu, self.b = m11, Xu, b
        self.heading_sign = heading_sign
        self.desired_heading = self.current_heading = 0.0
        self.desired_speed = self.current_speed = self.current_sway = 0.0
        self.psi = _Channel(heading, "heading")
        self.u = _Channel(speed, "speed")
        self.heading_error = self.speed_error = 0.0
        self.tau_u = self.tau_r = 0.0
        self.F1 = self.F2 = 0.0
        self.last_t = None

    def update(self, dt=None):
        if dt is None:
            t = time.time()
            dt = 0.0 if self.last_t is None else t - self.last_t
            self.last_t = t
        if dt > 0:
            self.speed_error = self.desired_speed - math.hypot(self.current_speed, self.current_sway)
            self.tau_u = self.u.step(self.speed_error, dt, self.smc)
            self.heading_error = sawtooth(self.current_heading - self.desired_heading) * self.heading_sign
            self.tau_r = self.psi.step(self.heading_error, dt, self.smc)
        F1, F2 = thrust_allocation(self.tau_u, self.tau_r, self.b)
        lim = IMPL["thruster_limit"]
        self.F1, self.F2 = _clamp(F1, -lim, lim), _clamp(F2, -lim, lim)
        return self.F1, self.F2


class NLPID(NRPID):
    """Ablation used in the field comparison: full adaptation, no sliding-mode term."""
    smc = False
