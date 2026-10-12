"""NR-PID parameters used in the field trials (paper Table `nrpid_params`).

Each (p, d, i) triple is ordered (proportional, derivative, integral) as in the
paper.  Heading = psi channel (tau_r), speed = u channel (tau_u).
"""

HEADING = dict(
    lam=0.52,
    base_gains=dict(p=4.0, i=0.1, d=3.0),          # (k_p, k_i, k_d)
    mu=dict(p=0.50, d=0.51, i=0.62),
    eps=dict(p=0.12, d=0.10, i=0.25),
    gamma=dict(p=0.02, d=0.01, i=0.005),
    alpha=dict(p=0.50, d=0.51, i=0.45),
    k_max=dict(p=6.0, d=4.0, i=4.5),
    theta0=1.5, theta_min=0.3, theta_max=3.0,
    Gamma=0.1, eps_theta=0.2, gamma_theta=0.03,
    Phi=12.0,
)

SPEED = dict(
    lam=0.63,
    base_gains=dict(p=60.0, i=0.5, d=8.0),
    mu=dict(p=0.40, d=0.50, i=0.55),
    eps=dict(p=0.20, d=0.12, i=0.25),
    gamma=dict(p=0.30, d=0.05, i=0.01),
    alpha=dict(p=0.49, d=0.53, i=0.60),
    k_max=dict(p=120.0, d=20.0, i=8.0),
    theta0=0.4, theta_min=0.02, theta_max=2.0,
    Gamma=0.04, eps_theta=0.25, gamma_theta=0.01,
    Phi=5.0,
)

# Implementation safeguards (NOT part of the analysed control law; the paper
# states them in the "Implementation details" paragraph).
IMPL = dict(
    deriv_filter_alpha=dict(heading=0.18, speed=0.20),   # low-pass on e_dot
    deriv_clamp=dict(heading=2.2, speed=3.5),            # |e_dot| limit
    integral_limit=dict(heading=5.0, speed=10.0),        # anti-windup on int(e)
    k_min=dict(heading=dict(p=0.5, d=0.5, i=0.0),        # lower bound on gains
               speed=dict(p=1.0, d=1.0, i=0.0)),
    tau_limit=dict(heading=50.0, speed=150.0),
    thruster_limit=20.0,                                  # |F1|,|F2| <= 20 N
)
