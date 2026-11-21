"""Person B's model of a sideways step (HANDOFF_PERSON_B.md, MESSAGES.md 16:15 UTC): after a delay d the sideways speed
rises to the commanded vel_y with time constant tau, from d 0.05 s and tau 0.3 s. Compared here with the saved
step-response runs (artifacts/env/dodge-step-response): the sideways displacement in the decision instant's heading frame
minus the vel_y 0 run at the same forward speed, per unit of commanded vel_y, over the first 1.0 s; and the (d, tau)
that fits all 12 runs best in least squares.
Run from the repository root: uv run python -X utf8 artifacts/env/research/step_fit.py
"""
import numpy as np
from scipy.optimize import least_squares

runs = []
for vx in (0.3, 0.6, 0.9):
    base = np.load(f"artifacts/env/dodge-step-response/vx{vx}_vy0.0.npz")
    for vy in (1.0, -1.0, 0.5, -0.5):
        run = np.load(f"artifacts/env/dodge-step-response/vx{vx}_vy{vy}.npz")
        first = run["t_s"] <= 1.0 + 1e-9
        runs.append((run["t_s"][first], (run["side_displacement_m"] - base["side_displacement_m"])[first] / vy))


def displacement(p, t):  # per unit command: integral of (1 - exp(-(t - d) / tau)) from d to t
    d, tau = p
    s = np.clip(t - d, 0, None)
    return s - tau * (1 - np.exp(-s / tau))


def residuals(p):
    return np.concatenate([displacement(p, t) - y for t, y in runs])


fit = least_squares(residuals, [0.05, 0.3], bounds=([0.0, 0.01], [0.5, 2.0]))
for name, p in (("Person B's d 0.05 s, tau 0.3 s", [0.05, 0.3]), (f"fit d {fit.x[0]:.3f} s, tau {fit.x[1]:.3f} s", fit.x)):
    rms = np.sqrt(np.mean(residuals(p) ** 2))
    at = [round(float(displacement(p, np.array([t]))[0]), 3) for t in (0.2, 0.4, 0.6, 1.0)]
    print(f"{name}: rms error {rms:.4f} m per m/s; displacement per m/s at 0.2/0.4/0.6/1.0 s {at}")
for t in (0.2, 0.4, 0.6, 1.0):
    measured = [float(y[np.abs(tt - t).argmin()]) for tt, y in runs]
    print(f"measured displacement per m/s at {t} s over the 12 runs: {min(measured):.3f} to {max(measured):.3f}")
