"""
Synthetic vibration channel generator.

Models radial vibration as:
    v(t) = A_base * sin(2π * f_base * t + φ)
          + A_harmonics * sin(2π * f_harmonic * t)
          + gaussian_noise
          [+ anomaly_spike  if anomaly_injected]

Physics rationale:
- Base frequency tied to rotational speed (1× running speed component)
- Harmonic at 3× simulates bearing defect frequency
- Noise floor σ ≈ 0.05 m/s²
- Anomaly spike: 5–10× normal amplitude, mimicking imbalance / bearing fault onset
"""

from __future__ import annotations

import math
import random


def _rpm_to_hz(rpm: float) -> float:
    return rpm / 60.0


def generate_vibration(
    *,
    rotational_speed_rpm: float,
    tool_wear_min: float,
    machine_failure: bool,
    t: float,
    rng: random.Random | None = None,
) -> float:
    """
    Generate a single vibration sample [m/s²] for a given machine state.

    Parameters
    ----------
    rotational_speed_rpm : Rotational speed — sets the base vibration frequency.
    tool_wear_min        : Wear level — gradually increases baseline amplitude.
    machine_failure      : Whether a failure is active — injects anomaly spike.
    t                    : Elapsed time [s] since adapter start (for sinusoid phase).
    rng                  : Optional seeded RNG for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    f_base = _rpm_to_hz(rotational_speed_rpm)
    f_harmonic = 3.0 * f_base

    # Baseline amplitude grows slightly with tool wear (0.2 → 0.5 m/s² over full wear)
    wear_factor = 0.2 + 0.3 * (tool_wear_min / 253.0)

    base = wear_factor * math.sin(2 * math.pi * f_base * t)
    harmonic = 0.08 * math.sin(2 * math.pi * f_harmonic * t)
    noise = rng.gauss(0.0, 0.05)

    # Anomaly spike: raised amplitude + phase offset
    spike = 0.0
    if machine_failure:
        spike = (
            rng.uniform(5.0, 10.0) * wear_factor * math.sin(2 * math.pi * f_base * t + math.pi / 4)
        )

    return round(base + harmonic + noise + spike, 4)
