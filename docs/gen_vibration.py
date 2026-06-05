import matplotlib

matplotlib.use('Agg')
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random

from data.vibration import generate_vibration

os.makedirs('docs/images', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

t = np.linspace(0, 0.5, 500)

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Normal signal
rng = random.Random(42)
normal = [generate_vibration(rotational_speed_rpm=1500, tool_wear_min=50,
          machine_failure=False, t=ti, rng=rng) for ti in t]
axes[0].plot(t, normal, color='#3fb950', linewidth=1.2, alpha=0.9)
axes[0].set_title('Normal Operation — Vibration Signal', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Acceleration [m/s²]')
axes[0].set_ylim(-1.5, 1.5)

# Anomaly signal (failure mode)
rng2 = random.Random(42)
anomaly = [generate_vibration(rotational_speed_rpm=1500, tool_wear_min=230,
           machine_failure=True, t=ti, rng=rng2) for ti in t]
axes[1].plot(t, anomaly, color='#f85149', linewidth=1.2, alpha=0.9)
axes[1].axhline(max(normal)*0.8, color='#ffa657', linestyle='--', linewidth=1,
                label='Normal envelope', alpha=0.7)
axes[1].axhline(-max(normal)*0.8, color='#ffa657', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_title('Failure Mode — Anomalous Vibration (Tool Wear + Failure)', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Acceleration [m/s²]')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

plt.suptitle('Synthetic Vibration Channel — SensorOps', color='#e6edf3',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('docs/images/vibration_channel.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('vibration done')
