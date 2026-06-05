import matplotlib

matplotlib.use('Agg')
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs('docs/images', exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')


def box(ax, x, y, w, h, color, label, sublabel=''):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.12', linewidth=1.5,
        edgecolor=color, facecolor=color + '22'
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.65, label, ha='center', va='center',
            color=color, fontsize=9.5, fontweight='bold')
    if sublabel:
        ax.text(x + w / 2, y + h * 0.28, sublabel, ha='center', va='center',
                color='#8b949e', fontsize=7.5)


def arrow(ax, x1, y1, x2, y2, color='#58a6ff'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# Title
ax.text(8, 8.55, 'SensorOps — Industrial AI Platform', ha='center', va='center',
        color='#e6edf3', fontsize=17, fontweight='bold')
ax.text(8, 8.15, 'Predictive Maintenance  ·  Industrie 4.0  ·  MLOps + LLM Integration',
        ha='center', va='center', color='#8b949e', fontsize=10)

# Layer labels (left margin)
ax.text(0.15, 6.8, 'LAYER 1  INGEST', color='#3fb950', fontsize=7.5, fontweight='bold', alpha=0.8)
ax.text(0.15, 4.8, 'LAYER 2  MLOPS', color='#58a6ff', fontsize=7.5, fontweight='bold', alpha=0.8)
ax.text(0.15, 2.8, 'LAYER 3  LLM', color='#d2a8ff', fontsize=7.5, fontweight='bold', alpha=0.8)
ax.text(0.15, 0.8, 'LAYER 4  COMPLIANCE', color='#ffa657', fontsize=7.5, fontweight='bold', alpha=0.8)

# Row 1 — Ingest
box(ax, 0.3, 5.6, 3.0, 1.0, '#3fb950', 'AI4I 2020 Dataset', '10k rows  3.5% failure rate')
box(ax, 3.8, 5.6, 3.0, 1.0, '#3fb950', 'CSV Replay Adapter', 'SensorEvent stream\nSynthetic vibration channel')
box(ax, 7.3, 5.6, 3.0, 1.0, '#3fb950', 'Pydantic Schema', 'Validated events\nUUID + UTC timestamp')
arrow(ax, 3.3, 6.1, 3.8, 6.1, '#3fb950')
arrow(ax, 6.8, 6.1, 7.3, 6.1, '#3fb950')

# Row 2 — MLOps
box(ax, 0.3, 3.5, 2.2, 1.05, '#58a6ff', 'Dagster', '4 assets + jobs\nSchedules + sensors')
box(ax, 2.9, 3.5, 2.2, 1.05, '#58a6ff', 'Feature Eng.', 'temp_delta  power\nvibration  wear_ratio')
box(ax, 5.5, 3.5, 2.2, 1.05, '#58a6ff', 'IF / LSTM / ESN', 'Anomaly scores [0,1]\nThreshold 0.6')
box(ax, 8.1, 3.5, 2.2, 1.05, '#58a6ff', 'MLflow', 'Experiment tracking\nModel registry')
box(ax, 10.7, 3.5, 2.2, 1.05, '#58a6ff', 'FastAPI', '/predict  /batch\nDocker  2 workers')
arrow(ax, 2.5, 4.0, 2.9, 4.0)
arrow(ax, 5.1, 4.0, 5.5, 4.0)
arrow(ax, 7.7, 4.0, 8.1, 4.0)
arrow(ax, 10.3, 4.0, 10.7, 4.0)

# Row 3 — LLM
box(ax, 0.3, 1.55, 3.0, 1.05, '#d2a8ff', 'ChromaDB RAG', 'Failure mode cards\nMaintenance manuals')
box(ax, 3.7, 1.55, 3.0, 1.05, '#d2a8ff', 'DeepSeek API', 'NL query engine\nIncident reporter')
box(ax, 7.1, 1.55, 3.0, 1.05, '#d2a8ff', 'Structured Reports', 'Root cause + actions\nEU AI Act audit fields')
arrow(ax, 3.3, 2.1, 3.7, 2.1, '#d2a8ff')
arrow(ax, 6.7, 2.1, 7.1, 2.1, '#d2a8ff')

# Row 4 — Compliance
box(ax, 0.3, 0.2, 2.8, 1.05, '#ffa657', 'OpenLineage', 'raw -> features -> scores\nMarquez / JSONL fallback')
box(ax, 3.5, 0.2, 2.8, 1.05, '#ffa657', 'EU AI Act Audit', 'Hash-chained JSONL\nmodel + dataset per pred')
box(ax, 6.7, 0.2, 2.8, 1.05, '#ffa657', 'GitHub Actions', 'Lint  test  docker\nScheduled pipeline')
box(ax, 9.9, 0.2, 2.8, 1.05, '#ffa657', 'Docker Compose', 'FastAPI + MLflow\n+ Dagster + Postgres')

plt.tight_layout()
plt.savefig('docs/images/architecture.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('architecture done')
