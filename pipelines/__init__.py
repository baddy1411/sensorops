"""
SensorOps Dagster pipeline package.

Exports the top-level Definitions object that Dagster reads on startup.
"""

try:
    # Full install: Dagster + all extras present.
    from pipelines.definitions import defs
except ImportError as _exc:  # Lean serving image: dagster/lineage not installed.
    defs = None
    print(f"[pipelines] Dagster definitions unavailable ({_exc}); running without orchestrator.")

__all__ = ["defs"]
