"""
SensorOps Dagster pipeline package.

Exports the top-level Definitions object that Dagster reads on startup.
"""

from pipelines.definitions import defs

__all__ = ["defs"]
