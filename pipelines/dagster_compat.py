"""
Optional-Dagster compatibility shims.

The lean serving image (``serving/Dockerfile``) intentionally omits Dagster —
the API container only needs the feature-engineering *functions*, not the
orchestrator. Asset/resource modules import Dagster names from here instead
of from ``dagster`` directly, so they stay importable in both environments:

- Full install (Dagster present): the real Dagster names are re-exported,
  behaviour is unchanged.
- Lean install (Dagster absent): harmless no-op fallbacks are provided —
  ``@asset`` becomes a pass-through decorator and ``build_asset_context()``
  returns a tiny shim with the ``log`` / ``add_output_metadata`` surface
  the assets use.
"""

from __future__ import annotations

try:
    from dagster import ConfigurableResource, asset, build_asset_context

    DAGSTER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in the lean image
    DAGSTER_AVAILABLE = False

    def asset(*dargs, **dkwargs):
        """No-op stand-in for ``dagster.asset`` — returns the function unchanged."""

        def _decorator(fn):
            return fn

        # Support bare ``@asset`` (no parens) as well as ``@asset(...)``.
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return _decorator

    class _ShimLog:
        """Minimal ``context.log`` replacement that prints to stdout."""

        @staticmethod
        def info(msg, *args, **kwargs) -> None:
            print(f"[dagster-shim] {msg}")

        @staticmethod
        def warning(msg, *args, **kwargs) -> None:
            print(f"[dagster-shim] WARNING: {msg}")

    class _ShimContext:
        """Minimal asset-execution context for direct function calls."""

        log = _ShimLog()

        def add_output_metadata(self, metadata: dict) -> None:
            # Nowhere to record metadata outside a Dagster run — ignore.
            pass

    def build_asset_context(*args, **kwargs) -> _ShimContext:
        return _ShimContext()

    from pydantic import BaseModel

    class ConfigurableResource(BaseModel):  # type: ignore[no-redef]
        """Minimal stand-in so resource definitions stay importable."""


__all__ = [
    "DAGSTER_AVAILABLE",
    "ConfigurableResource",
    "asset",
    "build_asset_context",
]
