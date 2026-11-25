"""
Third-party extensions through fixed input/output contracts.

Three contracts, each one small Python file dropped in `~/.mlti_cal/plugins/`
(or a folder named by `MLTI_CAL_PLUGIN_PATH`); the full statement is in
`docs/extending.md`, templates in `examples/plugins/`:

    A  detector   detect(image, settings) -> (N, 3) float [target_id, x, y]
    B  object     a points file, one `id, x, y, z` line per target (metres)
    C  camera     project(params, points) -> (N, 2) pixels

`contract.py` is the gate every plug-in result passes through. Nothing in this
package imports the detectors or the models at import time: discovery in
`detectors/registry.py` imports `loader` lazily, so the package can be
imported from anywhere without a cycle.
"""

from mlti_cal.plugins.contract import PLUGIN_API, ContractError

__all__ = ["PLUGIN_API", "ContractError"]
