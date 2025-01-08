from __future__ import annotations  # type hinting

from typing import Any

from rateslib.dual import DualTypes

STATE_MAP = {
    1: ["SUCCESS", "`conv_tol` reached"],
    2: ["SUCCESS", "`func_tol` reached"],
    3: ["SUCCESS", "closed form valid"],
    -1: ["FAILURE", "`max_iter` breached"],
}


def _solver_result(
    state: int, i: int, func_val: DualTypes, time: float, log: bool, algo: str
) -> dict[str, Any]:
    if log:
        print(
            f"{STATE_MAP[state][0]}: {STATE_MAP[state][1]} after {i} iterations "
            f"({algo}), `f_val`: {func_val}, "
            f"`time`: {time:.4f}s",
        )
    return {
        "status": STATE_MAP[state][0],
        "state": state,
        "g": func_val,
        "iterations": i,
        "time": time,
    }
