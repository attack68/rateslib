from __future__ import annotations

# globals namespace
from typing import Any, TYPE_CHECKING
from json import loads, dumps

from rateslib.curves.rs import CurveRs
from rateslib.dual import Dual, Dual2
from rateslib.dual.utils import _to_number
from rateslib.fx import FXRates
from rateslib.rs import from_json as from_json_rs

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, Number

NAMES_RsPy = {  # this is a mapping of native Rust obj names to Py obj names
    "FXRates": FXRates,
    "Curve": CurveRs,
}


def from_json(json: str) -> Any:
    """
    Create an object from JSON string.

    Parameters
    ----------
    json: str
        JSON string in appropriate format to construct the class.

    Returns
    -------
    Object
    """
    try:
        return from_json_rs(json)
    except ValueError:
        # then object is not serialised from Serde in Rust
        if json[:15] == '{"PyWrapped":{"':
            # then object is a Rust struct wrapped by a Python class.
            # determine the Python class name and reconstruct the Python class from the Rust struct.
            class_name, parsed_json = json[15 : json[15:].find('"') + 15], json[13:-1]
            # objs = globals()
            class_obj = NAMES_RsPy[class_name]
            return class_obj.__init_from_obj__(obj=from_json_rs(parsed_json))  # type: ignore[attr-defined]

        # else use native Python json
        return loads(json)


# Dualtypes handles case of rust wrapped Dual/Dual2 datatype intermixed with float.

def _dualtypes_to_json(val: DualTypes) -> str:
    val_: Number = _to_number(val)
    if isinstance(val_, Dual | Dual2):
        return val_.to_json()
    else:
        return dumps(val_)
