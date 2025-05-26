from __future__ import annotations

from json import dumps, loads

# globals namespace
from typing import TYPE_CHECKING, Any

from rateslib.curves.curves import _CurveMeta
from rateslib.curves.rs import CurveRs
from rateslib.default import NoInput
from rateslib.fx import FXRates
from rateslib.rs import from_json as from_json_rs

if TYPE_CHECKING:
    pass

NAMES_RsPy = {  # this is a mapping of native Rust obj names to Py obj names
    "FXRates": FXRates,
    "Curve": CurveRs,
}


class NoInputFromJson:
    @classmethod
    def _from_json(cls, val) -> NoInput:
        return NoInput(val)


NAMES_Py = {  # this is a mapping of native Python object classes
    "_CurveMeta": _CurveMeta,
    "NoInput": NoInputFromJson,
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
    obj = loads(json)
    if isinstance(obj, dict):
        if "PyWrapped" in obj:
            # then object is a Rust struct wrapped by a Python class.
            # determine the Python class name and reconstruct the Python class from the Rust struct.
            class_name = next(iter(obj["PyWrapped"].keys()))
            restructured_json = dumps(obj["PyWrapped"])
            # objs = globals()
            class_obj = NAMES_RsPy[class_name]
            return class_obj.__init_from_obj__(obj=from_json_rs(restructured_json))  # type: ignore[attr-defined]
        elif "PyNative" in obj:
            class_name = next(iter(obj["PyNative"].keys()))
            class_obj = NAMES_Py[class_name]
            return class_obj._from_json(obj["PyNative"][class_name])
        else:
            # the dict may have been a native Rust object, try loading directly
            # this will raise if all combination exhausted
            return from_json_rs(json)
    else:
        # object is a native Python element
        return obj
