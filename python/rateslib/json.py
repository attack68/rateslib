# globals namespace
from rateslib.curves.rs import CurveRs
from rateslib.fx import FXRates
from rateslib.rs import from_json as from_json_rs

NAMES_RsPy = {  # this is a mapping of native Rust obj names to Py obj names
    "FXRates": FXRates,
    "Curve": CurveRs,
}


def from_json(json: str):
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
    if json[:8] == '{"Py":{"':
        class_name, parsed_json = json[8 : json[8:].find('"') + 8], json[6:-1]
        # objs = globals()
        class_obj = NAMES_RsPy[class_name]
        return class_obj.__init_from_obj__(obj=from_json_rs(parsed_json))
    return from_json_rs(json)
