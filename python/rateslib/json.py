from rateslib.rs import from_json as from_json_rs
from rateslib import FXRates

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
    if json[:5] == '{"Py_':
        class_name = json[5:json[5:].find('"')+5]
        parsed_json = json[7+len(class_name):-2]
        return locals()[class_name](json_obj=from_json_rs(parsed_json))
    return from_json_rs(json)


def _make_py_json(json, class_name):
    return f'{{"Py_{class_name}":' + json + '}'
