from rateslib.rs import from_json as from_json_rs
from rateslib.fxdev import FXRates

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
        parsed_json = json[7+len(class_name):-1]
        objs = globals()
        class_obj = objs[class_name]
        return class_obj.__init_from_rs__(obj=from_json_rs(parsed_json))
    return from_json_rs(json)
