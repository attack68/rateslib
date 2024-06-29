from rateslib.rs import from_json as from_json_rs
from rateslib.default import NoInput
from typing import Any


def from_json(json: str, klass: Any = NoInput(0)):
    """
    Create an object from JSON string.

    Parameters
    ----------
    json: str
        JSON string in appropriate format to construct the class.
    klass: Any, optional
        The known Python class to construct if the JSON contains an object with a Py wrapper.

    Returns
    -------
    Object
    """
    if json[:8] == '{"Py":{"':
        class_name, parsed_json = json[8:json[8:].find('"')+8], json[6:-1]
        # objs = globals()
        # class_obj = objs[class_name]
        if klass is NoInput.blank:
            raise TypeError(
                "Must provide a Python class in the `klass` argument when reconstructing from wrapped JSON"
            )
        return klass.__init_from_obj__(obj=from_json_rs(parsed_json))
    return from_json_rs(json)
