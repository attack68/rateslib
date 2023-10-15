from rateslib.default import NoInput
from rateslib.fx import FXRates
from typing import Any, Union
from datetime import datetime
import json


class Serialise:
    """
    Class to create a string based representation of rateslib objects for network transmittance.

    Parameters
    ----------
    obj : Any
        A valid rateslib *Curve* type, *FXRates* or *FXForwards* instance, *Instrument* or
        *Solver*.
    """

    def __init__(self, obj: Any):
        self.obj = obj

    def to_json(self):
        type_str = type(self.obj).__name__
        method = getattr(self, f"_{type_str}_to", None)
        if method is None:
            raise NotImplementedError(
                f"The class {type_str} does not yet have a JSON implementation"
            )
        return method()

    @classmethod
    def from_json(cls, obj: Any, data: str):
        """
        Create a *rateslib* class object from a JSON string.

        Parameters
        ----------
        obj : Any
            The *rateslib* class object to create. E.g. "IRS".
        data : str
            The data to constitute the object from in JSON format.

        Returns
        -------
        Any

        Examples
        --------
        """
        klass = obj.__name__
        method = getattr(cls, f"_{klass}_from", None)
        if method is None:
            return NotImplementedError(
                f"The class {klass} does not yet have a JSON implementation"
            )
        return method(data)

    def _FXRates_to(self):
        return json.dumps(
            {
                "fx_rates": {k: float(v) for k, v in self.obj.fx_rates.items()},
                "settlement": _date_or_noinput_to(self.obj.settlement),
                "base": self.obj.base
            },
            default=str
        )

    @classmethod
    def _FXRates_from(cls, data: str):
        kwargs = json.loads(data)
        kwargs["settlement"] = _date_or_noinput_from(kwargs["settlement"])
        return FXRates(**kwargs)


def _date_or_noinput_to(date: datetime):
    if date is NoInput.blank:
        return None
    elif isinstance(date, datetime):
        return f"{date.strftime('%Y-%m-%d')}"
    else:
        raise ValueError(f"Date must be NoInput or datetime object, got type {type(date)}")


def _date_or_noinput_from(date: Union[str, None]):
    if date is None:
        return NoInput(0)
    else:
        return datetime.strptime(date, "%Y-%m-%d")
