from rateslib.default import NoInput
from rateslib.fx import FXRates
from rateslib.curves import Curve
from rateslib.calendars import create_calendar, Holiday
from typing import Any, Union
from datetime import datetime
import json


class Serialise:
    """
    Class to create a string based representation of rateslib objects for database storage or
    network transmission.

    Parameters
    ----------
    obj : Any
        A valid rateslib *Curve* type, *FXRates* or *FXForwards* instance, *Instrument* or
        *Solver*.
    """

    def __init__(self, obj: Any):
        self.obj = obj
        self.name = type(obj).__name__
        self._to_json = getattr(self, f"_{self.name}_to", None)
        if self._to_json is None:
            raise NotImplementedError(
                f"The class {self.name} does not yet have a JSON implementation."
            )

    def to_json(self):
        """
        Convert the *rateslib* object to JSON format for database storage or network transmit.

        Returns
        -------
        str
        """
        return self._to_json()

    @classmethod
    def from_json(cls, obj: Any, data: str):
        """
        Create a *rateslib* class object from a JSON string.

        Parameters
        ----------
        obj : Any, str
            The *rateslib* class object to create. E.g. "IRS" or :class:`~rateslib.instruments.IRS`
        data : str
            The data to constitute the object from in JSON format.

        Returns
        -------
        Any

        Examples
        --------
        """
        klass = obj if isinstance(obj, str) else obj.__name__
        method = getattr(cls, f"_{klass}_from", None)
        if method is None:
            return NotImplementedError(
                f"The class {klass} does not yet have a JSON implementation"
            )
        return method(data)

    def _FXRates_to(self):
        data = {
            "fx_rates": {k: float(v) for k, v in self.obj.fx_rates.items()},
            "settlement": _date_or_noinput_to(self.obj.settlement),
            "base": self.obj.base
        }
        return json.dumps(data, default=str)

    @classmethod
    def _FXRates_from(cls, data: str):
        kwargs = json.loads(data)
        kwargs["settlement"] = _date_or_noinput_from(kwargs["settlement"])
        return FXRates(**kwargs)

    def _Curve_data(self):
        return {
            "nodes": {dt.strftime("%Y-%m-%d"): float(v) for dt, v in self.obj.nodes.items()},
            "interpolation": self.obj.interpolation if isinstance(self.obj.interpolation, str) else None,
            "t": _datelist_or_noinput_to(self.obj.t),
            "c": self.obj.spline.c if self.obj.c_init else None,
            "id": self.obj.id,
            "convention": self.obj.convention,
            "endpoints": self.obj.spline_endpoints,
            "modifier": self.obj.modifier,
            "calendar_type": self.obj.calendar_type,
            "ad": self.obj.ad,
        }

    def _Curve_to(self):
        data = self._Curve_data()
        if data["calendar_type"] == "null":
            data.update({"calendar": None})
        elif "named: " in data["calendar_type"]:
            data.update({"calendar": data.calendar_type[7:]})
        else:  # calendar type is custom
            data.update(
                {
                    "calendar": {
                        "weekmask": self.obj.calendar.weekmask,
                        "holidays": [
                            d.item().strftime("%Y-%m-%d")
                            for d in self.obj.calendar.holidays  # numpy/pandas timestamp to py
                        ],
                    }
                }
            )
        return json.dumps(data, default=str)

    @classmethod
    def _Curve_from(cls, data: str):
        kwargs = json.loads(data)
        kwargs["nodes"] = {
            datetime.strptime(dt, "%Y-%m-%d"): v for dt, v in kwargs["nodes"].items()
        }
        if kwargs["calendar_type"] == "custom":
            # must load and construct a custom holiday calendar from serial dates
            def parse(d: datetime):
                return Holiday("", year=d.year, month=d.month, day=d.day)

            dates = [
                parse(datetime.strptime(d, "%Y-%m-%d")) for d in kwargs["calendar"]["holidays"]
            ]
            kwargs["calendar"] = create_calendar(
                rules=dates, weekmask=kwargs["calendar"]["weekmask"]
            )

        kwargs["t"] = _datelist_or_noinput_from(kwargs["t"])
        kwargs["c"] = NoInput(0) if kwargs["c"] is None else kwargs["c"]

        serial = {k: v for k, v in kwargs.items() if v is not None}
        return Curve(**{**serial, **kwargs})


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


def _datelist_or_noinput_to(datelist: datetime):
    if datelist is NoInput.blank:
        return None
    else:
        return [date.strftime('%Y-%m-%d') for date in datelist]


def _datelist_or_noinput_from(datelist: Union[str, None]):
    if datelist is None:
        return NoInput(0)
    else:
        return [datetime.strptime(date, "%Y-%m-%d") for date in datelist]
