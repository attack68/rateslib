from datetime import datetime
from typing import Any, Union

from rateslib.default import NoInput, _make_py_json
from rateslib.dual import DualTypes
from rateslib.fxdev.rs import Ccy, FXRate
from rateslib.fxdev.rs import FXRates as FXRatesObj


class FXRates:

    def __init__(
        self,
        fx_rates: dict[str, DualTypes],
        settlement: datetime,
        base: Union[str, NoInput] = NoInput(0),
    ):
        fx_rates_ = [FXRate(k[0:3], k[3:6], v, settlement) for k, v in fx_rates.items()]
        base_ = None if base is NoInput(0) else Ccy(base)
        self.obj = FXRatesObj(fx_rates_, settlement, base_)
        self.__init_post_obj__()

    @classmethod
    def __init_from_obj__(cls, obj):
        """Construct the class instance from a given rust object which is wrapped."""
        # create a default instance and overwrite it
        new = cls({"usdeur": 1.0}, datetime(2000, 1, 1))
        new.obj = obj
        new.__init_post_obj__()
        return new

    def __init_post_obj__(self):
        self.currencies = {ccy.name: i for (i, ccy) in enumerate(self.obj.currencies)}
        self._fx_array = None

    def __eq__(self, other: Any):
        if isinstance(other, FXRates):
            return self.obj == other.obj
        return False

    @property
    def fx_array(self):
        if self._fx_array is None:
            self._fx_array = self.obj.fx_array
        return self._fx_array

    def rate(self, pair: str) -> DualTypes:
        """
        Return a specified FX rate for a given currency pair.

        Parameters
        ----------
        pair : str
            The FX pair in usual domestic:foreign convention (6 digit code).

        Returns
        -------
        Dual

        Examples
        --------

        .. ipython:: python

           fxr = FXRates({"usdeur": 2.0, "usdgbp": 2.5})
           fxr.rate("eurgbp")
        """
        domi, fori = self.currencies[pair[:3].lower()], self.currencies[pair[3:].lower()]
        # domi, fori = self.obj.get_ccy_index(Ccy(pair[:3])), self.obj.get_ccy_index(Ccy(pair[3:]))
        return self.fx_array[domi][fori]

    def to_json(self):
        return _make_py_json(self.obj.to_json(), "FXRates")
