from rateslib.fxdev.rs import FXRates as FXRatesObj
from rateslib.fxdev.rs import FXRate, Ccy
from rateslib.dual import DualTypes
from rateslib.default import NoInput
from rateslib.json import _make_py_json
from datetime import datetime
from typing import Union


class FXRates:

    def __init__(
        self,
        fx_rates: dict[str, DualTypes],
        settlement: datetime,
        base: Union[str, NoInput] = NoInput(0),
        json_obj: Union[FXRatesObj, NoInput] = NoInput(0),
    ):
        if json_obj is not NoInput.blank:
            self.obj = json_obj
            self.__init_from_obj__()
        fx_rates_ = [FXRate(k[0:3], k[3:6], v, settlement) for k, v in fx_rates.items()]
        base_ = None if base is NoInput(0) else Ccy(base)
        self.obj = FXRatesObj(fx_rates_, settlement, base_)
        self.__init_from_obj__()

    def __init_from_obj__(self):
        self.currencies = {ccy.name: i for (i, ccy) in enumerate(self.obj.currencies)}
        self._fx_array = None

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