import warnings
from datetime import datetime
from typing import Any, Union

import numpy as np
from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.default import NoInput, _drb, _make_py_json
from rateslib.dual import Dual, DualTypes, _get_adorder, gradient
from rateslib.rs import Ccy, FXRate
from rateslib.rs import FXRates as FXRatesObj

"""
.. ipython:: python
   :suppress:

   from rateslib.curves import Curve
   from rateslib.fx import FXRates
   from datetime import datetime as dt
"""


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FXRates:
    """
    Object to store and calculate FX rates for a consistent settlement date.

    Parameters
    ----------
    fx_rates : dict
        Dict whose keys are 6-character domestic-foreign currency pairs, and whose
        values are the relevant rates.
    settlement : datetime, optional
        The settlement date for the FX rates.
    base : str, optional
        The base currency (3-digit code). If not given defaults to either:

        - the base currency defined in `defaults`, if it is present in the list of currencies,
        - the first currency detected.

    Notes
    -----

    .. note::
       When this class uses ``Dual`` numbers to represent sensitivities of values to
       certain FX rates the variable names are called `"fx_domfor"` where `"dom"`
       is a domestic currency and `"for"` is a foreign currency. See the examples
       contained in class methods for clarification.

    Examples
    --------
    An FX rates market of *n* currencies is completely defined by *n-1*
    independent FX pairs.

    Below we define an FX rates market in 4 currencies with 3 FX pairs,

    .. ipython:: python

       fxr = FXRates({"eurusd": 1.1, "gbpusd": 1.25, "usdjpy": 100})
       fxr.currencies
       fxr.rate("gbpjpy")

    Ill defined FX markets will raise ``ValueError`` and are either **overspecified**,

    .. ipython:: python

       try:
           FXRates({"eurusd": 1.1, "gbpusd": 1.25, "usdjpy": 100, "gbpjpy": 125})
       except ValueError as e:
           print(e)

    or are **underspecified**,

    .. ipython:: python

       try:
           FXRates({"eurusd": 1.1, "gbpjpy": 125})
       except ValueError as e:
           print(e)

    or use redundant, co-dependent information,

    .. ipython:: python

       try:
           FXRates({"eurusd": 1.1, "usdeur": 0.90909, "gbpjpy": 125})
       except ValueError as e:
           print(e)

    """

    def __init__(
        self,
        fx_rates: dict[str, DualTypes],
        settlement: Union[datetime, NoInput] = NoInput(0),
        base: Union[str, NoInput] = NoInput(0),
    ):
        settlement = _drb(None, settlement)
        fx_rates_ = [FXRate(k[0:3], k[3:6], v, settlement) for k, v in fx_rates.items()]
        if base is NoInput(0):
            default_ccy = defaults.base_currency.lower()
            if any([default_ccy in k.lower() for k in fx_rates.keys()]):
                base_ = Ccy(defaults.base_currency)
            else:
                base_ = None
        else:
            base_ = Ccy(base)
        self.obj = FXRatesObj(fx_rates_, base_)
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

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        obj = FXRates.__init_from_obj__(self.obj.__copy__())
        obj.__init_post_obj__()
        return obj

    @property
    def fx_array(self):
        if self._fx_array is None:
            self._fx_array = np.array(self.obj.fx_array)
        return self._fx_array

    @property
    def base(self):
        return self.obj.base.name

    @property
    def settlement(self):
        return self.obj.fx_rates[0].settlement

    @property
    def pairs(self):
        return [fxr.pair for fxr in self.obj.fx_rates]

    @property
    def fx_rates(self):
        return {fxr.pair: fxr.rate for fxr in self.obj.fx_rates}

    @property
    def currencies_list(self):
        return [ccy.name for ccy in self.obj.currencies]

    @property
    def q(self):
        return len(self.obj.currencies)

    @property
    def fx_vector(self):
        return self.fx_array[0, :]

    @property
    def pairs_settlement(self):
        return {k: self.settlement for k in self.pairs}

    @property
    def variables(self):
        return tuple(f"fx_{pair}" for pair in self.pairs)

    @property
    def _ad(self):
        return self.obj.ad

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
        return self.fx_array[domi][fori]

    def restate(self, pairs: list[str], keep_ad: bool = False):
        """
        Create a new :class:`FXRates` class using other (or fewer) currency pairs as majors.

        Parameters
        ----------
        pairs : list of str
            The new currency pairs with which to define the ``FXRates`` class.
        keep_ad : bool, optional
            Keep the original derivative exposures defined by ``Dual``, instead
            of redefinition. It is advised against setting this to *True*, it is mainly used
            internally.

        Returns
        --------
        FXRates

        Notes
        -----
        This will redefine the pairs to which delta risks are expressed in ``Dual``
        outputs.

        If ``pairs`` match the existing object and ``keep_ad`` is
        requested then the existing object is returned unchanged as new copy.

        Examples
        --------
        Re-expressing an *FXRates* class with new majors, to which *Dual* sensitivities are
        measured.

        .. ipython:: python

           fxr = FXRates({"eurgbp": 0.9, "gbpjpy": 125, "usdjpy": 100})
           fxr.convert(100, "gbp", "usd")
           fxr2 = fxr.restate(["eurusd", "gbpusd", "usdjpy"])
           fxr2.convert(100, "gbp", "usd")

        Extracting an *FXRates* subset from a larger object.

        .. ipython:: python

           fxr = FXRates({"eurgbp": 0.9, "gbpjpy": 125, "usdjpy": 100, "audusd": 0.85})
           fxr2 = fxr.restate({"eurusd", "gbpusd"})
           fxr2.rates_table()
        """
        if set(pairs) == set(self.pairs) and keep_ad:
            return self.__copy__()  # no restate needed but return new instance

        restated_fx_rates = FXRates(
            {pair: self.rate(pair) if keep_ad else self.rate(pair).real for pair in pairs},
            settlement=self.settlement,
            base=self.base,
        )
        return restated_fx_rates

    def update(self, fx_rates: Union[dict, NoInput] = NoInput(0)):
        """
        Update all or some of the FX rates of the instance with new market data.

        Parameters
        ----------
        fx_rates : dict, optional
            Dict whose keys are 6-character domestic-foreign currency pairs and
            which are present in FXRates.pairs, and whose
            values are the relevant rates to update.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. It
           is **best practice** to create objects and any associations and then use the
           ``update`` methods to push new market data to them. Recreating objects with
           new data will break object-oriented associations and possibly lead to
           undetected market data based pricing errors.

        Suppose an *FXRates* class has been instantiated and resides in memory.

        .. ipython:: python

           fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.25}, settlement=dt(2022, 1, 3), base="usd")
           id(fxr)

        This object may be linked to others, probably an :class:`~rateslib.fx.FXForwards` class.
        It can be updated with some new market data. This will preserve its memory id and
        association with other objects (however, any linked objects should also be updated to
        cascade new calculations).

        .. ipython:: python

           linked_obj = fxr
           fxr.update({"eurusd": 1.06})
           id(fxr)  # <- SAME as above
           linked_obj.rate("eurusd")

        Do **not** do the following because overwriting a variable name will not eliminate the
        previous object from memory. Linked objects will still refer to the previous *FXRates*
        class still in memory.

        .. ipython:: python

           fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3), base="usd")
           id(fxr)  # <- NEW memory id, linked objects still associated with old fxr in memory
           linked_obj.rate("eurusd")  # will NOT return rate from the new `fxr` object

        Examples
        --------

        .. ipython:: python

           fxr = FXRates({"usdeur": 0.9, "eurnok": 8.5})
           fxr.rate("usdnok")
           fxr.update({"usdeur": 1.0})
           fxr.rate("usdnok")
        """
        if fx_rates is NoInput.blank:
            return None
        fx_rates_ = [FXRate(k[0:3], k[3:6], v, self.settlement) for k, v in fx_rates.items()]
        self.obj.update(fx_rates_)
        self.__init_post_obj__()

    def convert(
        self,
        value: Union[Dual, float],
        domestic: str,
        foreign: Union[str, NoInput] = NoInput(0),
        on_error: str = "ignore",
    ):
        """
        Convert an amount of a domestic currency into a foreign currency.

        Parameters
        ----------
        value : float or Dual
            The amount of the domestic currency to convert.
        domestic : str
            The domestic currency (3-digit code).
        foreign : str, optional
            The foreign currency to convert to (3-digit code). Uses instance
            ``base`` if not given.
        on_error : str in {"ignore", "warn", "raise"}
            The action taken if either ``domestic`` or ``foreign`` are not contained
            in the FX framework. `"ignore"` and `"warn"` will still return `None`.

        Returns
        -------
        Dual or None

        Examples
        --------

        .. ipython:: python

           fxr = FXRates({"usdnok": 8.0})
           fxr.convert(1000000, "nok", "usd")
           fxr.convert(1000000, "nok", "inr")  # <- returns None, "inr" not in fxr.

        """
        foreign = self.base if foreign is NoInput.blank else foreign.lower()
        domestic = domestic.lower()
        for ccy in [domestic, foreign]:
            if ccy not in self.currencies:
                if on_error == "ignore":
                    return None
                elif on_error == "warn":
                    warnings.warn(
                        f"'{ccy}' not in FXRates.currencies: returning None.",
                        UserWarning,
                    )
                    return None
                else:
                    raise ValueError(f"'{ccy}' not in FXRates.currencies.")

        i, j = self.currencies[domestic.lower()], self.currencies[foreign.lower()]
        return value * self.fx_array[i, j]

    def convert_positions(
        self,
        array: Union[np.ndarray, list],
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Convert an array of currency cash positions into a single base currency.

        Parameters
        ----------
        array : list, 1d ndarray of floats, or Series
            The cash positions to simultaneously convert in the base currency. **Must**
            be ordered by currency as defined in the attribute ``FXRates.currencies``.
        base : str, optional
            The currency to convert to (3-digit code). Uses instance ``base`` if not
            given.

        Returns
        -------
        Dual

        Examples
        --------

        .. ipython:: python

           fxr = FXRates({"usdnok": 8.0})
           fxr.currencies
           fxr.convert_positions([0, 1000000], "usd")
        """
        base = self.base if base is NoInput.blank else base.lower()
        array_ = np.asarray(array)
        j = self.currencies[base]
        return np.sum(array_ * self.fx_array[:, j])

    def positions(
        self,
        value,
        base: Union[str, NoInput] = NoInput(0),
    ):
        """
        Convert a base value with FX rate sensitivities into an array of cash positions.

        Parameters
        ----------
        value : float or Dual
            The amount expressed in base currency to convert to cash positions.
        base : str, optional
            The base currency in which ``value`` is given (3-digit code). If not given
            assumes the ``base`` of the object.

        Returns
        -------
        Series

        Examples
        --------
        .. ipython:: python

           fxr = FXRates({"usdnok": 8.0})
           fxr.positions(Dual(125000, ["fx_usdnok"], [-15625]), "usd")
           fxr.positions(100, base="nok")

        """
        if isinstance(value, (float, int)):
            value = Dual(value, [], [])
        base = self.base if base is NoInput.blank else base.lower()
        _ = np.array([0 if ccy != base else float(value) for ccy in self.currencies_list])
        for pair in value.vars:
            if pair[:3] == "fx_":
                delta = gradient(value, [pair])[0]
                _ += self._get_positions_from_delta(delta, pair[3:], base)
        return Series(_, index=self.currencies_list)

    def _get_positions_from_delta(self, delta: float, pair: str, base: str):
        """Return an array of cash positions determined from an FX pair delta risk."""
        b_idx = self.currencies[base]
        domestic, foreign = pair[:3], pair[3:]
        d_idx, f_idx = self.currencies[domestic], self.currencies[foreign]
        _ = np.zeros(self.q)

        # f_val = -delta * float(self.fx_array[b_idx, d_idx]) * float(self.fx_array[d_idx,f_idx])**2
        # _[f_idx] = f_val
        # _[d_idx] = -f_val / float(self.fx_array[d_idx, f_idx])
        # return _
        f_val = delta * float(self.fx_array[b_idx, f_idx])
        _[d_idx] = f_val
        _[f_idx] = -f_val / float(self.fx_array[f_idx, d_idx])
        return _  # calculation is more efficient from a domestic pov than foreign

    def rates_table(self):
        """
        Return a DataFrame of all FX rates in the object.

        Returns
        -------
        DataFrame
        """
        return DataFrame(
            np.vectorize(float)(self.fx_array),
            index=self.currencies_list,
            columns=self.currencies_list,
        )

    def _set_ad_order(self, order):
        """
        Change the node values to float, Dual or Dual2 based on input parameter.
        """

        self.obj.set_ad_order(_get_adorder(order))
        self.__init_post_obj__()

    def to_json(self):
        return _make_py_json(self.obj.to_json(), "FXRates")


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
