from __future__ import annotations

import warnings
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.default import (
    NoInput,
    _drb,
    _make_py_json,
)
from rateslib.dual import Dual, gradient
from rateslib.dual.utils import _get_adorder
from rateslib.mutability import (
    _clear_cache_post,
    _new_state_post,
    _WithState,
)
from rateslib.rs import Ccy, FXRate
from rateslib.rs import FXRates as FXRatesObj

if TYPE_CHECKING:
    from rateslib.typing import Arr1dF64, Arr1dObj, Arr2dObj, DualTypes, Number

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


class FXRates(_WithState):
    """
    Object to store and calculate FX rates for a consistent settlement date.

    Parameters
    ----------
    fx_rates : dict[str, float]
        Dict whose keys are 6-character currency pairs, and whose
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
       certain FX rates the variable names are called `"fx_cc1cc2"` where `"cc1"`
       is left hand currency and `"cc2"` is the right hand currency in the currency pair.
       See the examples contained in class methods for clarification.

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
        settlement: datetime | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        # Temporary declaration - will be overwritten
        self._currencies: dict[str, int] = {}

        settlement_: datetime | None = _drb(None, settlement)
        fx_rates_ = [FXRate(k[0:3], k[3:6], v, settlement_) for k, v in fx_rates.items()]
        if isinstance(base, NoInput):
            default_ccy = defaults.base_currency.lower()
            if any(default_ccy in k.lower() for k in fx_rates):
                base_ = Ccy(defaults.base_currency)
            else:
                base_ = None
        else:
            base_ = Ccy(base)
        self.obj = FXRatesObj(fx_rates_, base_)
        self.__init_post_obj__()
        self._clear_cache()
        self._set_new_state()

    @classmethod
    def __init_from_obj__(cls, obj: FXRatesObj) -> FXRates:
        """Construct the class instance from a given rust object which is wrapped."""
        # create a default instance and overwrite it
        new = cls({"usdeur": 1.0}, datetime(2000, 1, 1))
        new.obj = obj
        new.__init_post_obj__()
        return new

    def __init_post_obj__(self) -> None:
        self._currencies = {ccy.name: i for (i, ccy) in enumerate(self.obj.currencies)}

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FXRates):
            return self.obj == other.obj
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __copy__(self) -> FXRates:
        obj = FXRates.__init_from_obj__(self.obj.__copy__())
        obj.__init_post_obj__()
        return obj

    def __repr__(self) -> str:
        if len(self.currencies_list) > 5:
            return (
                f"<rl.FXRates:[{','.join(self.currencies_list[:2])},"
                f"+{len(self.currencies_list) - 2} others] at {hex(id(self))}>"
            )
        else:
            return f"<rl.FXRates:[{','.join(self.currencies_list)}] at {hex(id(self))}>"

    @cached_property
    def fx_array(self) -> Arr2dObj:
        """An array containing all of the FX pairs/crosses available on the object."""
        # caching this prevents repetitive data transformations between Rust/Python
        return np.array(self.obj.fx_array)  # type: ignore[return-value]

    def _fx_array_el(self, i: int, j: int) -> Number:
        # this is for typing since this numpy object array can only hold float | Dual | Dual2
        return self.fx_array[i, j]  # type: ignore

    @property
    def base(self) -> str:
        """The assumed base currency of the object which may be used as the default ``base``
        currency in ``npv`` calculations when otherwise omitted.

        The base currency has index 0 in the ``currencies`` dict and is that which the ``fx_vector``
        is defined relative to.
        """
        return self.obj.base.name

    @property
    def settlement(self) -> datetime:
        """The settlement date of the FX rates that define the object."""
        return self.obj.fx_rates[0].settlement

    @property
    def pairs(self) -> list[str]:
        """A list of the currency pairs that define the object. The number of pairs is one
        less than ``q``."""
        return [fxr.pair for fxr in self.obj.fx_rates]

    @property
    def fx_rates(self) -> dict[str, DualTypes]:
        """The dict of currency pairs and their FX rates that define the object."""
        return {fxr.pair: fxr.rate for fxr in self.obj.fx_rates}

    @property
    def currencies_list(self) -> list[str]:
        """An list of currencies available in the object. Aligns with ``currencies``."""
        return [ccy.name for ccy in self.obj.currencies]

    @property
    def currencies(self) -> dict[str, int]:
        """A dict whose keys are the currencies contained in the object and the value is the
        ordered index of that currencies in other attributes such as ``fx_array`` and
        ``currencies_list``."""
        return self._currencies

    @property
    def q(self) -> int:
        """The number of currencies contained in the object."""
        return len(self.obj.currencies)

    @property
    def fx_vector(self) -> Arr1dObj:
        """A vector of currency FX rates all relative to the stated ``base`` currency."""
        return self.fx_array[0, :]  # type: ignore[return-value]

    @property
    def pairs_settlement(self) -> dict[str, datetime]:
        """A dict aggregating each FX pair and its settlement date. In an *FXRates* object
        all pairs settle on the same settlement date."""
        return dict.fromkeys(self.pairs, self.settlement)

    @property
    def variables(self) -> tuple[str, ...]:
        """The names of the variables associated with the object for automatic differentiation (AD)
        purposes."""
        return tuple(f"fx_{pair}" for pair in self.pairs)

    @property
    def _ad(self) -> int:
        return self.obj.ad

    def rate(self, pair: str) -> Number:
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
        return self._fx_array_el(domi, fori)

    def restate(self, pairs: list[str], keep_ad: bool = False) -> FXRates:
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
        if pairs == self.pairs and keep_ad:
            return self.__copy__()  # no restate needed but return new instance

        restated_fx_rates = FXRates(
            {pair: self.rate(pair) if keep_ad else self.rate(pair).real for pair in pairs},
            settlement=self.settlement,
            base=self.base,
        )
        return restated_fx_rates

    def convert(
        self,
        value: DualTypes,
        domestic: str,
        foreign: str | NoInput = NoInput(0),
        on_error: str = "ignore",
    ) -> DualTypes | None:
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
        foreign = self.base if isinstance(foreign, NoInput) else foreign.lower()
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
        return value * self._fx_array_el(i, j)

    def convert_positions(
        self,
        array: Arr1dF64 | list[float],
        base: str | NoInput = NoInput(0),
    ) -> Number:
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
        base = self.base if isinstance(base, NoInput) else base.lower()
        array_ = np.asarray(array)
        j = self.currencies[base]
        return np.sum(array_ * self.fx_array[:, j])  # type: ignore[no-any-return]

    def positions(
        self,
        value: DualTypes,
        base: str | NoInput = NoInput(0),
    ) -> Series[float]:
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
        if isinstance(value, float | int):
            value = Dual(value, [], [])
        base_: str = self.base if isinstance(base, NoInput) else base.lower()
        _ = np.array([0 if ccy != base_ else value.real for ccy in self.currencies_list])
        for pair in value.vars:
            if pair[:3] == "fx_":
                delta = gradient(value, [pair])[0]
                _ += self._get_positions_from_delta(delta, pair[3:], base_)
        return Series(_, index=self.currencies_list)

    def _get_positions_from_delta(
        self, delta: float, pair: str, base: str
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return an array of cash positions determined from an FX pair delta risk."""
        b_idx = self.currencies[base]
        domestic, foreign = pair[:3], pair[3:]
        d_idx, f_idx = self.currencies[domestic], self.currencies[foreign]
        _: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(self.q, dtype=np.float64)

        # f_val = -delta * float(self.fx_array[b_idx, d_idx]) * float(self.fx_array[d_idx,f_idx])**2
        # _[f_idx] = f_val
        # _[d_idx] = -f_val / float(self.fx_array[d_idx, f_idx])
        # return _
        f_val = delta * float(self._fx_array_el(b_idx, f_idx))
        _[d_idx] = f_val
        _[f_idx] = -f_val / float(self._fx_array_el(f_idx, d_idx))
        return _  # calculation is more efficient from a domestic pov than foreign

    def rates_table(self) -> DataFrame:
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

    # Cache management

    def _clear_cache(self) -> None:
        """
        Clear the cache ID so the fx_array can be fetched and cached from Rust object.
        """
        # the fx_array is a cached property.
        self.__dict__.pop("fx_array", None)

    # Mutation

    @_new_state_post
    @_clear_cache_post
    def update(self, fx_rates: dict[str, float] | NoInput = NoInput(0)) -> None:
        """
        Update all or some of the FX rates of the instance with new market data.

        Parameters
        ----------
        fx_rates : dict, optional
            Dict whose keys are 6-character domestic-foreign currency pairs and
            which are present in FXRates.pairs, and whose
            values are the relevant rates to update. An empty dict will be ignored and
            perform no update.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of an *FXRates* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *FXRates* instance.
           This class is labelled as a **mutable on update** object.

        Suppose an *FXRates* class has been instantiated and resides in memory.

        .. ipython:: python

           fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.25}, settlement=dt(2022, 1, 3), base="usd")
           id(fxr)

        This object may be linked to others, probably an :class:`~rateslib.fx.FXForwards` class.
        It can be updated with some new market data. This will preserve its memory id and
        association with other objects. Any :class:`~rateslib.fx.FXForwards` objects referencing
        this will detect this change and will also lazily update via *rateslib's* state
        management.

        .. ipython:: python

           linked_obj = fxr
           fxr.update({"eurusd": 1.06})
           id(fxr)  # <- SAME as above
           linked_obj.rate("eurusd")

        Examples
        --------

        .. ipython:: python

           fxr = FXRates({"usdeur": 0.9, "eurnok": 8.5})
           fxr.rate("usdnok")
           fxr.update({"usdeur": 1.0})
           fxr.rate("usdnok")
        """
        if isinstance(fx_rates, NoInput) or len(fx_rates) == 0:
            return None
        fx_rates_ = [FXRate(k[0:3], k[3:6], v, self.settlement) for k, v in fx_rates.items()]
        self.obj.update(fx_rates_)

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        """
        Change the node values to float, Dual or Dual2 based on input parameter.
        """
        self.obj.set_ad_order(_get_adorder(order))

    # Serialization

    def to_json(self) -> str:
        """Return a JSON representation of the object.

        Returns
        -------
        str
        """
        return _make_py_json(self.obj.to_json(), "FXRates")


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
