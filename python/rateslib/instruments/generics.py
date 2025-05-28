from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import DataFrame, DatetimeIndex, concat

from rateslib import defaults
from rateslib.calendars import dcf
from rateslib.curves import Curve
from rateslib.curves._parsers import _validate_curve_is_not_dict, _validate_curve_not_no_input
from rateslib.curves.utils import _CurveType
from rateslib.default import NoInput, _drb
from rateslib.dual import dual_log
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface
from rateslib.instruments.base import Metrics
from rateslib.instruments.sensitivities import Sensitivities
from rateslib.instruments.utils import (
    _composit_fixings_table,
    _get_curves_fx_and_base_maybe_from_solver,
    _get_fxvol_maybe_from_solver,
)
from rateslib.solver import Solver

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        Curves_,
        DualTypes,
        FXVol_,
        NoReturn,
        Solver_,
        SupportsMetrics,
        datetime_,
        str_,
    )

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


class Value(Metrics):
    """
    A null *Instrument* which can be used within a :class:`~rateslib.solver.Solver`
    to directly parametrise a *Curve* node, via some calculated value.

    Parameters
    ----------
    effective : datetime
        The datetime index for which the `rate`, which is just the curve value, is
        returned.
    curves : Curve, LineCurve, str or list of such, optional
        A single :class:`~rateslib.curves.Curve`,
        :class:`~rateslib.curves.LineCurve` or id or a
        list of such. Only uses the first *Curve* in a list.
    convention : str, optional,
        Day count convention used with certain ``metric``.
    metric : str in {"curve_value", "index_value", "cc_zero_rate"}, optional
        Configures which value to extract from the *Curve*.

    Examples
    --------
    The below :class:`~rateslib.curves.Curve` is solved directly
    from a calibrating DF value on 1st Nov 2022.

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="v")
       instruments = [(Value(dt(2022, 11, 1)), (curve,), {})]
       solver = Solver([curve], [], instruments, [0.99])
       curve[dt(2022, 1, 1)]
       curve[dt(2022, 11, 1)]
       curve[dt(2023, 1, 1)]
    """

    _rate_scalars = {
        "curve_value": 100.0,
        "index_value": 100.0,
        "cc_zero_rate": 1.0,
    }

    def __init__(
        self,
        effective: datetime,
        convention: str_ = NoInput(0),
        metric: str = "curve_value",
        curves: Curves_ = NoInput(0),
    ) -> None:
        self.effective = effective
        self.curves = curves
        self.convention = _drb(defaults.convention, convention)
        self.metric = metric.lower()
        self._rate_scalar = self._rate_scalars.get(self.metric, 1.0)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return a value derived from a *Curve*.

        Parameters
        ----------
        curves : Curve, LineCurve, str or list of such
            Uses only one *Curve*, the one given or the first in the list.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            Not used.
        base : str, optional
            Not used.
        metric: str in {"curve_value", "index_value", "cc_zero_rate"}, optional
            Configures which type of value to return from the applicable *Curve*.

        Returns
        -------
        float, Dual, Dual2

        Notes
        ------
        The ``metric`` *"index_value"* will use a *"daily*" style interpolation to derive the
        value for the ``effective`` date. If the objective is to set a monthly inflation value
        during a curve calibration then the suggestion would be to define the ``effective`` date
        of the :class:`~rateslib.instruments.Value` as the 1st of a given month, in which case
        both interpolation methods give the same result.

        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            "_",
        )
        metric = _drb(self.metric, metric).lower()
        curve_0: Curve = _validate_curve_not_no_input(_validate_curve_is_not_dict(curves_[0]))
        if metric == "curve_value":
            return curve_0[self.effective]
        elif metric == "cc_zero_rate":
            if curve_0._base_type != _CurveType.dfs:
                raise TypeError(
                    "`curve` used with `metric`='cc_zero_rate' must be discount factor based.",
                )
            dcf_ = dcf(curve_0.nodes.initial, self.effective, self.convention)
            ret: DualTypes = (dual_log(curve_0[self.effective]) / -dcf_) * 100
            return ret
        elif metric == "index_value":
            ret = curve_0.index_value(self.effective, curve_0.meta.index_lag, "daily")
            return ret
        raise ValueError("`metric`must be in {'curve_value', 'cc_zero_rate', 'index_value'}.")

    def npv(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`Value` instrument has no concept of NPV.")

    def cashflows(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`Value` instrument has no concept of cashflows.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`Value` instrument has no concept of analytic delta.")


class VolValue(Metrics):
    """
    A null *Instrument* which can be used within a :class:`~rateslib.solver.Solver`
    to directly parametrise a *Vol* node, via some calculated metric.

    Parameters
    ----------
    index_value : float, Dual, Dual2
        The value of some index to the *VolSmile* or *VolSurface*.
    expiry: datetime, optional
        The expiry at which to evaluate. This will only be used with *Surfaces*, not *Smiles*.
    metric: str, optional
        The default metric to return from the ``rate`` method.
    vol: str, FXDeltaVolSmile, optional
        The associated object from which to determine the ``rate``.

    Examples
    --------
    The below :class:`~rateslib.fx_volatility.FXDeltaVolSmile` is solved directly
    from calibrating volatility values.

    .. ipython:: python
       :suppress:

       from rateslib.fx_volatility import FXDeltaVolSmile
       from rateslib.instruments import VolValue
       from rateslib.solver import Solver

    .. ipython:: python

       smile = FXDeltaVolSmile(
           nodes={0.25: 10.0, 0.5: 10.0, 0.75: 10.0},
           eval_date=dt(2023, 3, 16),
           expiry=dt(2023, 6, 16),
           delta_type="forward",
           id="VolSmile",
       )
       instruments = [
           VolValue(0.25, vol="VolSmile"),
           VolValue(0.5, vol="VolSmile"),
           VolValue(0.75, vol=smile)
       ]
       solver = Solver(curves=[smile], instruments=instruments, s=[8.9, 7.8, 9.9])
       smile[0.25]
       smile[0.5]
       smile[0.75]
    """

    def __init__(
        self,
        index_value: DualTypes,
        expiry: datetime_ = NoInput(0),
        # index_type: str = "delta",
        # delta_type: str = NoInput(0),
        metric: str = "vol",
        vol: FXVol_ = NoInput(0),
    ):
        self.index_value = index_value
        self.expiry = expiry
        # self.index_type = index_type
        # self.delta_type = delta_type
        self.vol = vol
        self.curves = NoInput(0)
        self.metric = metric.lower()

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
        metric: str = "vol",
    ) -> DualTypes:
        """
        Return a value derived from a *Curve*.

        Parameters
        ----------
        curves : Curve, LineCurve, str or list of such
            Uses only one *Curve*, the one given or the first in the list.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            Not used.
        base : str, optional
            Not used.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface
            The volatility used in calculation.
        metric: str in {"curve_value", "index_value", "cc_zero_rate"}, optional
            Configures which type of value to return from the applicable *Curve*.

        Returns
        -------
        float, Dual, Dual2

        """
        vol_ = _get_fxvol_maybe_from_solver(self.vol, vol, solver)
        metric = _drb(self.metric, metric).lower()

        if metric == "vol":
            if isinstance(vol_, FXDeltaVolSmile | FXDeltaVolSurface):
                # Must initialise with an ``expiry`` if a Surface is used
                return vol_._get_index(self.index_value, self.expiry)  # type: ignore[arg-type]
            else:
                raise ValueError("`vol` as an object must be provided for VolValue.")

        raise ValueError("`metric` must be in {'vol'}.")

    def npv(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of NPV.")

    def cashflows(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of cashflows.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of analytic delta.")


class Spread(Sensitivities):
    """
    A spread instrument defined as the difference in rate between two *Instruments*.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s.
    instrument2 : Instrument
        The second instrument, usually the longest tenor, e.g. 10Y in 5s10s.

    Notes
    -----
    When using a :class:`Spread` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    Creating a dynamic :class:`Spread` where the *Instruments* are dynamically priced,
    and each share the pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q")
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q")
       spread = Spread(irs1, irs2)
       spread.npv(curve1)
       spread.rate(curve1)
       spread.cashflows(curve1)

    Creating an assigned :class:`Spread`, where each *Instrument* has its own
    assigned pricing arguments.

    .. ipython:: python

       curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.995, dt(2022, 7, 1):0.985})
       curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 4, 1):0.99, dt(2022, 7, 1):0.98})
       irs1 = IRS(dt(2022, 1, 1), "3M", "Q", curves=curve1)
       irs2 = IRS(dt(2022, 1, 1), "6M", "Q", curves=curve2)
       spread = Spread(irs1, irs2)
       spread.npv()
       spread.rate()
       spread.cashflows()
    """

    _rate_scalar = 100.0

    def __init__(self, instrument1: SupportsMetrics, instrument2: SupportsMetrics) -> None:
        self.instrument1 = instrument1
        self.instrument2 = instrument2

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        """
        Return the NPV of the composited object by summing instrument NPVs.

        Parameters
        ----------
        args :
            Positional arguments required for the ``npv`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``npv`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----

        If the argument ``local`` is added to return a dict of currencies, ensure
        that this is added as a **keyword** argument and not a positional argument.
        I.e. use `local=True`.
        """
        leg1_npv = self.instrument1.npv(*args, **kwargs)
        leg2_npv = self.instrument2.npv(*args, **kwargs)
        if kwargs.get("local", False):
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0)  # type: ignore[union-attr]
                for k in set(leg1_npv) | set(leg2_npv)  # type: ignore[arg-type]
            }
        else:
            return leg1_npv + leg2_npv  # type: ignore[operator]

    # def npv(self, *args, **kwargs):
    #     if len(args) == 0:
    #         args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
    #         args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
    #     else:
    #         args1 = args
    #         args2 = args
    #     return self.instrument1.npv(*args1) + self.instrument2.npv(*args2)

    def rate(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the mid-market rate of the composited via the difference of instrument
        rates.

        Parameters
        ----------
        args :
            Positional arguments required for the ``rate`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``rate`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_rate = self.instrument1.rate(*args, **kwargs)
        leg2_rate = self.instrument2.rate(*args, **kwargs)
        return (leg2_rate - leg1_rate) * 100.0

    # def rate(self, *args, **kwargs):
    #     if len(args) == 0:
    #         args1 = (kwargs.get("curve1", None), kwargs.get("disc_curve1", None))
    #         args2 = (kwargs.get("curve2", None), kwargs.get("disc_curve2", None))
    #     else:
    #         args1 = args
    #         args2 = args
    #     return self.instrument2.rate(*args2) - self.instrument1.rate(*args1)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        return concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2"],
        )

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the *Instruments*.

        For arguments see :meth:`XCS.fixings_table()<rateslib.instruments.XCS.fixings_table>`,
        and/or :meth:`IRS.fixings_table()<rateslib.instruments.IRS.fixings_table>`

        Returns
        -------
        DataFrame
        """
        pf = Portfolio(
            [
                self.instrument1,
                self.instrument2,
            ]
        )
        return pf.fixings_table(curves, solver, fx, base, approximate, right=right)


class Fly(Sensitivities):
    """
    A butterfly instrument which is, mechanically, the spread of two spread instruments.

    Parameters
    ----------
    instrument1 : Instrument
        The initial instrument, usually the shortest tenor, e.g. 5Y in 5s10s15s.
    instrument2 : Instrument
        The second instrument, usually the mid-length tenor, e.g. 10Y in 5s10s15s.
    instrument3 : Instrument
        The third instrument, usually the longest tenor, e.g. 15Y in 5s10s15s.

    Notes
    -----
    When using a :class:`Fly` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    _rate_scalar = 100.0

    def __init__(
        self,
        instrument1: SupportsMetrics,
        instrument2: SupportsMetrics,
        instrument3: SupportsMetrics,
    ) -> None:
        self.instrument1 = instrument1
        self.instrument2 = instrument2
        self.instrument3 = instrument3

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        """
        Return the NPV of the composited object by summing instrument NPVs.

        Parameters
        ----------
        args :
            Positional arguments required for the ``npv`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``npv`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_npv = self.instrument1.npv(*args, **kwargs)
        leg2_npv = self.instrument2.npv(*args, **kwargs)
        leg3_npv = self.instrument3.npv(*args, **kwargs)
        if kwargs.get("local", False):
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) + leg3_npv.get(k, 0)  # type: ignore[union-attr]
                for k in set(leg1_npv) | set(leg2_npv) | set(leg3_npv)  # type: ignore[arg-type]
            }
        else:
            return leg1_npv + leg2_npv + leg3_npv  # type: ignore[operator]

    def rate(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the mid-market rate of the composited via the difference of instrument
        rates.

        Parameters
        ----------
        args :
            Positional arguments required for the ``rate`` method of both of the
            underlying ``Instruments``.
        kwargs :
            Keyword arguments required for the ``rate`` method of both of the underlying
            ``Instruments``.

        Returns
        -------
        float, Dual or Dual2
        """
        leg1_rate = self.instrument1.rate(*args, **kwargs)
        leg2_rate = self.instrument2.rate(*args, **kwargs)
        leg3_rate = self.instrument3.rate(*args, **kwargs)
        return (-leg3_rate + 2 * leg2_rate - leg1_rate) * 100.0

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        _: DataFrame = concat(
            [
                self.instrument1.cashflows(*args, **kwargs),
                self.instrument2.cashflows(*args, **kwargs),
                self.instrument3.cashflows(*args, **kwargs),
            ],
            keys=["instrument1", "instrument2", "instrument3"],
        )
        return _

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the *Instruments*.

        For arguments see :meth:`XCS.fixings_table()<rateslib.instruments.XCS.fixings_table>`,
        and/or :meth:`IRS.fixings_table()<rateslib.instruments.IRS.fixings_table>`

        Returns
        -------
        DataFrame
        """
        pf = Portfolio(
            [
                self.instrument1,
                self.instrument2,
                self.instrument3,
            ]
        )
        return pf.fixings_table(curves, solver, fx, base, approximate, right=right)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


def _instrument_npv(
    instrument: SupportsMetrics, *args: Any, **kwargs: Any
) -> NPV:  # pragma: no cover
    # this function is captured by TestPortfolio pooling but is not registered as a parallel process
    # used for parallel processing with Portfolio.npv
    return instrument.npv(*args, **kwargs)


class Portfolio(Sensitivities, Metrics):
    """
    Create a collection of *Instruments* to group metrics

    Parameters
    ----------
    instruments : list
        This should be a list of *Instruments*.

    Notes
    -----
    When using a :class:`Portfolio` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
    """

    def __init__(self, instruments: Sequence[SupportsMetrics]) -> None:
        if not isinstance(instruments, Sequence):
            raise ValueError("`instruments` should be a list of Instruments.")
        self.instruments = instruments

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        **kwargs: Any,
    ) -> NPV:
        """
        Return the NPV of the *Portfolio* by summing instrument NPVs.

        For arguments see :meth:`BaseDerivative.npv()<rateslib.instruments.BaseDerivative.npv>`.
        """
        # TODO look at legs.npv where args len is used.
        if not local and isinstance(base, NoInput) and isinstance(fx, NoInput):
            warnings.warn(
                "No ``base`` currency is inferred, using ``local`` output. To return a single "
                "PV specify a ``base`` currency and ensure an ``fx`` or ``solver.fx`` object "
                "is available to perform the conversion if the currency differs from the local.",
                UserWarning,
            )
            local = True

        # if the pool is 1 do not do any parallel processing and return the single core func
        if defaults.pool == 1:
            return self._npv_single_core(
                curves=curves,
                solver=solver,
                fx=fx,
                base=base,
                local=local,
                **kwargs,
            )

        from functools import partial
        from multiprocessing import Pool

        func = partial(
            _instrument_npv,
            curves=curves,
            solver=solver,
            fx=fx,
            base=base,
            local=local,
            **kwargs,
        )
        p = Pool(defaults.pool)
        results = p.map(func, self.instruments)
        p.close()

        if local:
            _ = DataFrame(results).fillna(0.0)
            _ = _.sum()
            val1: dict[str, DualTypes] = _.to_dict()

            # ret = {}
            # for result in results:
            #     for ccy in result:
            #         if ccy in ret:
            #             ret[ccy] += result[ccy]
            #         else:
            #             ret[ccy] = result[ccy]

            ret: NPV = val1
        else:
            val2: DualTypes = sum(results)  # type: ignore[arg-type]
            ret = val2

        return ret

    def _npv_single_core(self, *args: Any, **kwargs: Any) -> NPV:
        if kwargs.get("local", False):
            # dicts = [instrument.npv(*args, **kwargs) for instrument in self.instruments]
            # result = dict(reduce(operator.add, map(Counter, dicts)))

            val1: dict[str, DualTypes] = {}
            for instrument in self.instruments:
                i_npv: dict[str, DualTypes] = instrument.npv(*args, **kwargs)  # type: ignore[assignment]
                for ccy in i_npv:
                    if ccy in val1:
                        val1[ccy] += i_npv[ccy]
                    else:
                        val1[ccy] = i_npv[ccy]
            ret: DualTypes | dict[str, DualTypes] = val1
        else:
            val2: DualTypes = sum(
                instrument.npv(*args, **kwargs)  # type: ignore[misc]
                for instrument in self.instruments
            )
            ret = val2
        return ret

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        _: DataFrame = concat(
            [_.cashflows(*args, **kwargs) for _ in self.instruments],
            keys=[f"inst{i}" for i in range(len(self.instruments))],
        )
        return _

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def exo_delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument* measured
        against user defined :class:`~rateslib.dual.Variable`.

        For arguments see
        :meth:`Sensitivities.exo_delta()<rateslib.instruments.Sensitivities.exo_delta>`.
        """
        return super().exo_delta(*args, **kwargs)

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the *Instruments*.

        For arguments see :meth:`XCS.fixings_table()<rateslib.instruments.XCS.fixings_table>`,
        and/or :meth:`IRS.fixings_table()<rateslib.instruments.IRS.fixings_table>`

        Returns
        -------
        DataFrame
        """
        df_result = DataFrame(
            index=DatetimeIndex([], name="obs_dates"),
        )
        for inst in self.instruments:
            try:
                df = inst.fixings_table(  # type: ignore[attr-defined]
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    base=base,
                    approximate=approximate,
                    right=right,
                )
            except AttributeError:
                continue
            df_result = _composit_fixings_table(df_result, df)
        return df_result

    def rate(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`rate` is not defined for Portfolio.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
