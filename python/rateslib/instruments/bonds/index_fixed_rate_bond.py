# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.dual import Dual, gradient
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.conventions import (
    BondCalcMode,
    _get_bond_calc_mode,
)
from rateslib.instruments.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg
from rateslib.periods.parameters import _IndexParams

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        CurvesT_,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
        IndexMethod,
        LegFixings,
        Number,
        RollDay,
        Sequence,
        Solver_,
        VolT_,
        _BaseCurve_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        float_,
        int_,
        str_,
    )


class IndexFixedRateBond(_BaseBondInstrument):
    """
    An *index-linked fixed rate bond* composed of a :class:`~rateslib.legs.FixedLeg`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import IndexFixedRateBond
       from datetime import datetime as dt
       from rateslib import fixings

    .. ipython:: python

       fixings.add("RPI_series", Series(index=[dt(2024, 4, 1), dt(2024, 5, 1)], data=[385.0, 386.4]))
       ifrb = IndexFixedRateBond(
           effective=dt(2024, 7, 12),
           termination="2y",
           fixed_rate=2.25,
           spec="us_gbi",
           index_fixings="RPI_series",
       )
       ifrb.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("RPI_series")

    .. rubric:: Pricing

    An *IndexFixedRateBond* requires an *index curve* and a *disc curve*. The following input
    formats are allowed:

    .. code-block:: python

       curves = [index_curve, disc_curve]   # two curves as a list
       curves = {"index_curve": index_curve, "disc_curve": disc_curve}  # dict form is explicit

    The available ``metric`` for the :meth:`~rateslib.instruments.IndexFixedRateBond.rate`
    are in *{'clean_price', 'dirty_price', 'ytm', 'indexed_ytm', 'indexed_clean_price',
    'indexed_dirty_price'}*.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define generalised **scheduling** parameters.

    effective : datetime, :red:`required`
        The unadjusted effective date. If given as adjusted, unadjusted alternatives may be
        inferred.
    termination : datetime, str, :red:`required`
        The unadjusted termination date. If given as adjusted, unadjusted alternatives may be
        inferred. If given as string tenor will be calculated from ``effective``.
    frequency : Frequency, str, :red:`required`
        The frequency of the schedule.
        If given as string will derive a :class:`~rateslib.scheduling.Frequency` aligning with:
        monthly ("M"), quarterly ("Q"), semi-annually ("S"), annually("A") or zero-coupon ("Z"), or
        a set number of calendar or business days ("_D", "_B"), weeks ("_W"), months ("_M") or
        years ("_Y").
        Where required, the :class:`~rateslib.scheduling.RollDay` is derived as per ``roll``
        and business day calendar as per ``calendar``.
    stub : StubInference, str in {"ShortFront", "LongFront", "ShortBack", "LongBack"}, :green:`optional`
        The stub type used if stub inference is required. If given as string will derive a
        :class:`~rateslib.scheduling.StubInference`.
    front_stub : datetime, :green:`optional`
        The unadjusted date for the start stub period. If given as adjusted, unadjusted
        alternatives may be inferred.
    back_stub : datetime, :green:`optional`
        The unadjusted date for the back stub period. If given as adjusted, unadjusted
        alternatives may be inferred.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    roll : RollDay, int in [1, 31], str in {"eom", "imm", "som"}, :green:`optional`
        The roll day of the schedule. If not given or not available in ``frequency`` will be
        inferred for monthly frequency variants.
    eom : bool, :green:`optional`
        Use an end of month preference rather than regular rolls for ``roll`` inference. Set by
        default. Not required if ``roll`` is defined.
    modifier : Adjuster, str in {"NONE", "F", "MF", "P", "MP"}, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` used for adjusting unadjusted schedule dates
        into adjusted dates. If given as string must define simple date rolling rules.
    calendar : calendar, str, :green:`optional`
        The business day calendar object to use. If string will call
        :meth:`~rateslib.scheduling.get_calendar`.
    payment_lag: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        a payment date. If given as integer will define the number of business days to
        lag payments by.
    payment_lag_exchange: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        additional payment date. If given as integer will define the number of business days to
        lag payments by.
    ex_div: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        additional dates, which may be used, for example by fixings schedules. If given as integer
        will define the number of business days to lag dates by.
    convention: str, :green:`optional (set by 'defaults')`
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.scheduling.dcf`.

        .. note::

           The following define generalised **settlement** parameters.

    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the *Instrument* (3-digit code).
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The initial leg notional, defined in units of *reference currency*.

        .. note::

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.

        .. note::

           The following parameters define **indexation**.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value applied as the base index value for all *Periods*.
        If not given and ``index_fixings`` is a string fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, 2-tuple or list, :green:`optional`
        The index value for the reference date.
        Best practice is to supply this value as string identifier relating to the global
        ``fixings`` object.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    calc_mode : str or BondCalcMode
        A calculation mode for dealing with bonds under different conventions. See notes.
    settle: int
        The number of days by which to lag 'today' to arrive at standard settlement.
    metric : str, :green:`optional` (set as 'clean_price')
        The pricing metric returned by :meth:`~rateslib.instruments.IndexFixedRateBond.rate`.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    """  # noqa: E501

    _rate_scalar = 1.0

    @property
    def fixed_rate(self) -> DualTypes_:
        """The fixed rate parameter of the composited
        :class:`~rateslib.legs.FixedLeg`."""
        return self.leg1.fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self.kwargs.leg1["fixed_rate"] = value
        self.leg1.fixed_rate = value

    @property
    def leg1(self) -> FixedLeg:
        """The :class:`~rateslib.legs.FixedLeg` of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def __init__(
        self,
        # scheduling
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: Frequency | str_ = NoInput(0),
        *,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: int | RollDay | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        convention: str_ = NoInput(0),
        # settlement parameters
        currency: str_ = NoInput(0),
        notional: float_ = NoInput(0),
        # amortization: float_ = NoInput(0),
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: LegFixings = NoInput(0),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        # meta parameters
        curves: CurvesT_ = NoInput(0),
        calc_mode: BondCalcMode | str_ = NoInput(0),
        settle: int_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "clean_price",
    ) -> None:
        user_args = dict(
            # scheduling
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            payment_lag_exchange=payment_lag_exchange,
            ex_div=ex_div,
            convention=convention,
            # settlement
            currency=currency,
            notional=notional,
            # amortization=amortization,
            # index_params
            index_base=index_base,
            index_lag=index_lag,
            index_method=index_method,
            index_fixings=index_fixings,
            # rate
            fixed_rate=fixed_rate,
            # meta
            curves=self._parse_curves(curves),
            calc_mode=calc_mode,
            settle=settle,
            metric=metric,
        )
        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            initial_exchange=False,
            final_exchange=True,
            vol=_Vol(),
        )

        default_args = dict(
            notional=defaults.notional,
            calc_mode=defaults.calc_mode[type(self).__name__],
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            payment_lag_exchange=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
            index_lag=defaults.index_lag,
            index_method=defaults.index_method,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=["curves", "calc_mode", "settle", "metric", "vol"],
        )
        self.kwargs.meta["calc_mode"] = _get_bond_calc_mode(self.kwargs.meta["calc_mode"])

        if isinstance(self.kwargs.leg1["fixed_rate"], NoInput):
            raise ValueError(f"`fixed_rate` must be provided for {type(self).__name__}.")

        self._leg1 = FixedLeg(**_convert_to_schedule_kwargs(self.kwargs.leg1, 1))
        self._legs = [self.leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An IFRB has two curve requirements: an index_curve and a disc_curve.

        No available index curve can be input as None or NoInput
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 2:
                return _Curves(
                    index_curve=curves[0] if curves[0] is not None else NoInput(0),
                    disc_curve=curves[1],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:
            raise ValueError(f"{type(self).__name__} requires 2 curve types. Got 1.")

    def index_ratio(self, settlement: datetime, index_curve: _BaseCurve_ = NoInput(0)) -> DualTypes:
        """
        Return the index ratio assigned to an *IndexFixedRateBond* for a given settlement.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from pandas import Series
           from datetime import datetime as dt
           from rateslib import fixings
           from rateslib.instruments import IndexFixedRateBond

        .. ipython:: python

           fixings.add("UK_RPI", Series(index=[dt(2025, 3, 1), dt(2025, 4, 1), dt(2025, 5, 1)], data=[395.3, 402.2, 402.9]))
           ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
               effective=dt(2025, 6, 11),
               termination=dt(2038, 9, 22),
               fixed_rate=1.75,
               spec="uk_gbi",
               index_fixings="UK_RPI"
           )
           ukti.index_ratio(settlement=dt(2025, 7, 29))

        .. ipython:: python
           :suppress:

           fixings.pop("UK_RPI")

        Parameters
        ----------
        settlement: datetime
            The settlement date of the bond.
        index_curve: _BaseCurve, optional
            A curve capable of forecasting index values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """  # noqa: E501

        left_index = self.leg1._period_index(settlement)
        period_index_params: _IndexParams = self.leg1._regular_periods[left_index].index_params  # type: ignore[assignment]

        new_index_params = _IndexParams(
            _index_method=period_index_params.index_method,
            _index_lag=period_index_params.index_lag,
            _index_base=period_index_params.index_base.value,
            _index_base_date=period_index_params.index_base.date,
            _index_reference_date=settlement,
            _index_fixings=period_index_params.index_fixing.identifier,
            _index_only=False,
        )
        return new_index_params.index_ratio(index_curve=index_curve)[0]

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate some pricing rate metric for the *Instrument*.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from pandas import Series
           from datetime import datetime as dt
           from rateslib import fixings, Curve
           from rateslib.instruments import IndexFixedRateBond

        .. ipython:: python

           disc_curve = Curve(
               nodes={dt(2025, 7, 28): 1.0, dt(2045, 7, 25): 1.0},
               convention="act365f"
           ).shift(250)  # curve begins at 0% and gets shifted by 250 Act365F O/N basis points
           index_curve = Curve(
               nodes={dt(2025, 5, 1): 1.0, dt(2045, 5, 1): 1.0},
               convention="act365f", index_lag=0, index_base=402.9
           ).shift(100)  # curves begins at 0% and gets shifted by 100 Ac6t365f O/N basis points
           fixings.add(
               "UK_RPI",
               Series(index=[dt(2025, 3, 1), dt(2025, 4, 1), dt(2025, 5, 1)], data=[395.3, 402.2, 402.9]),
           )
           ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
               effective=dt(2025, 6, 11),
               termination=dt(2038, 9, 22),
               fixed_rate=1.75,
               spec="uk_gbi",
               index_fixings="UK_RPI"
           )
           ukti.rate(curves=[index_curve, disc_curve], metric="clean_price")  # settles T+1 i.e. 29th July
           ukti.rate(curves=[index_curve, disc_curve], metric="dirty_price")
           ukti.rate(curves=[index_curve, disc_curve], metric="indexed_clean_price")
           ukti.rate(curves=[index_curve, disc_curve], metric="indexed_dirty_price")
           ukti.rate(curves=[index_curve, disc_curve], metric="ytm")
           ukti.rate(curves=[index_curve, disc_curve], metric="indexed_ytm")

        .. ipython:: python
           :suppress:

           fixings.pop("UK_RPI")

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        local: bool, :green:`optional (set as False)`
            An override flag to return a dict of NPV values indexed by string currency.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.
        metric: str, :green:`optional`
            The specific calculation to perform and the value to return.
            See **Pricing** on each *Instrument* for details of allowed inputs.

        Returns
        -------
        float, Dual, Dual2, Variable
        """  # noqa: E501
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()
        _curves = self._parse_curves(curves)
        disc_curve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=_curves,
                name="disc_curve",
                solver=solver,
            ),
            "disc_curve",
        )
        index_curve = _maybe_get_curve_maybe_from_solver(
            curves_meta=self.kwargs.meta["curves"],
            curves=_curves,
            name="index_curve",
            solver=solver,
        )

        if isinstance(settlement, NoInput):
            settlement_ = self.leg1.schedule.calendar.lag_bus_days(
                disc_curve.nodes.initial,
                self.kwargs.meta["settle"],
                True,
            )
        else:
            settlement_ = settlement
        npv = self.leg1.local_npv(
            index_curve=index_curve,
            disc_curve=disc_curve,
            settlement=settlement_,
            forward=settlement_,
        )
        # scale price to par 100 (npv is already projected forward to settlement)
        index_dirty_price = npv * 100 / -self.leg1.settlement_params.notional
        index_ratio = self.index_ratio(settlement_, index_curve)
        dirty_price = index_dirty_price / index_ratio

        if metric_ == "dirty_price":
            return dirty_price
        elif metric_ == "clean_price":
            return dirty_price - self.accrued(settlement_)
        elif metric_ == "ytm":
            return self.ytm(dirty_price, settlement_, True)
        elif metric_ == "index_dirty_price" or metric_ == "indexed_dirty_price":
            return index_dirty_price
        elif metric_ == "index_clean_price" or metric_ == "indexed_clean_price":
            return index_dirty_price - self.accrued(settlement_) * index_ratio
        elif metric_ == "index_ytm" or metric_ == "indexed_ytm":
            return self.ytm(
                price=index_dirty_price,
                settlement=settlement_,
                dirty=True,
                indexed_price=True,
                indexed_ytm=True,
                index_curve=index_curve,
            )
        else:
            raise ValueError(
                "`metric` must be in {'dirty_price', 'clean_price', 'ytm', "
                "'indexed_dirty_price', 'indexed_clean_price', 'indexed_ytm'}.",
            )

    def accrued(
        self, settlement: datetime, indexed: bool = False, index_curve: _BaseCurve_ = NoInput(0)
    ) -> DualTypes:
        """
        Calculate the accrued amount per nominal par value of 100.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from pandas import Series
           from datetime import datetime as dt
           from rateslib import fixings
           from rateslib.instruments import IndexFixedRateBond

        .. ipython:: python

           fixings.add("UK_RPI", Series(index=[dt(2025, 3, 1), dt(2025, 4, 1), dt(2025, 5, 1)], data=[395.3, 402.2, 402.9]))
           ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
               effective=dt(2025, 6, 11),
               termination=dt(2038, 9, 22),
               fixed_rate=1.75,
               spec="uk_gbi",
               index_fixings="UK_RPI"
           )
           ukti.accrued(settlement=dt(2025, 7, 29))
           ukti.accrued(settlement=dt(2025, 7, 29), indexed=True)

        .. ipython:: python
           :suppress:

           fixings.pop("UK_RPI")


        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.
        indexed : bool
            Whether to calculate the accrued amount indexed up according to settlement.
        index_curve : _BaseCurve, optional
            The curve used to forecast index values if required.

        Notes
        -----
        Calculation depends upon the
        :class:`~rateslib.instruments.bonds.conventions.BondCalcMode` of the
        *Instrument*.
        """  # noqa: E501
        unindexed_accrued = super().accrued(settlement=settlement)
        if indexed:
            index_ratio = self.index_ratio(settlement=settlement, index_curve=index_curve)
            return unindexed_accrued * index_ratio
        else:
            return unindexed_accrued

    def fwd_from_repo(
        self,
        price: DualTypes,
        settlement: datetime,
        forward_settlement: datetime,
        repo_rate: DualTypes,
        convention: str_ = NoInput(0),
        dirty: bool = False,
        method: str = "proceeds",
        indexed: bool = False,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> DualTypes:
        """
        Return a forward price implied by a given repo rate.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        repo_rate : float, Dual or Dual2
            The rate which is used to calculate values.
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.
        method : str in {"proceeds", "compounded"}, optional
            The method for determining the forward price.
        indexed : bool, optional
            Whether the given price is expressed with indexation.
        index_curve : _BaseCurve, optional
            The curve for forecasting index values if required.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        match (indexed, dirty):
            # need to adjust any input to yield an indexed_dirty_price
            case (True, True):
                indexed_dirty_price = price
            case (False, True):
                indexed_dirty_price = price * self.index_ratio(
                    settlement=settlement, index_curve=index_curve
                )
            case (True, False):
                indexed_dirty_price = price + self.accrued(
                    settlement, indexed=True, index_curve=index_curve
                )
            case (False, False):
                indexed_dirty_price = (
                    price + self.accrued(settlement, indexed=False)
                ) * self.index_ratio(settlement=settlement, index_curve=index_curve)

        forward_indexed_dirty_price = super().fwd_from_repo(
            price=indexed_dirty_price,
            settlement=settlement,
            forward_settlement=forward_settlement,
            repo_rate=repo_rate,
            convention=convention,
            dirty=True,
            method=method,
        )

        match (indexed, dirty):
            # reverse adjust the forward indexed_dirty_price to suit the input arguments
            case (True, True):
                forward_price = forward_indexed_dirty_price
            case (False, True):
                forward_price = forward_indexed_dirty_price / self.index_ratio(
                    forward_settlement, index_curve=index_curve
                )
            case (True, False):
                forward_price = forward_indexed_dirty_price - self.accrued(
                    forward_settlement, indexed=True, index_curve=index_curve
                )
            case (False, False):
                forward_price = forward_indexed_dirty_price / self.index_ratio(
                    forward_settlement, index_curve=index_curve
                ) - self.accrued(forward_settlement, indexed=False)

        return forward_price

    def repo_from_fwd(
        self,
        price: DualTypes,
        settlement: datetime,
        forward_settlement: datetime,
        forward_price: DualTypes,
        convention: str_ = NoInput(0),
        dirty: bool = False,
        indexed: bool = False,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> DualTypes:
        """
        Return an implied repo rate from a forward price.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        forward_price : float, Dual or Dual2
            The forward price which implies the repo rate
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.
        indexed : bool, optional
            Whether the given price is expressed with indexation.
        index_curve : _BaseCurve, optional
            The curve for forecasting index values if required.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        match (indexed, dirty):
            # must convert input price to indexed_dirty_price equivalents
            case (True, True):
                indexed_dirty_price = price
                forward_indexed_dirty_price = forward_price
            case (False, True):
                indexed_dirty_price = price * self.index_ratio(
                    settlement=settlement, index_curve=index_curve
                )
                forward_indexed_dirty_price = forward_price * self.index_ratio(
                    settlement=forward_settlement, index_curve=index_curve
                )
            case (True, False):
                indexed_dirty_price = price + self.accrued(
                    settlement, indexed=True, index_curve=index_curve
                )
                forward_indexed_dirty_price = forward_price + self.accrued(
                    forward_settlement, indexed=True, index_curve=index_curve
                )
            case (False, False):
                indexed_dirty_price = (
                    price + self.accrued(settlement, indexed=False)
                ) * self.index_ratio(settlement=settlement, index_curve=index_curve)
                forward_indexed_dirty_price = (
                    forward_price + self.accrued(forward_settlement, indexed=False)
                ) * self.index_ratio(settlement=forward_settlement, index_curve=index_curve)

        repo = super().repo_from_fwd(
            price=indexed_dirty_price,
            settlement=settlement,
            forward_settlement=forward_settlement,
            forward_price=forward_indexed_dirty_price,
            convention=convention,
            dirty=True,
        )
        return repo

    def duration(
        self,
        ytm: DualTypes,
        settlement: datetime,
        metric: str = "risk",
        indexed_price: bool = False,
        indexed_ytm: bool = False,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> float:
        """
        Return the (negated) derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.
        metric : str
            The specific duration calculation to return. See notes.
        indexed_price: bool, :green:`optional (set as False)`
            Indicated whether the returned price should be indexed or not.
        indexed_ytm: bool, :green:`optional (set as False)`
            Indicates if the given ``ytm`` is expressed indexed or not.
        index_curve : _BaseCurve, optional
            If either the ytm or the price are indicated as indexed then an index curve may be
            required to forecast index values.

        Returns
        -------
        float

        Notes
        -----
        For an *IndexFixedRateBond* both the price and the ytm are expressible unindexed or
        indexed. The below notation :math:`P_i` and :math:`y_j` describes either of these
        varieties provided they align with the ``indexed_price`` and ``indexed_ytm`` arguments.

        The available metrics are:

        - *"risk"*: the derivative of price w.r.t. ytm, scaled to -1bp.

          .. math::

             risk = - \\frac{\\partial P_i }{\\partial y_j}

        - *"modified"*: the modified duration which is *risk* divided by dirty price.

          .. math::

             mod \\; duration = \\frac{risk}{P_i} = - \\frac{1}{P_i} \\frac{\\partial P_i }{\\partial y_j}

        - *"duration"* (or *"macaulay"*): the duration which is modified duration reverse modified.

          .. math::

             duration = mod \\; duration \\times (1 + y_j / f)

        """  # noqa: E501
        # TODO: this is not AD safe: returns only float
        ytm_: Dual = Dual(_dual_float(ytm), ["__y__§"], [])
        dirty_price: Dual = self.price(  # type: ignore[assignment]
            ytm=ytm_,
            settlement=settlement,
            dirty=True,
            indexed_price=indexed_price,
            indexed_ytm=indexed_ytm,
            index_curve=index_curve,
        )

        if metric == "risk":
            ret: float = -gradient(dirty_price, ["__y__§"])[0]
        elif metric == "modified":
            ret = -gradient(dirty_price, ["__y__§"])[0] / _dual_float(dirty_price) * 100
        elif metric == "duration" or metric == "macaulay":
            f = self.leg1.schedule.periods_per_annum
            v = _dual_float(1 + ytm_ / (100 * f))
            ret = -gradient(dirty_price, ["__y__§"])[0] / _dual_float(dirty_price) * v * 100
        else:
            raise ValueError(
                "`metric` must be one of {'risk', 'modified', 'duration'}."
            )  # pragma: no cover
        return ret

    def ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        rate_curve: CurveOption_ = NoInput(0),
        calc_mode: BondCalcMode | str_ = NoInput(0),
        indexed_price: bool = False,
        indexed_ytm: bool = False,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> Number:
        # overloaded ytm by IndexFixedRateBond
        """
        Calculate the yield-to-maturity of the security given its price.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import FixedRateBond, dt, Dual, Dual2

        .. ipython:: python

           aapl_bond = FixedRateBond(dt(2013, 5, 4), dt(2043, 5, 4), fixed_rate=3.85, spec="us_corp")
           aapl_bond.ytm(price=87.24, settlement=dt(2014, 3, 5))
           aapl_bond.ytm(price=87.24, settlement=dt(2014, 3, 5), calc_mode="us_gb_tsy")

        .. role:: red

        .. role:: green

        Parameters
        ----------
        price: float, Dual, Dual2, Variable, :red:`required`
            The price, per 100 nominal, against which to determine the yield. Can be given as
            either clean or dirty, and either unindexed or indexed.
        settlement: datetime, :red:`required`
            The settlement date on which to determine the price.
        dirty: bool, :green:`optional (set as False)`
            If `True` will assume the
            :meth:`~rateslib.instruments.FixedRateBond.accrued` is included in the price.
        rate_curve: _BaseCurve or dict of such, :green:`optional`
            Used to forecast floating rates if required.
        calc_mode: str or BondCalcMode, :green:`optional`
            An alternative calculation mode to use. The ``calc_mode`` is typically set at
            *Instrument* initialisation and is not required, but is useful as an override to
            allow comparisons, e.g. of *"us_gb"* street convention versus *"us_gb_tsy"* treasury
            convention.
        indexed_price: bool, :green:`optional (set as False)`
            Indicates whether the input price is indexed or not.
        indexed_ytm: bool, :green:`optional (set as False)`
            Indicates whether the returned ``ytm`` is expressed indexed or not.
        index_curve: _BaseCurve :green:`optional`
            If any element is ``indexed`` then a *Curve* may be required to determine
            index ratio's in order to properly index up cashflows.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        .. ipython:: python

           aapl_bond.ytm(price=Dual(87.24, ["price", "a"], [1, -0.75]), settlement=dt(2014, 3, 5))
           aapl_bond.ytm(price=Dual2(87.24, ["price", "a"], [1, -0.75], []), settlement=dt(2014, 3, 5))

        """  # noqa: E501
        match (indexed_price, indexed_ytm):
            case (False, False) | (True, True):
                # when both price and yield are expressed in the same indexation this will be
                # handled directly
                adjusted_price = price
            case (False, True):
                # if the ytm is requested indexed but the price is given unindexed then it
                # must be indexed-up for calculation
                adjusted_price = price * self.index_ratio(
                    settlement=settlement, index_curve=index_curve
                )
            case (True, False):
                # if the ytm is requested unindexed but the price is given as indexed then it must
                # be indexed down for calculation
                adjusted_price = price / self.index_ratio(
                    settlement=settlement, index_curve=index_curve
                )
            case _:  # pragma: no cover
                raise ValueError(
                    "`indexed_price` and `indexed_ytm` must each be given as a boolean."
                )

        return self._ytm(
            price=adjusted_price,
            settlement=settlement,
            dirty=dirty,
            rate_curve=rate_curve,
            calc_mode=calc_mode,
            indexed=indexed_ytm,
            index_curve=index_curve,
        )

    def price(
        self,
        ytm: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        indexed_price: bool = False,
        indexed_ytm: bool = False,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate the price of the security per nominal value of 100, given
        yield-to-maturity.

        .. role:: red

        .. role:: green

        Parameters
        ----------
        ytm : float, :red:`required`
            The yield-to-maturity against which to determine the price. If ``indexed`` this
            should be given as a nominal ytm.
        settlement : datetime, :red:`required`
            The settlement date on which to determine the price.
        dirty : bool, optional, :green:`optional (set as False)`
            If `True` will include the
            :meth:`rateslib.instruments.FixedRateBond.accrued` in the price.
        indexed_price: bool, :green:`optional (set as False)`
            Indicated whether the returned price should be indexed or not.
        indexed_ytm: bool, :green:`optional (set as False)`
            Indicates if the given ``ytm`` is expressed indexed or not.
        index_curve: _BaseCurve, :green:`optional`
            An inflation curve to forecast index ratios if required.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------

        .. ipython:: python
           :suppress:

           from pandas import Series
           from datetime import datetime as dt
           from rateslib import fixings, Curve
           from rateslib.instruments import IndexFixedRateBond

        .. ipython:: python

           index_curve = Curve(
               nodes={dt(2025, 5, 1): 1.0, dt(2045, 5, 1): 1.0},
               convention="act365f", index_lag=0, index_base=402.9
           ).shift(100)  # curves begins at 0% and gets shifted by 100 Act365f O/N basis points
           ukti = IndexFixedRateBond(  # ISIN: GB00BMY62Z61
               effective=dt(2025, 6, 11),
               termination=dt(2038, 9, 22),
               fixed_rate=1.75,
               spec="uk_gbi",
               index_base=397.6,
           )
           ukti.index_ratio(index_curve=index_curve, settlement=dt(2025, 8, 5))
           ukti.price(ytm=2.5, settlement=dt(2025, 8, 5), indexed_ytm=True, index_curve=index_curve)
           ukti.price(ytm=1.5, settlement=dt(2025, 8, 5), indexed_ytm=False)
           ukti.price(ytm=2.5, settlement=dt(2025, 8, 5), dirty=True, indexed_ytm=True, index_curve=index_curve)
           ukti.price(ytm=1.5, settlement=dt(2025, 8, 5), dirty=True, indexed_ytm=False)

        """  # noqa: E501
        _price = self._price_from_ytm(
            ytm=ytm,
            settlement=settlement,
            calc_mode=NoInput(0),  # will be set to kwargs.meta
            dirty=dirty,
            rate_curve=NoInput(0),
            indexed=indexed_ytm,
            index_curve=index_curve,
        )

        match (indexed_price, indexed_ytm):
            case (True, True) | (False, False):
                # then both price and ytm has the same indexation expression
                return _price
            case (True, False):
                # then the yield is given unindexed but the returned price must be indexed-up
                return _price * self.index_ratio(settlement, index_curve)
            case (False, True):
                # then the yield is given unindexed but the returned price requires indexing-down
                return _price / self.index_ratio(settlement, index_curve)
            case _:  # pragma: no cover
                raise ValueError(
                    "`indexed_price` and `indexed_ytm` must each be given as a boolean."
                )
