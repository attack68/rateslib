from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.conventions import (
    BondCalcMode,
    _get_bond_calc_mode,
)
from rateslib.instruments.bonds.protocols import _BaseBondInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _Vol,
)
from rateslib.legs import FixedLeg
from rateslib.periods.parameters import _IndexParams

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurveOption_,
        Curves_,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
        IndexMethod,
        RollDay,
        Series,
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
    A *interest rate swap (IRS)* composing a :class:`~rateslib.legs.FixedLeg`
    and a :class:`~rateslib.legs.FloatLeg`.

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
    amortization: float, Dual, Dual2, Variable, str, Amortization, :green:`optional (set as zero)`
        Set a non-constant notional per *Period*. If a scalar value, adjusts the ``notional`` of
        each successive period by that same value. Should have
        sign equal to that of notional if the notional is to reduce towards zero.

        .. note::

           The following are **rate parameters**.

    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.

        .. note::

           The following are **meta parameters**.

    curves : XXX
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument.
    calc_mode : str or BondCalcMode
        A calculation mode for dealing with bonds under different conventions. See notes.
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
    def legs(self) -> list[_BaseLeg]:
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
        amortization: float_ = NoInput(0),
        # index params
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: Series[DualTypes] | str_ = NoInput(0),
        # rate parameters
        fixed_rate: DualTypes_ = NoInput(0),
        # meta parameters
        curves: Curves_ = NoInput(0),
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
            amortization=amortization,
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

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """
        An IFRB has two curve requirements: an index_curve and a disc_curve.

        1 element will be assigned as the discount curve only. fixings might be published.

        When given as 2 elements the first is treated as the index curve and the 2nd as disc curve.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    disc_curve=curves[0],
                    index_curve=NoInput(0),
                )
            elif len(curves) == 2:
                return _Curves(
                    index_curve=curves[0],
                    disc_curve=curves[1],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 2 curve types. Got {len(curves)}."
                )
        else:  # `curves` is just a single input which is set as the discount curve.
            return _Curves(
                disc_curve=curves,
                index_curve=NoInput(0),
            )

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
        period_index_params = self.leg1._regular_periods[left_index].index_params

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
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes_:
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "ytm", "index_clean_price", "index_dirty_price"}
        forward_settlement : datetime, optional
            The forward settlement date. If not given uses the discount *Curve* and the ``settle``
            attribute of the bond.

        Returns
        -------
        float, Dual, Dual2
        """
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()
        if metric_ in [
            "clean_price",
            "dirty_price",
            "index_clean_price",
            "ytm",
            "index_dirty_price",
        ]:
            _curves = self._parse_curves(curves)
            disc_curve = _maybe_get_curve_or_dict_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=_curves,
                name="disc_curve",
                solver=solver,
            )
            index_curve = _maybe_get_curve_or_dict_maybe_from_solver(
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
            elif metric_ == "index_dirty_price":
                return index_dirty_price
            elif metric_ == "index_clean_price":
                return index_dirty_price - self.accrued(settlement_) * index_ratio
        else:
            raise ValueError(
                "`metric` must be in {'dirty_price', 'clean_price', 'ytm', "
                "'index_dirty_price', 'index_clean_price'}.",
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
        if not indexed:
            ir = self.index_ratio(settlement=settlement, index_curve=index_curve)
            if not dirty:
                # convert unindexed clean price
                idx_dp = (price + self.accrued(settlement, indexed=False)) * ir
            else:
                # convert unindexed dirty price
                idx_dp = price * ir
        else:
            if not dirty:
                # convert indexed clean price
                idx_dp = price + self.accrued(settlement, indexed=True, index_curve=index_curve)
            else:
                # price is indexed dirty
                idx_dp = price

        fwd_idx_dp = super().fwd_from_repo(
            price=idx_dp,
            settlement=settlement,
            forward_settlement=forward_settlement,
            repo_rate=repo_rate,
            convention=convention,
            dirty=True,
            method=method,
        )

        if not indexed:
            irf = self.index_ratio(settlement=forward_settlement, index_curve=index_curve)
            if not dirty:
                # revert back to unindexed clean price
                price = fwd_idx_dp / irf - self.accrued(forward_settlement, indexed=False)
            else:
                # revert to unindexed dirty price
                price = fwd_idx_dp / irf
        else:
            if not dirty:
                # revert to indexed clean price
                price = fwd_idx_dp - self.accrued(
                    settlement=forward_settlement, indexed=True, index_curve=index_curve
                )
            else:
                price = fwd_idx_dp

        return price

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
        if not indexed:
            ir = self.index_ratio(settlement=settlement, index_curve=index_curve)
            fir = self.index_ratio(settlement=forward_settlement, index_curve=index_curve)
            if not dirty:
                # convert unindexed clean price
                idx_dp = (price + self.accrued(settlement, indexed=False)) * ir
                fwd_idx_dp = (forward_price + self.accrued(forward_settlement, indexed=False)) * fir
            else:
                # convert unindexed dirty price
                idx_dp = price * ir
                fwd_idx_dp = forward_price * fir
        else:
            if not dirty:
                # convert indexed clean price
                idx_dp = price + self.accrued(settlement, indexed=True, index_curve=index_curve)
                fwd_idx_dp = forward_price + self.accrued(
                    forward_settlement, indexed=True, index_curve=index_curve
                )
            else:
                # price is indexed dirty
                idx_dp = price
                fwd_idx_dp = forward_price

        repo = super().repo_from_fwd(
            price=idx_dp,
            settlement=settlement,
            forward_settlement=forward_settlement,
            forward_price=fwd_idx_dp,
            convention=convention,
            dirty=True,
        )
        return repo

    def duration(
        self,
        ytm: DualTypes,
        settlement: datetime,
        metric: str = "risk",
        indexed: bool = False,
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
        indexed : bool, optional
            Whether the metric is indexed or not.

        Returns
        -------
        float

        Notes
        -----
        The available metrics are:

        - *"risk"*: the derivative of price w.r.t. ytm, scaled to -1bp.

          .. math::

             risk = - \\frac{\\partial P }{\\partial y}

        - *"modified"*: the modified duration which is *risk* divided by price.

          .. math::

             mduration = \\frac{risk}{P} = - \\frac{1}{P} \\frac{\\partial P }{\\partial y}

        - *"duration"*: the duration which is modified duration reverse modified.

          .. math::

             duration = mduration \\times (1 + y / f)

        """
        # TODO: this is not AD safe: returns only float
        value = super().duration(
            ytm=ytm,
            settlement=settlement,
            metric=metric,
        )
        if metric == "risk" and indexed:
            return value * self.index_ratio(settlement=settlement, index_curve=index_curve)
        else:
            return value
