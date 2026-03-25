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

from datetime import datetime as dt
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _convert_to_schedule_kwargs, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _Vol,
)
from rateslib.legs import CustomLeg
from rateslib.periods import Cashflow
from rateslib.scheduling import Frequency

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Adjuster,
        CalInput,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        IndexMethod,
        PeriodFixings,
        Solver_,
        VolT_,
        _BaseLeg,
        bool_,
        datetime,
        datetime_,
        float_,
        int_,
        str_,
    )


class Fee(_BaseInstrument):
    """
    A single :class:`~rateslib.periods.Cashflow` payable on a payment date.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Fee
       from datetime import datetime as dt

    .. ipython:: python

       fee = Fee(dt(2022, 1, 4), notional=2e6, calendar="nyc", payment_lag=0)
       fee.cashflows()

    .. rubric:: Pricing

    A *Fee* requires just one *Curve* for discounting, unless it is also indexed, in which
    case it may also require an additional index *Curve*

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = [index_curve, disc_curve]  #  two curves given the specific order
       curves = {  # dict form is explicit
           "disc_curve": disc_curve,
           "index_curve": index_curve,
       }

    The concept of *rate* is alien to a *Fee*, and these are not *Instruments* that would
    typically be expected to form part of a *Solver* framework. However, for flexibility,
    two *rate* ``metric`` that are available are:

    - *'npv'*: returns the result of the :meth:`~rateslib.instruments.Fee.npv` method.
    - *'payment'*: returns the physical settlement amount.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define generalised **settlement** parameters.

    effective : datetime, :red:`required`
        The datetime index for which the `rate`, which is just the curve value, is
        returned.
    notional : float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The initial leg notional, defined in units of *reference currency*.
    currency : str, :green:`optional (set by 'defaults')`
        The local settlement currency of the *Instrument* (3-digit code).
        calendar : calendar, str, :green:`optional`
        The business day calendar object to use. If string will call
        :meth:`~rateslib.scheduling.get_calendar`.
    calendar : calendar, str, :green:`optional`
        The business day calendar object to use for date manipulation. If string will call
        :meth:`~rateslib.scheduling.get_calendar`.
    payment_lag: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` used to modify the ``effective`` payment date
        according to a given ``calendar``.
    ex_div: Adjuster, int, :green:`optional`
        The :class:`~rateslib.scheduling.Adjuster` to use to map the adjusted payment date into
        an additional date acting an ex-dividend indicator. If given as integer
        will define the number of business days to lag dates by.

    .. note::

           The following define **non-deliverability** parameters. If the fee is
           directly deliverable do not use these parameters.

    pair: FXIndex, str, :green:`optional`
        The currency pair for :class:`~rateslib.data.fixings.FXFixing` that determines *Period*
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` according
        to non-deliverability.

        .. note::

           The following parameters define **indexation**. The *Period* will be considered
           indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
           are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value set of the base index value.
        If not given and ``index_fixings`` is a str fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The index value for the reference date.
        If a scalar value this is used directly. If a string identifier will link to the
        central ``fixings`` object and data loader. See :ref:`fixings <fixings-doc>`.
    index_base_date: datetime, :green:`optional`
        The reference date for determining the base index value. Not required if ``index_base``
        value is given directly, but required for indexation in all other cases.
    index_reference_date: datetime, :green:`optional (set as 'payment')`
        The reference date for determining the index value. Not required if ``_index_fixings``
        is given as a scalar value.
    index_only: bool, :green:`optional (set as False)`
        A flag which determines non-payment of notional on supported *Periods*.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    metric : str, :green:`optional` (set as 'curve_value')
        The pricing metric returned by :meth:`~rateslib.instruments.Value.rate`. See
        **Pricing**.

    """

    _rate_scalar = 1.0

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument*."""
        return self._leg1

    @property
    def legs(self) -> list[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs  # type: ignore[return-value]

    def __init__(
        self,
        # settlement
        effective: datetime,
        notional: float_ = NoInput(0),
        *,
        currency: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: Adjuster | str | int_ = NoInput(0),
        ex_div: Adjuster | str | int_ = NoInput(0),
        # non-deliverability
        pair: str_ = NoInput(0),
        fx_fixings: PeriodFixings = NoInput(0),
        # index-args:
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: PeriodFixings = NoInput(0),
        index_only: bool_ = NoInput(0),
        index_base_date: datetime_ = NoInput(0),
        index_reference_date: datetime_ = NoInput(0),
        # meta
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> None:
        user_args = dict(
            effective=effective,
            notional=notional,
            ex_div=ex_div,
            currency=currency,
            calendar=calendar,
            payment_lag=payment_lag,
            # non-deliverable
            pair=pair,
            fx_fixings=fx_fixings,
            # indexation
            index_base=index_base,
            index_lag=index_lag,
            index_method=index_method,
            index_fixings=index_fixings,
            index_only=index_only,
            index_base_date=index_base_date,
            index_reference_date=index_reference_date,
            # meta
            curves=self._parse_curves(curves),
            metric=metric,
            vol=_Vol(),
        )
        default_args = dict(
            metric="npv",
            notional=defaults.notional,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            calendar="all",
        )
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=user_args,
            default_args=default_args,
            meta_args=["curves", "metric", "vol"],
        )

        _ = _convert_to_schedule_kwargs(
            dict(
                effective=dt(1600, 1, 1),
                termination=effective,
                frequency=Frequency.Zero(),
                payment_lag=self.kwargs.leg1["payment_lag"],
                calendar=self.kwargs.leg1["calendar"],
                ex_div=self.kwargs.leg1["ex_div"],
            ),
            1,
        )["schedule"]

        self._leg1 = CustomLeg(
            periods=[
                Cashflow(
                    payment=_.pschedule[-1],
                    notional=self.kwargs.leg1["notional"],
                    currency=self.kwargs.leg1["currency"],
                    ex_dividend=_.pschedule3[-1],
                    # non-deliverable
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"],
                    delivery=NoInput(0),  # set as payment
                    # indexation
                    index_base=self.kwargs.leg1["index_base"],
                    index_lag=self.kwargs.leg1["index_lag"],
                    index_method=self.kwargs.leg1["index_method"],
                    index_fixings=self.kwargs.leg1["index_fixings"],
                    index_only=self.kwargs.leg1["index_only"],
                    index_base_date=self.kwargs.leg1["index_base_date"],
                    index_reference_date=self.kwargs.leg1["index_reference_date"],
                )
            ]
        )
        self._legs = [self._leg1]

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    @classmethod
    def _parse_curves(cls, curves: CurvesT_) -> _Curves:
        """
        A Value requires only one 1 curve, if not indexed, which is set as all element values.

        If the fee is indexed then an `index_curve` may be required.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("index_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    disc_curve=curves[0],
                    index_curve=curves[0],
                )
            elif len(curves) == 2:
                return _Curves(
                    disc_curve=curves[1],
                    index_curve=curves[0],
                )
            else:
                raise ValueError(
                    f"{type(cls).__name__} requires upto 2 curve type. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input
            return _Curves(
                disc_curve=curves,  # type: ignore[arg-type]
                index_curve=curves,  # type: ignore[arg-type]
            )

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
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "npv":
            return self.npv(  # type: ignore[return-value]
                curves=curves,
                solver=solver,
                fx=fx,
                vol=vol,
                base=base,
                settlement=settlement,
                forward=forward,
                local=False,
            )
        elif metric_ == "payment":
            return -1 * self.settlement_params.notional
        else:
            raise ValueError("`metric`must be in {'npv', 'cashflow'}.")

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        return super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
        )

    def cashflows(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )

    def analytic_delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        return super().analytic_delta(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement,
            forward=forward,
            leg=leg,
        )

    def local_analytic_rate_fixings(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
        )
