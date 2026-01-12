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

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.data.fixings import FXIndex, _get_fx_index
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_fx_forwards_maybe_from_solver,
    _Vol,
)
from rateslib.legs import CustomLeg
from rateslib.periods import Cashflow
from rateslib.periods.utils import _validate_fx_as_forwards
from rateslib.scheduling.frequency import _get_fx_expiry_and_delivery_and_payment

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Adjuster,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXIndex,
        FXIndex_,
        LegFixings,
        PeriodFixings,
        Sequence,
        Solver_,
        VolT_,
        _BaseLeg,
        bool_,
        datetime_,
        str_,
    )


class NDF(_BaseInstrument):
    """
    A *non-deliverable FX forward* (NDF), composing two
    :class:`~rateslib.legs.CustomLeg`
    of individual :class:`~rateslib.periods.Cashflow`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import fixings, NDF
       from datetime import datetime as dt
       from rateslib.data.fixings import FXIndex

    .. ipython:: python

       ndf = NDF(dt(2026, 1, 5), FXIndex("usdbrl", "fed", 2), fx_rate=5.5)
       ndf.cashflows()

    .. rubric:: Pricing

    The methods of an *NDF* require an :class:`~rateslib.fx.FXForwards` object for ``fx`` .

    They also require a *disc curve*, which is an appropriate curve to discount the
    cashflows of the deliverable settlement currency. The following input
    formats are allowed:

    .. code-block:: python

       curves = disc_curve | [disc_curve]  # one curve
       curves = [None, disc_curve, None, disc_curve]  # four curves
       curves = {  # dict form is explicit
           "disc_curve": disc_curve,
           "leg2_disc_curve": disc_curve,
       }

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following are **settlement parameters**.

    settlement : datetime, str, :red:`required`
        The date of settlement for the currency ``pair`` and payment date.
    pair : FXIndex, str, :red:`required`
        The :class:`~rateslib.data.fixings.FXIndex` containing the FX pair implying the
        reference currencies and notional of *leg1* and *leg2* respectively.
    currency : str, :green:`optional (set as LHS currency in pair)`
        The physical *settlement currency* of each leg. If not a currency in ``pair`` then each
        leg will be non-deliverable (3-digit code).
    notional : float, :green:`optional`
        The notional of *leg1* expressed in units of LHS currency of ``pair``. This can be
        derived from ``fx_rate`` and ``leg2_notional``.
    leg2_notional : float, :green:`optional`
        The notional of *leg2* expressed in units of RHS currency of ``pair``. This can be
        derived from ``fx_rate`` and ``notional``.
    fx_rate : float, :green:`optional`
        The transational FX rate of ``pair``. This can be derived from ``notional`` and
        ``leg2_notional``.

        .. note::

           The following are **scheduling parameters** required only if ``settlement`` given
           as string tenor.

    eval_date: datetime, :green:`optional`
        Today's date from which spot and other dates may be determined.
    modifier: Adjuster, str, :green:`optional`
        The date adjuster for determining tenor dates under the convention for ``pair``.
    eom: bool, :green:`optional`
        Whether tenors under ``pair`` adopt EOM convention or not.

        .. note::

           The following are **FX fixing parameters** defining the settlement of the transaction.

    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for settlement of *leg1* if
        that leg is non-deliverable. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    reversed: bool, :green:`optional (set as False)`
        Only used by a 3-currency NDF. Standard direction of the pair is '*settlement:reference*',
        unless ``reversed`` is *True*, in which case '*reference:settlement*' is used.
    leg2_fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing` for settlement of *leg2* if
        that leg is non-deliverable. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    leg2_reversed: bool, :green:`optional (set as False)`
        Only used by a 3-currency NDF. Standard direction of the pair is '*settlement:reference*',
        unless ``reversed`` is *True*, in which case '*reference:settlement*' is used.

        .. note::

           The following are **meta parameters**.

    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    spec: str, :green:`optional`
        A collective group of parameters. See
        :ref:`default argument specifications <defaults-arg-input>`.

    Notes
    -----
    *NDFs* in *rateslib* replicate an :class:`~rateslib.instruments.FXForward` whose cashflows
    are paid out netted in a single *settlement currency*. Two types are allowed:

    - A **two currency** *NDF* where one *Leg* is directly deliverable in its own currency and
      the other *Leg* is non-deliverable.
    - A **three currency** *NDF* when both *Legs* with cashflow currencies of ``pair`` are
      non-deliverable into a third ``currency``.

    .. ipython:: python

       fixings.add("WMR_10AM_TY0_USDINR", Series(index=[dt(2026, 2, 16)], data=[92.5]))
       fixings.add("WMR_10AM_TY0_USDSGD", Series(index=[dt(2026, 2, 16)], data=[1.290]))

    .. tabs::

       .. tab:: Two Currency NDF

          The **required** parameters of a two currency NDF are as follows;

          - A ``pair`` which defines the currency pair and implicitly determines the
            *reference currency*. The *settlement currency* for both *Legs* is inferred as the
            LHS, although this can be manually set by using the ``currency`` argument.
          - A ``notional`` or ``leg2_notional``. Each notional should be expressed in the
            *reference currency* for that *Leg*. If both are given that defines the
            transactional ``fx_rate``. If an ``fx_rate`` is given that will imply the missing
            notional.
          - ``fx_fixings`` or ``leg2_fx_fixings``. FX fixings can only be added to the
            non-deliverable *Leg*.

          This example is a USDINR *NDF* in 500mm INR payment with an initially agreed FX rate of
          USDINR 92.0

          .. ipython:: python

             ndf = NDF(
                 settlement=dt(2026, 2, 18),
                 currency="usd",              #  <-  USD settlement currency
                 pair="usdinr",               #  <-  INR reference currency implied
                 leg2_notional=500e6,         #  <-  Leg2 is based on the reference currency (INR)
                 leg2_fx_fixings="WMR_10AM_TY0",
                 fx_rate=92.0,                #  <-  Leg1 notional is implied as -5.43mm
             )
             ndf.cashflows()

       .. tab:: Three Currency NDXCS

          The **required** parameters of a three currency NDXCS are as follows;

          - A ``currency`` which defines the *settlement currency* on both legs.
          - A ``pair`` which defines the currency pair and implicitly determines
            the *reference currency 1* and *reference currency 2*.
          - A ``notional`` or ``leg2_notional``. Each notional should be expressed in the
            *reference currency* for that *Leg*. If both are given that defines the
            transactional ``fx_rate``. If an ``fx_rate`` is given that will imply the missing
            notional.
          - ``fx_fixings`` and ``leg2_fx_fixings``. Both legs are non-deliverable so FX fixings
            may be provided to both *Leg*.

          This example is a SGDINR *NDF* in 500mm INR payment with an initially agreed FX rate of
          SGDINR 70.1

          .. ipython:: python

             ndf = NDF(
                 settlement=dt(2026, 2, 18),
                 currency="usd",               #  <-  USD settlement currency
                 pair=FXIndex("SGDINR", "mum", 2),  #  <-  SGD + INR reference currencies
                 leg2_notional=500e6,          #  <-  INR notional
                 fx_rate=70.1,                 #  <-  Transaction rate of pair
                 fx_fixings="WMR_10AM_TY0",       #  <-  Data series tag for FXFixings on Leg1
                 leg2_fx_fixings="WMR_10AM_TY0",  #  <-  Data series tag for FXFixings on Leg2
             )
             ndf.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("WMR_10AM_TY0_USDINR")
       fixings.pop("WMR_10AM_TY0_USDSGD")

    """

    _rate_scalar = 1.0

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument*."""
        return self._leg1

    @property
    def leg2(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument*."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _Vol()

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An NDF requires 1 disc curve for the cashflows in the delivery currency.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                disc_curve=curves.get("disc_curve", NoInput(0)),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[0],
                )
            elif len(curves) == 4:
                return _Curves(
                    disc_curve=curves[1],
                    leg2_disc_curve=curves[3],
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} requires 1 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                disc_curve=curves,  # type: ignore[arg-type]
                leg2_disc_curve=curves,  # type: ignore[arg-type]
            )

    def __init__(
        self,
        settlement: datetime,
        pair: FXIndex | str,
        *,
        # settlement and rate
        currency: str_ = NoInput(0),
        fx_rate: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        leg2_notional: DualTypes_ = NoInput(0),
        # scheduling
        eval_date: datetime_ = NoInput(0),
        modifier: Adjuster | str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        # fx fixings
        fx_fixings: PeriodFixings = NoInput(0),
        leg2_fx_fixings: PeriodFixings = NoInput(0),
        reversed: bool_ = NoInput(0),  # noqa: A002
        leg2_reversed: bool_ = NoInput(0),
        # meta
        curves: CurvesT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ):
        (currency_, pair_, leg2_pair_, notional_, leg2_notional_, fx_rate_, fx_index_) = (
            _validated_ndf_input_combinations(
                currency=currency,
                pair=pair,
                notional=notional,
                leg2_notional=leg2_notional,
                fx_fixings=fx_fixings,
                leg2_fx_fixings=leg2_fx_fixings,
                fx_rate=fx_rate,
                reversed=reversed,
                leg2_reversed=leg2_reversed,
                spec=spec,
            )
        )
        del currency, pair, notional, leg2_notional, fx_rate

        user_args = dict(
            currency=currency_,
            pair=pair_,
            leg2_currency=currency_,
            leg2_pair=leg2_pair_,
            notional=notional_,
            leg2_notional=leg2_notional_,
            fx_rate=fx_rate_,
            curves=self._parse_curves(curves),
            eval_date=eval_date,
            modifier=modifier,
            eom=eom,
            settlement=settlement,
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
        )

        instrument_args = dict(  # these are hard coded arguments specific to this instrument
            vol=_Vol(),
            leg2_settlement=NoInput(1),
            fx_index=fx_index_,
        )
        default_args = dict(
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            modifier=defaults.modifier,
            eom=defaults.eom_fx,
        )
        self._kwargs = _KWArgs(
            spec=spec,
            user_args={**user_args, **instrument_args},
            default_args=default_args,
            meta_args=[
                "curves",
                "eval_date",
                "calendar",
                "modifier",
                "payment_lag",
                "eom",
                "vol",
                "fx_rate",
                "fx_index",
            ],
        )

        # post input determination for 'settlement'
        if not isinstance(self.kwargs.leg1["settlement"], datetime):
            _, settlement_, _ = _get_fx_expiry_and_delivery_and_payment(
                eval_date=self.kwargs.meta["eval_date"],
                expiry=self.kwargs.leg1["settlement"],
                delivery_lag=self.kwargs.meta["fx_index"].settle,
                calendar=self.kwargs.meta["fx_index"].calendar,
                modifier=self.kwargs.meta["modifier"],
                eom=self.kwargs.meta["eom"],
                payment_lag=0,
            )
            self.kwargs.leg1["settlement"] = settlement_
            self.kwargs.leg2["settlement"] = settlement_

        # construct legs
        self._leg1 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg1["currency"],
                    notional=-1.0
                    * (
                        0.0
                        if isinstance(self.kwargs.leg1["notional"], NoInput)
                        else self.kwargs.leg1["notional"]
                    ),
                    payment=self.kwargs.leg1["settlement"],
                    pair=self.kwargs.leg1["pair"],
                    fx_fixings=self.kwargs.leg1["fx_fixings"],
                ),
            ]
        )
        self._leg2 = CustomLeg(
            periods=[
                Cashflow(
                    currency=self.kwargs.leg2["currency"],
                    notional=-1.0
                    * (
                        0.0
                        if isinstance(self.kwargs.leg2["notional"], NoInput)
                        else self.kwargs.leg2["notional"]
                    ),
                    payment=self.kwargs.leg2["settlement"],
                    pair=self.kwargs.leg2["pair"],
                    fx_fixings=self.kwargs.leg2["fx_fixings"],
                )
            ]
        )
        self._legs = [self._leg1, self._leg2]

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
        fx_ = _validate_fx_as_forwards(_get_fx_forwards_maybe_from_solver(solver=solver, fx=fx))
        return fx_.rate(
            pair=self.kwargs.meta["fx_index"].pair, settlement=self.kwargs.leg1["settlement"]
        )

    def _set_pricing_mid(
        self,
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
    ) -> None:
        if isinstance(self.kwargs.meta["fx_rate"], NoInput):
            # determine the mid-market FX rate and set the notional of the appropriate leg
            mid_market_rate = self.rate(fx=fx, solver=solver)

            if isinstance(self.kwargs.leg2["notional"], NoInput):
                self.leg2.periods[0].settlement_params._notional = _dual_float(
                    -self.leg1.periods[0].settlement_params.notional * mid_market_rate
                )
            elif isinstance(self.kwargs.leg1["notional"], NoInput):
                self.leg1.periods[0].settlement_params._notional = _dual_float(
                    -self.leg2.periods[0].settlement_params.notional / mid_market_rate
                )
            else:
                raise RuntimeError(  # pragma: no cover
                    "The is no `notional` to determine. Please report this bug. Detailing the"
                    "initialisation of the NDF."
                )

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
        self._set_pricing_mid(
            solver=solver,
            fx=fx,
        )
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


def _validated_ndf_input_combinations(
    currency: str_,
    pair: FXIndex | str_,
    notional: DualTypes_,
    leg2_notional: DualTypes_,
    fx_fixings: LegFixings,
    leg2_fx_fixings: LegFixings,
    fx_rate: DualTypes_,
    reversed: bool_,  # noqa: A002
    leg2_reversed: bool_,
    spec: str_,
) -> tuple[str, FXIndex_, FXIndex_, DualTypes_, DualTypes_, DualTypes_, FXIndex]:
    """Method to handle arg parsing for 2 or 3 currency NDF instruments with default value
    setting and erroring raising.

    Returns
    -------
    (currency, pair, leg2_pair, notional, leg2_notional, fx_rate)
    """

    kw = _KWArgs(
        user_args=dict(
            currency=currency,
            leg2_currency=NoInput(1),
            pair=pair,
            notional=notional,
            leg2_notional=leg2_notional,
            fx_fixings=fx_fixings,
            leg2_fx_fixings=leg2_fx_fixings,
            fx_rate=fx_rate,
            reversed=reversed,
            leg2_reversed=leg2_reversed,
        ),
        default_args=dict(
            reversed=False,
            leg2_reversed=False,
        ),
        spec=spec,
        meta_args=["pair", "fx_rate"],
    )

    fx_index_ = _get_fx_index(kw.meta["pair"])

    # set a default settlement `currency` if none is provided
    if isinstance(kw.leg1["currency"], NoInput):
        kw.leg1["currency"] = fx_index_.pair[:3]
        kw.leg2["currency"] = fx_index_.pair[:3]
    else:
        kw.leg1["currency"] = kw.leg1["currency"].lower()
        kw.leg2["currency"] = kw.leg2["currency"].lower()

    if kw.leg1["currency"] not in fx_index_.pair:
        # then the NDF is a 3-currency instrument
        return _validated_3ccy_ndf_input_combinations(
            currency=kw.leg1["currency"],
            fx_index=fx_index_,
            notional=kw.leg1["notional"],
            leg2_notional=kw.leg2["notional"],
            fx_rate=kw.meta["fx_rate"],
            reversed=kw.leg1["reversed"],
            leg2_reversed=kw.leg2["reversed"],
        )
    else:
        return _validated_2ccy_ndf_input_combinations(
            currency=kw.leg1["currency"],
            fx_index=fx_index_,
            notional=kw.leg1["notional"],
            leg2_notional=kw.leg2["notional"],
            fx_fixings=kw.leg1["fx_fixings"],
            leg2_fx_fixings=kw.leg2["fx_fixings"],
            fx_rate=kw.meta["fx_rate"],
        )


def _validated_2ccy_ndf_input_combinations(
    currency: str,
    fx_index: FXIndex,
    notional: DualTypes_,
    leg2_notional: DualTypes_,
    fx_fixings: LegFixings,
    leg2_fx_fixings: LegFixings,
    fx_rate: DualTypes_,
) -> tuple[str, FXIndex_, FXIndex_, DualTypes_, DualTypes_, DualTypes_, FXIndex]:
    """Method to handle arg parsing for 2 currency NDF instruments with default value
    setting and erroring raising.

    Notional:
    if no notional is given then leg1 is set from 'defaults'
    if both notionals are given then the fx_rate is inferred.
    if one notional and the fx_rate is given then the alternative notional is inferred.
    two notionals AND fx_rate imply possible triangulation failure and raise
    notional can be given on any leg and the alternative notional is inferred from the `fx_rate`

    Returns
    -------
    (currency, pair, leg2_pair, notional, leg2_notional, fx_rate)
    """
    leg1_nd = fx_index.pair[3:] == currency
    if leg1_nd:
        pair_: FXIndex_ = fx_index
        leg2_pair_: FXIndex_ = NoInput(0)
    else:
        pair_ = NoInput(0)
        leg2_pair_ = fx_index

    notional_, leg2_notional_, fx_rate_ = _notional_and_fx_rate_validation(
        notional, leg2_notional, fx_rate
    )

    # parse the fixings input: should only be relevant for the single non-deliverable leg
    if not leg1_nd and not isinstance(fx_fixings, NoInput):
        raise ValueError(
            f"Leg1 of NDF is directly deliverable (reference ccy '{fx_index.pair[:3]}' and "
            f"settlement ccy '{currency}').\n"
            "Do not supply `fx_fixings` for leg1, perhaps you meant `leg2_fx_fixings`?"
        )
    if leg1_nd and not isinstance(leg2_fx_fixings, NoInput):
        raise ValueError(
            f"Leg2 of NDF is directly deliverable (reference ccy '{fx_index.pair[3:]}' and "
            f"settlement ccy '{currency}').\n"
            "Do not supply `leg2_fx_fixings` for leg2, perhaps you meant `fx_fixings`?"
        )

    return (
        currency,
        pair_,
        leg2_pair_,
        notional_,
        leg2_notional_,
        fx_rate_,
        fx_index,
    )


def _validated_3ccy_ndf_input_combinations(
    currency: str,
    fx_index: FXIndex,
    notional: DualTypes_,
    leg2_notional: DualTypes_,
    fx_rate: DualTypes_,
    reversed: bool,  # noqa: A002
    leg2_reversed: bool,
) -> tuple[str, FXIndex_, FXIndex_, DualTypes_, DualTypes_, DualTypes_, FXIndex]:
    """Method to handle arg parsing for 3 currency NDF instruments with default value
    setting and erroring raising.

    Returns
    -------
    (currency, pair, leg2_pair, notional, leg2_notional, fx_rate)
    """
    # both legs are non-deliverable
    if reversed:
        pair = f"{fx_index.pair[:3]}{currency}"
    else:
        pair = f"{currency}{fx_index.pair[:3]}"

    if leg2_reversed:
        leg2_pair = f"{fx_index.pair[3:]}{currency}"
    else:
        leg2_pair = f"{currency}{fx_index.pair[3:]}"

    try:
        pair_index: FXIndex = _get_fx_index(pair)
    except ValueError:
        # no index exists in STATIC, clone from fx_index
        pair_index = FXIndex(pair=pair, calendar=fx_index.calendar, settle=fx_index.settle)
    pair_index = FXIndex(
        pair=pair_index.pair,
        calendar=pair_index.calendar,
        settle=pair_index.settle,
        isda_mtm_calendar=fx_index.isda_mtm_calendar,
        isda_mtm_settle=fx_index.isda_mtm_settle,
    )

    try:
        leg2_pair_index: FXIndex = _get_fx_index(leg2_pair)
    except ValueError:
        # no index exists in STATIC, clone from fx_index
        leg2_pair_index = FXIndex(pair=pair, calendar=fx_index.calendar, settle=fx_index.settle)
    leg2_pair_index = FXIndex(
        pair=leg2_pair_index.pair,
        calendar=leg2_pair_index.calendar,
        settle=leg2_pair_index.settle,
        isda_mtm_calendar=fx_index.isda_mtm_calendar,
        isda_mtm_settle=fx_index.isda_mtm_settle,
    )

    notional_, leg2_notional_, fx_rate_ = _notional_and_fx_rate_validation(
        notional, leg2_notional, fx_rate
    )

    return (
        currency,
        pair_index,
        leg2_pair_index,
        notional_,
        leg2_notional_,
        fx_rate_,
        fx_index,
    )


def _notional_and_fx_rate_validation(
    notional: DualTypes_,
    leg2_notional: DualTypes_,
    fx_rate: DualTypes_,
) -> tuple[DualTypes_, DualTypes_, DualTypes_]:
    """
    method to parse the input arguments in their various combinations.

    Notional:
    if no notional is given then leg1 is set from 'defaults'
    if both notionals are given then the fx_rate is inferred.
    if one notional and the fx_rate is given then the alternative notional is inferred.
    two notionals AND fx_rate imply possible triangulation failure and raise
    notional can be given on any leg and the alternative notional is inferred from the `fx_rate`
    """

    # set a default `notional` if no notional on any leg is given
    if isinstance(notional, NoInput) and isinstance(leg2_notional, NoInput):
        notional_: DualTypes_ = defaults.notional
        leg2_notional_: DualTypes_ = leg2_notional
    else:
        notional_ = notional
        leg2_notional_ = leg2_notional
    del notional, leg2_notional

    # parse fx_rate / notional / and / leg2_notional
    if not isinstance(notional_, NoInput) and not isinstance(leg2_notional_, NoInput):
        if not isinstance(fx_rate, NoInput):
            raise ValueError(
                "`notional`, `leg2_notional` and `fx_rate` cannot all be given simultaneously.\n"
                "Provide, at most, two of these arguments for an NDF."
            )
        if notional_ * leg2_notional_ > 0:
            raise ValueError(
                "When providing `notional` and `leg2_notional` on an NDF, the two must be opposite "
                "signs, indicating both a buy and a sell ."
            )
        else:
            fx_rate_: DualTypes_ = -leg2_notional_ / notional_
    elif isinstance(notional_, NoInput) and not isinstance(leg2_notional_, NoInput):
        if isinstance(fx_rate, NoInput):
            # then the NDF is unpriced and will requiring setting to mid-market at price time
            fx_rate_ = NoInput(0)
        else:
            fx_rate_ = fx_rate
            notional_ = -leg2_notional_ / fx_rate
    elif not isinstance(notional_, NoInput) and isinstance(leg2_notional_, NoInput):
        if isinstance(fx_rate, NoInput):
            # then the NDF is unpriced and will requiring setting to mid-market at price time
            fx_rate_ = NoInput(0)
        else:
            fx_rate_ = fx_rate
            leg2_notional_ = -notional_ * fx_rate
    else:
        raise RuntimeError(  # pragma: no cover
            "This line should never be reached. "
            "Report issue for NDF initialization providing input arguments."
        )

    return notional_, leg2_notional_, fx_rate_
