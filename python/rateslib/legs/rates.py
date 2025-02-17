from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.default import NoInput
from rateslib.legs.base import BaseLeg, _FixedLegMixin, _FloatLegMixin
from rateslib.periods import Cashflow, FixedPeriod, FloatPeriod
from rateslib.periods.utils import _validate_float_args

if TYPE_CHECKING:
    from pandas import DataFrame

    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        Schedule,
        datetime_,
        int_,
        str_,
    )


class FixedLeg(_FixedLegMixin, BaseLeg):  # type: ignore[misc]
    """
    Create a fixed leg composed of :class:`~rateslib.periods.FixedPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float, optional
        The rate applied to determine cashflows in % (i.e 5.0 = 5%). Can be left unset and
        designated later, perhaps after a mid-market rate for all periods has been calculated.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a fixed leg is the sum of the period NPVs.

    .. math::

       P = \\underbrace{- R \\sum_{i=1}^n {N_i d_i v_i(m_i)}}_{\\text{regular flows}} \\underbrace{+ N_1 v(m_0) - \\sum_{i=1}^{n-1}v(m_i)(N_{i}-N_{i+1})  - N_n v(m_n)}_{\\text{exchange flows}}

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial R} = \\sum_{i=1}^n {N_i d_i v_i(m_i)}

    Examples
    --------

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
       fixed_leg_exch = FixedLeg(
           dt(2022, 1, 1), "9M", "Q",
           fixed_rate=2.0,
           notional=1000000,
           amortization=200000,
           initial_exchange=True,
           final_exchange=True,
       )
       fixed_leg_exch.cashflows(curve)
       fixed_leg_exch.npv(curve)
    """  # noqa: E501

    periods: list[FixedPeriod | Cashflow]  # type: ignore[assignment]
    _regular_periods: tuple[FixedPeriod, ...]

    def __init__(
        self, *args: Any, fixed_rate: DualTypes | NoInput = NoInput(0), **kwargs: Any
    ) -> None:
        self._fixed_rate = fixed_rate
        super().__init__(*args, **kwargs)
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *FixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *FixedLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *FixedLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def _set_periods(self) -> None:
        return super()._set_periods()


class FloatLeg(_FloatLegMixin, BaseLeg):
    """
    Create a floating leg composed of :class:`~rateslib.periods.FloatPeriod` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseLeg`.
    float_spread : float, optional
        The spread applied to determine cashflows in bps (i.e. 100 = 1%). Can be set to `None`
        and designated later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, Series, 2-tuple, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``. If a 2-tuple of value and *Series*, the first
        scalar value is applied to the first period and latter periods handled as with *Series*.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----
    The NPV of a *FloatLeg* is the sum of the period NPVs.

    .. math::

       P = \\underbrace{- \\sum_{i=1}^n {N_i r_i(r_j, z) d_i v_i(m_i)}}_{\\text{regular flows}} \\underbrace{+ N_1 v(m_0) - \\sum_{i=1}^{n-1}v(m_i)(N_{i}-N_{i+1})  - N_n v(m_n)}_{\\text{exchange flows}}

    The analytic delta is the sum of the period analytic deltas.

    .. math::

       A = -\\frac{\\partial P}{\\partial z} = \\sum_{i=1}^n {\\frac{\\partial r_i}{\\partial z} N_i d_i v_i(m_i)}


    .. warning::

       When floating rates are determined from historical fixings the forecast
       ``Curve`` ``calendar`` will be used to determine fixing dates.
       If this calendar does not align with the ``Leg`` ``calendar`` then
       spurious results or errors may be generated.

       Including the curve calendar within a *Leg* multi-holiday calendar
       is acceptable, i.e. a *Leg* calendar of *"nyc,ldn,tgt"* and a curve
       calendar of *"ldn"* is valid. A *Leg* calendar of just *"nyc,tgt"* may
       give errors.

    Examples
    --------
    Set the first fixing on an historic IBOR leg.

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2021, 12, 1),
           termination="9M",
           frequency="Q",
           fixing_method="ibor",
           fixings=2.00,
       )
       float_leg.cashflows(curve)

    Set multiple fixings on an historic IBOR leg.

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2021, 9, 1),
           termination="12M",
           frequency="Q",
           fixing_method="ibor",
           fixings=[1.00, 2.00],
       )
       float_leg.cashflows(curve)

    It is **not** best practice to supply fixings as a list of values. It is better to supply
    a *Series* indexed by IBOR publication date (in this case lagged by zero days).

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2021, 9, 1),
           termination="12M",
           frequency="Q",
           fixing_method="ibor",
           method_param=0,
           fixings=Series([1.00, 2.00], index=[dt(2021, 9, 1), dt(2021, 12, 1)])
       )
       float_leg.cashflows(curve)

    Set the initial RFR fixings in the first period of an RFR leg (notice the sublist
    and the implied -10% year end turn spread).

    .. ipython:: python

       swestr_curve = Curve({dt(2023, 1, 2): 1.0, dt(2023, 7, 2): 0.99}, calendar="stk")
       float_leg = FloatLeg(
           effective=dt(2022, 12, 28),
           termination="2M",
           frequency="M",
           fixings=[[1.19, 1.19, -8.81]],
           currency="SEK",
           calendar="stk"
       )
       float_leg.cashflows(swestr_curve)
       float_leg.fixings_table(swestr_curve)[dt(2022,12,28):dt(2023,1,4)]

    Again, this is poor practice. It is **best practice** to supply a *Series* of RFR rates by
    reference value date.

    .. ipython:: python

       float_leg = FloatLeg(
           effective=dt(2022, 12, 28),
           termination="2M",
           frequency="M",
           fixings=Series([1.19, 1.19, -8.81], index=[dt(2022, 12, 28), dt(2022, 12, 29), dt(2022, 12, 30)]),
           currency="SEK",
           calendar="stk",
       )
       float_leg.cashflows(swestr_curve)
       float_leg.fixings_table(swestr_curve)[dt(2022,12,28):dt(2023,1,4)]
    """  # noqa: E501

    _delay_set_periods: bool = True  # do this to set fixings first
    _regular_periods: tuple[FloatPeriod, ...]
    schedule: Schedule

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(*args, **kwargs)
        self._set_fixings(fixings)
        self._set_periods()

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *FloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Return the properties of the *FloatLeg* used in calculating cashflows.

        For arguments see
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`.
        """
        return super().cashflows(*args, **kwargs)

    def npv(self, *args: Any, **kwargs: Any) -> NPV:
        """
        Return the NPV of the *FloatLeg* via summing all periods.

        For arguments see
        :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`.
        """
        return super().npv(*args, **kwargs)

    def fixings_table(
        self,
        curve: CurveOption_,
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curve : Curve, optional
            The forecasting curve object.
        disc_curve : Curve, optional
            The discounting curve object used in calculations.
            Set equal to ``curve`` if not given and ``curve`` is discount factor based.
        fx : float, FXRates, FXForwards, optional
            Only used in the case of :class:`~rateslib.legs.FloatLegMtm` to derive FX fixings.
        base : str, optional
            Not used by ``fixings_table``.
        approximate: bool
            Whether to use a faster (3x) but marginally less accurate (0.1% error) calculation.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        return super()._fixings_table(
            curve=curve, disc_curve=disc_curve, approximate=approximate, right=right
        )

    def _set_periods(self) -> None:
        return super(_FloatLegMixin, self)._set_periods()

    # @property
    # def _is_complex(self):
    #     """
    #     A complex float leg is one which is RFR based and for which each individual
    #     RFR fixing is required is order to calculate correctly. This occurs in the
    #     following cases:
    #
    #     1) The ``fixing_method`` is *"lookback"* - since fixing have arbitrary
    #        weightings misaligned with their standard weightings due to
    #        arbitrary shifts.
    #     2) The ``spread_compound_method`` is not *"none_simple"* - this is because the
    #        float spread is compounded alongside the rates so there is a non-linear
    #        relationship. Note if spread is zero this is negated and can be ignored.
    #     3) The ``fixing_method`` is *"lockout"* - technically this could be made semi
    #        efficient by splitting calculations into two parts. As of now it
    #        remains within the inefficient complex section.
    #     4) ``fixings`` are given which need to be incorporated into the calculation.
    #
    #
    #     """
    #     if self.fixing_method in ["rfr_payment_delay", "rfr_observation_shift"]:
    #         if self.fixings is not None:
    #             return True
    #         elif abs(self.float_spread) < 1e-9 or \
    #                 self.spread_compound_method == "none_simple":
    #             return False
    #         else:
    #             return True
    #     elif self.fixing_method == "ibor":
    #         return False
    #     return True
