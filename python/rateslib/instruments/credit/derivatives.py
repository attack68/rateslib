from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.dual.utils import _dual_float
from rateslib.instruments.base import BaseDerivative
from rateslib.instruments.utils import (
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _update_not_noinput,
    _update_with_defaults,
)
from rateslib.legs import CreditPremiumLeg, CreditProtectionLeg

if TYPE_CHECKING:
    from rateslib.typing import FX_, NPV, Any, Curves_, DataFrame, DualTypes, Solver_, datetime


class CDS(BaseDerivative):
    """
    Create a credit default swap composing a :class:`~rateslib.legs.CreditPremiumLeg` and
    a :class:`~rateslib.legs.CreditProtectionLeg`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None, optional
        The rate applied to determine the cashflow on the premium leg. If `None`, can be set later,
        typically after a mid-market rate for all periods has been calculated.
        Entered in percentage points, e.g. 50bps is 0.50.
    premium_accrued : bool, optional
        Whether the premium is accrued within the period to default.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.
    """

    _rate_scalar = 1.0
    _fixed_rate_mixin = True
    leg1: CreditPremiumLeg
    leg2: CreditProtectionLeg
    kwargs: dict[str, Any]

    def __init__(
        self,
        *args: Any,
        fixed_rate: float | NoInput = NoInput(0),
        premium_accrued: bool | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        cds_specific: dict[str, Any] = dict(
            initial_exchange=False,  # CDS have no exchanges
            final_exchange=False,
            leg2_initial_exchange=False,
            leg2_final_exchange=False,
            leg2_frequency="Z",  # CDS protection is only ever one payoff
            fixed_rate=fixed_rate,
            premium_accrued=premium_accrued,
        )
        self.kwargs = _update_not_noinput(self.kwargs, cds_specific)

        # set defaults for missing values
        default_kwargs = dict(
            premium_accrued=defaults.cds_premium_accrued,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        self.leg1 = CreditPremiumLeg(**_get(self.kwargs, leg=1))
        self.leg2 = CreditProtectionLeg(**_get(self.kwargs, leg=2))
        self._fixed_rate = self.kwargs["fixed_rate"]

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
    ) -> None:
        # the test for an unpriced IRS is that its fixed rate is not set.
        if isinstance(self.fixed_rate, NoInput):
            # set a rate for the purpose of generic methods NPV will be zero.
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta`.
        """
        return super().analytic_delta(*args, **kwargs)

    def analytic_rec_risk(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic recovery risk of the derivative object.

        See :meth:`BaseDerivative.analytic_delta`.
        """
        return self.leg2.analytic_rec_risk(*args, **kwargs)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market credit spread of the CDS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg2_npv: DualTypes = self.leg2.npv(curves_[2], curves_[3], local=False)  # type: ignore[assignment]
        return self.leg1._spread(-leg2_npv, curves_[0], curves_[1]) * 0.01

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def accrued(self, settlement: datetime) -> DualTypes | None:
        """
        Calculate the amount of premium accrued until a specific date within the relevant *Period*.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        float or None

        Notes
        ------
        If the *CDS* is unpriced, i.e. there is no specified ``fixed_rate`` then None will be
        returned.
        """
        return self.leg1.accrued(settlement)
