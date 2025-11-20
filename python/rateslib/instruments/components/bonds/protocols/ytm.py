from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.bonds.conventions import BOND_MODE_MAP
from rateslib.instruments.components.bonds.protocols.accrued import _WithAccrued

if TYPE_CHECKING:
    from rateslib.instruments.components.bonds.conventions import (  # pragma: no cover
        AccrualFunction,
        BondCalcMode,
        CashflowFunction,
        YtmDiscountFunction,
    )
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        FixedLeg,
        FloatLeg,
        Number,
        _KWArgs,
        datetime,
        str_,
    )


class _WithYTM(_WithAccrued, Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    @property
    def kwargs(self) -> _KWArgs: ...

    @property
    def leg1(self) -> FixedLeg | FloatLeg: ...

    def _price_from_ytm(
        self,
        ytm: DualTypes,
        settlement: datetime,
        calc_mode: BondCalcMode | str_,
        dirty: bool,
        curve: CurveOption_,
    ) -> DualTypes:
        """
        Loop through all future cashflows and discount them with ``ytm`` to achieve
        correct price.
        """
        calc_mode_ = _drb(self.kwargs.meta["calc_mode"], calc_mode)
        if isinstance(calc_mode_, str):
            calc_mode_ = BOND_MODE_MAP[calc_mode_]
        try:
            return self._generic_price_from_ytm(
                ytm=ytm,
                settlement=settlement,
                dirty=dirty,
                f1=calc_mode_._v1,
                f2=calc_mode_._v2,
                f3=calc_mode_._v3,
                c1=calc_mode_._c1,
                ci=calc_mode_._ci,
                cn=calc_mode_._cn,
                accrual=calc_mode_._ytm_accrual,
                curve=curve,
            )
        except KeyError:
            raise ValueError(f"Cannot calculate with `calc_mode`: {calc_mode}")

    def _generic_price_from_ytm(
        self,
        ytm: DualTypes,
        settlement: datetime,
        dirty: bool,
        f1: YtmDiscountFunction,
        f2: YtmDiscountFunction,
        f3: YtmDiscountFunction,
        c1: CashflowFunction,
        ci: CashflowFunction,
        cn: CashflowFunction,
        accrual: AccrualFunction,
        curve: CurveOption_,
    ) -> DualTypes:
        """
        Refer to supplementary material.

        Note: `curve` is only needed for FloatRate Periods on `_period_cashflow`
        """
        f: float = self.leg1.schedule.frequency_obj.periods_per_annum()
        acc_idx: int = self.leg1._period_index(settlement)
        _is_ex_div: bool = self.leg1.ex_div(settlement)
        if settlement == self.leg1.schedule.aschedule[acc_idx + 1]:
            # then settlement aligns with a cashflow: manually adjust to next period
            _is_ex_div = False
            acc_idx += 1

        v2 = f2(self, ytm, f, settlement, acc_idx, None, accrual, -100000)
        v1 = f1(self, ytm, f, settlement, acc_idx, v2, accrual, acc_idx)
        v3 = f3(
            self,
            ytm,
            f,
            settlement,
            self.leg1.schedule.n_periods - 1,
            v2,
            accrual,
            self.leg1.schedule.n_periods - 1,
        )

        # Sum up the coupon cashflows discounted by the calculated factors
        d: DualTypes = 0.0
        n = self.leg1.schedule.n_periods
        for i, p_idx in enumerate(range(acc_idx, n)):
            if i == 0 and _is_ex_div:
                # no coupon cashflow is received so no addition to the sum
                continue
            elif i == 0:
                # then this is the first period: c1 and v1 are used
                cf1 = c1(self, ytm, f, acc_idx, p_idx, n, curve)
                d += cf1 * v1
            elif p_idx == (self.leg1.schedule.n_periods - 1):
                # then this is last period, but it is not the first (i>0).
                # cn and v3 are relevant, but v1 is also used, and if i > 1 then v2 is also used.
                cfn = cn(self, ytm, f, acc_idx, p_idx, n, curve)
                d += cfn * v2 ** (i - 1) * v3 * v1
            else:
                # this is not the first and not the last period.
                # ci and v2i are relevant, but v1 is also required and v2 may also be used if i > 1.
                # v2i allows for a per-period adjustment to the v2 discount factor, e.g. BTPs.
                cfi = ci(self, ytm, f, acc_idx, p_idx, n, curve)
                v2i = f2(self, ytm, f, settlement, acc_idx, v2, accrual, p_idx)
                d += cfi * v2 ** (i - 1) * v2i * v1

        # Add the redemption payment discounted by relevant factors
        redemption: Cashflow | IndexCashflow = self.leg1._exchange_periods[1]  # type: ignore[assignment]
        if i == 0:  # only looped 1 period, only use the last discount
            d += self._period_cashflow(redemption, curve) * v1
        elif i == 1:  # only looped 2 periods, no need for v2
            d += self._period_cashflow(redemption, curve) * v3 * v1
        else:  # looped more than 2 periods, regular formula applied
            d += self._period_cashflow(redemption, curve) * v2 ** (i - 1) * v3 * v1

        # discount all by the first period factor and scaled to price
        p = d / -self.leg1.settlement_params.notional * 100

        return p if dirty else p - self._accrued(settlement, accrual)

    def _ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        curve: CurveOption_,
        dirty: bool,
    ) -> Number:
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float, Dual, Dual2
            The price, per 100 nominal, against which to determine the yield.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will assume the
            :meth:`~rateslib.instruments.FixedRateBond.accrued` is included in the price.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        """  # noqa: E501

        def s(g: DualTypes) -> DualTypes:
            return self._price_from_ytm(
                ytm=g, settlement=settlement, calc_mode=NoInput(0), dirty=dirty, curve=curve
            )

        result = ift_1dim(
            s,
            s_tgt=price,
            h="ytm_quadratic",
            ini_h_args=(-3.0, 2.0, 12.0),
            func_tol=1e-9,
            conv_tol=1e-9,
            raise_on_fail=True,
        )
        return result["g"]  # type: ignore[no-any-return]

    def ytm(self, price: DualTypes, settlement: datetime, dirty: bool = False) -> Number:
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float, Dual, Dual2, Variable
            The price, per 100 nominal, against which to determine the yield.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will assume the
            :meth:`~rateslib.instruments.components.FixedRateBond.accrued` is included in the price.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ytm(
               price=141.0701315,
               settlement=dt(1999,5,27),
               dirty=True
           )
           gilt.ytm(Dual(141.0701315, ["price", "a", "b"], [1, -0.5, 2]), dt(1999, 5, 27), True)
           gilt.ytm(Dual2(141.0701315, ["price", "a", "b"], [1, -0.5, 2], []), dt(1999, 5, 27), True)

        """  # noqa: E501
        return self._ytm(price=price, settlement=settlement, dirty=dirty, curve=NoInput(0))

    def _period_cashflow(self, period: Cashflow | FixedPeriod, curve: _BaseCurve_) -> DualTypes:  # type: ignore[override]
        """Nominal fixed rate bonds use the known "cashflow" attribute on the *Period*."""
        return period.cashflow()  # type: ignore[return-value]  # FixedRate on bond cannot be NoInput
