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

from typing import TYPE_CHECKING, Protocol

from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.conventions import BOND_MODE_MAP
from rateslib.instruments.bonds.protocols.accrued import _WithAccrued

if TYPE_CHECKING:
    from rateslib.instruments.bonds.conventions import (  # pragma: no cover
        BondCalcMode,
    )
    from rateslib.instruments.bonds.conventions.accrued import (  # pragma: no cover
        AccrualFunction,
    )
    from rateslib.instruments.bonds.conventions.discounting import (  # pragma: no cover
        CashflowFunction,
        YtmDiscountFunction,
    )
    from rateslib.typing import (  # pragma: no cover
        Cashflow,
        CurveOption_,
        DualTypes,
        FixedLeg,
        FixedPeriod,
        FloatLeg,
        FloatPeriod,
        Number,
        _BaseCurve_,
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
        rate_curve: CurveOption_,
        index_curve: _BaseCurve_,
        indexed: bool,
    ) -> DualTypes:
        """
        Loop through all future cashflows and discount them with ``ytm`` to achieve
        correct price.
        """
        calc_mode_ = _drb(self.kwargs.meta["calc_mode"], calc_mode)
        if isinstance(calc_mode_, str):
            calc_mode_ = BOND_MODE_MAP[calc_mode_]
        try:
            if indexed:
                return self._generic_price_from_ytm_indexed(
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
                    rate_curve=rate_curve,
                    index_curve=index_curve,
                )
            else:
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
                    rate_curve=rate_curve,
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
        rate_curve: CurveOption_,
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
                cf1 = c1(self, ytm, f, acc_idx, p_idx, n, rate_curve)
                d += cf1 * v1
            elif p_idx == (self.leg1.schedule.n_periods - 1):
                # then this is last period, but it is not the first (i>0).
                # cn and v3 are relevant, but v1 is also used, and if i > 1 then v2 is also used.
                cfn = cn(self, ytm, f, acc_idx, p_idx, n, rate_curve)
                d += cfn * v2 ** (i - 1) * v3 * v1
            else:
                # this is not the first and not the last period.
                # ci and v2i are relevant, but v1 is also required and v2 may also be used if i > 1.
                # v2i allows for a per-period adjustment to the v2 discount factor, e.g. BTPs.
                cfi = ci(self, ytm, f, acc_idx, p_idx, n, rate_curve)
                v2i = f2(self, ytm, f, settlement, acc_idx, v2, accrual, p_idx)
                d += cfi * v2 ** (i - 1) * v2i * v1

        # Add the redemption payment discounted by relevant factors
        redemption: Cashflow = self.leg1._exchange_periods[1]  # type: ignore[assignment]
        if i == 0:  # only looped 1 period, only use the last discount
            d += self._period_cashflow(redemption, rate_curve) * v1
        elif i == 1:  # only looped 2 periods, no need for v2
            d += self._period_cashflow(redemption, rate_curve) * v3 * v1
        else:  # looped more than 2 periods, regular formula applied
            d += self._period_cashflow(redemption, rate_curve) * v2 ** (i - 1) * v3 * v1

        # discount all by the first period factor and scaled to price
        p = d / -self.leg1.settlement_params.notional * 100

        return p if dirty else p - self._accrued(settlement, accrual)

    def _generic_price_from_ytm_indexed(
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
        rate_curve: CurveOption_,
        index_curve: _BaseCurve_,
    ) -> DualTypes:
        """
        Very similar to `_generic_price_from_ytm` except every cashflow is indexed by the
        index ratio.
        """
        assert hasattr(self, "index_ratio")  # noqa: S101 # i.e. object is an IndexFixedRatedBond

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
            irn = self.index_ratio(self.leg1.schedule.aschedule[p_idx + 1], index_curve=index_curve)
            if i == 0 and _is_ex_div:
                # no coupon cashflow is received so no addition to the sum
                continue
            elif i == 0:
                # then this is the first period: c1 and v1 are used
                cf1 = c1(self, ytm, f, acc_idx, p_idx, n, rate_curve)
                d += cf1 * v1 * irn
            elif p_idx == (self.leg1.schedule.n_periods - 1):
                # then this is last period, but it is not the first (i>0).
                # cn and v3 are relevant, but v1 is also used, and if i > 1 then v2 is also used.
                cfn = cn(self, ytm, f, acc_idx, p_idx, n, rate_curve)
                d += cfn * v2 ** (i - 1) * v3 * v1 * irn
            else:
                # this is not the first and not the last period.
                # ci and v2i are relevant, but v1 is also required and v2 may also be used if i > 1.
                # v2i allows for a per-period adjustment to the v2 discount factor, e.g. BTPs.
                cfi = ci(self, ytm, f, acc_idx, p_idx, n, rate_curve)
                v2i = f2(self, ytm, f, settlement, acc_idx, v2, accrual, p_idx)
                d += cfi * v2 ** (i - 1) * v2i * v1 * irn

        # Add the redemption payment discounted by relevant factors
        redemption: Cashflow = self.leg1._exchange_periods[1]  # type: ignore[assignment]
        if i == 0:  # only looped 1 period, only use the last discount
            d += self._period_cashflow(redemption, rate_curve) * v1 * irn
        elif i == 1:  # only looped 2 periods, no need for v2
            d += self._period_cashflow(redemption, rate_curve) * v3 * v1 * irn
        else:  # looped more than 2 periods, regular formula applied
            d += self._period_cashflow(redemption, rate_curve) * v2 ** (i - 1) * v3 * v1 * irn

        # discount all by the first period factor and scaled to price
        p = d / -self.leg1.settlement_params.notional * 100

        settle_ir = self.index_ratio(settlement=settlement, index_curve=index_curve)
        return p if dirty else p - self._accrued(settlement, accrual) * settle_ir

    def _ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        rate_curve: CurveOption_,
        dirty: bool,
        indexed: bool,
        calc_mode: BondCalcMode | str_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
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
                ytm=g,
                settlement=settlement,
                calc_mode=calc_mode,
                dirty=dirty,
                rate_curve=rate_curve,
                index_curve=index_curve,
                indexed=indexed,
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

    def ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        rate_curve: CurveOption_ = NoInput(0),
        calc_mode: BondCalcMode | str_ = NoInput(0),
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

        .. image:: https://ebrary.net/imag/econom/smith_bondm/image232.jpg
           :align: center
           :alt: Image from ebrary.net
           :height: 310
           :width: 433

        .. role:: red

        .. role:: green

        Parameters
        ----------
        price: float, Dual, Dual2, Variable, :red:`required`
            The price, per 100 nominal, against which to determine the yield.
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
        return self._ytm(
            price=price,
            settlement=settlement,
            dirty=dirty,
            rate_curve=rate_curve,
            calc_mode=calc_mode,
            indexed=False,
        )

    def _period_cashflow(
        self, period: Cashflow | FixedPeriod | FloatPeriod, rate_curve: CurveOption_
    ) -> DualTypes:
        """Nominal fixed rate bonds use the known "cashflow" attribute on the *Period*."""
        return period.unindexed_cashflow(rate_curve=rate_curve)
