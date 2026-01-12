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

from datetime import datetime as dt

import pytest
import rateslib.errors as err
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    # IndexCashflow,
    # IndexFixedPeriod,
    # IndexFloatPeriod,
    # NonDeliverableCashflow,
    # NonDeliverableFixedPeriod,
    # NonDeliverableFloatPeriod,
    # NonDeliverableIndexCashflow,
    # NonDeliverableIndexFixedPeriod,
    # NonDeliverableIndexFloatPeriod,
    ZeroFixedPeriod,
)
from rateslib.periods.cashflow import MtmCashflow
from rateslib.scheduling import Schedule


class TestCashflow:
    def test_init(self):
        Cashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))
        pass


class TestFixedPeriod:
    def test_init(self):
        FixedPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
        )
        pass


class TestFloatPeriod:
    def test_init(self):
        FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
        )
        pass


# @pytest.mark.skip(reason="Deprecated objects.")
# class TestIndexFixedPeriod:
#     def test_init(self):
#         IndexFixedPeriod(
#             start=dt(2000, 1, 1),
#             end=dt(2000, 2, 1),
#             payment=dt(2000, 2, 1),
#             frequency="M",
#             notional=2e6,
#             currency="usd",
#             convention="act365f",
#             calendar="tgt",
#             adjuster="mf",
#             index_base=100.0,
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
#             IndexFixedPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#             )
#
#         with pytest.raises(ValueError, match=err.VE_HAS_ND_CURRENCY_PARAMS[:15]):
#             IndexFixedPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 index_base=100.0,
#                 pair="eurusd",
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestNonDeliverableIndexFixedPeriod:
#     def test_init(self):
#         NonDeliverableIndexFixedPeriod(
#             pair="eurusd",
#             start=dt(2000, 1, 1),
#             end=dt(2000, 2, 1),
#             payment=dt(2000, 2, 1),
#             frequency="M",
#             notional=2e6,
#             currency="usd",
#             convention="act365f",
#             calendar="tgt",
#             adjuster="mf",
#             index_base=100.0,
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
#             NonDeliverableIndexFixedPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 index_base=100.0,
#             )
#
#         with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
#             NonDeliverableIndexFixedPeriod(
#                 pair="eurusd",
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestNonDeliverableCashflow:
#     def test_init(self):
#         NonDeliverableCashflow(
#             currency="usd", pair="brlusd", notional=2e6, payment=dt(2000, 1, 1)
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
#             NonDeliverableCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))
#
#         with pytest.raises(ValueError, match=err.VE_HAS_INDEX_PARAMS[:15]):
#             NonDeliverableCashflow(
#                 currency="usd",
#                 pair="eurusd",
#                 notional=2e6,
#                 payment=dt(2000, 1, 1),
#                 index_base=100.0,
#             )
#
#     def test_undefined_currencies(self):
#         with pytest.raises(ValueError, match=err.VE_MISMATCHED_ND_PAIR[:15]):
#             NonDeliverableCashflow(
#                 pair="eurbrl",
#                 payment=dt(2000, 1, 1),
#                 notional=2e6,
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestIndexCashflow:
#     def test_init(self):
#         IndexCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1), index_base=100.0)
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
#             IndexCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))
#
#         with pytest.raises(ValueError, match=err.VE_HAS_ND_CURRENCY_PARAMS[:15]):
#             IndexCashflow(
#                 currency="usd",
#                 pair="eurusd",
#                 notional=2e6,
#                 payment=dt(2000, 1, 1),
#                 index_base=100.0,
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestNonDeliverableIndexCashflow:
#     def test_init(self):
#         NonDeliverableIndexCashflow(
#             currency="usd", pair="eurusd", notional=2e6, payment=dt(2000, 1, 1), index_base=100.0
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
#             NonDeliverableIndexCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))
#
#         with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
#             NonDeliverableIndexCashflow(
#                 currency="usd",
#                 notional=2e6,
#                 payment=dt(2000, 1, 1),
#                 index_base=100.0,
#             )


class TestMtmCashflow:
    def test_init(self):
        MtmCashflow(
            currency="usd",
            notional=2e6,
            payment=dt(2000, 1, 10),
            pair="eurusd",
            fx_fixings_start=2.0,
            fx_fixings_end=3.0,
            start=dt(2000, 1, 1),
            end=dt(2000, 1, 10),
        )


# @pytest.mark.skip(reason="Deprecated objects.")
# class TestNonDeliverableFixedPeriod:
#     def test_init(self):
#         NonDeliverableFixedPeriod(
#             start=dt(2000, 1, 1),
#             end=dt(2000, 2, 1),
#             payment=dt(2000, 2, 1),
#             frequency="M",
#             notional=2e6,
#             currency="usd",
#             convention="act365f",
#             calendar="tgt",
#             adjuster="mf",
#             pair="brlusd",
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
#             NonDeliverableFixedPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#             )
#
#         with pytest.raises(ValueError, match=err.VE_HAS_INDEX_PARAMS[:15]):
#             NonDeliverableFixedPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 pair="brlusd",
#                 index_base=100.0,
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestNonDeliverableFloatPeriod:
#     def test_init(self):
#         NonDeliverableFloatPeriod(
#             start=dt(2000, 1, 1),
#             end=dt(2000, 2, 1),
#             payment=dt(2000, 2, 1),
#             frequency="M",
#             notional=2e6,
#             currency="usd",
#             convention="act365f",
#             calendar="tgt",
#             adjuster="mf",
#             pair="brlusd",
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
#             NonDeliverableFloatPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#             )
#
#         with pytest.raises(ValueError, match=err.VE_HAS_INDEX_PARAMS[:15]):
#             NonDeliverableFloatPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 pair="brlusd",
#                 index_base=100.0,
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestIndexFloatPeriod:
#     def test_init(self):
#         IndexFloatPeriod(
#             start=dt(2000, 1, 1),
#             end=dt(2000, 2, 1),
#             payment=dt(2000, 2, 1),
#             frequency="M",
#             notional=2e6,
#             currency="usd",
#             convention="act365f",
#             calendar="tgt",
#             adjuster="mf",
#             index_base=100.0,
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
#             IndexFloatPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#             )
#
#         with pytest.raises(ValueError, match=err.VE_HAS_ND_CURRENCY_PARAMS[:15]):
#             IndexFloatPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 index_base=100.0,
#                 pair="eurusd",
#             )
#
#
# @pytest.mark.skip(reason="Deprecated objects.")
# class TestNonDeliverableIndexFloatPeriod:
#     def test_init(self):
#         NonDeliverableIndexFloatPeriod(
#             start=dt(2000, 1, 1),
#             end=dt(2000, 2, 1),
#             payment=dt(2000, 2, 1),
#             frequency="M",
#             notional=2e6,
#             currency="usd",
#             convention="act365f",
#             calendar="tgt",
#             adjuster="mf",
#             index_base=100.0,
#             pair="eurusd",
#         )
#         pass
#
#     def test_errors(self):
#         with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
#             NonDeliverableIndexFloatPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 pair="eurusd",
#             )
#
#         with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
#             NonDeliverableIndexFloatPeriod(
#                 start=dt(2000, 1, 1),
#                 end=dt(2000, 2, 1),
#                 payment=dt(2000, 2, 1),
#                 frequency="M",
#                 notional=2e6,
#                 currency="usd",
#                 convention="act365f",
#                 calendar="tgt",
#                 adjuster="mf",
#                 index_base=100.0,
#             )


class TestCreditPremiumPeriod:
    def test_init(self):
        CreditPremiumPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            premium_accrued=False,
        )


class TestCreditProtectionPeriod:
    def test_init(self):
        CreditProtectionPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
        )


class TestZeroFixedPeriod:
    def test_init(self):
        ZeroFixedPeriod(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2000, 9, 1),
                frequency="M",
            ),
            convention="act365f",
        )
