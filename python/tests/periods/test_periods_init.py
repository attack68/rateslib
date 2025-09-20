from datetime import datetime as dt

import pytest
import rateslib.errors as err
from rateslib.periods.components import (
    Cashflow,
    FixedPeriod,
    FloatPeriod,
    IndexCashflow,
    IndexFixedPeriod,
    IndexFloatPeriod,
    NonDeliverableCashflow,
    NonDeliverableFixedPeriod,
    NonDeliverableFloatPeriod,
    NonDeliverableIndexCashflow,
    NonDeliverableIndexFixedPeriod,
    NonDeliverableIndexFloatPeriod,
)


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


class TestIndexFixedPeriod:
    def test_init(self):
        IndexFixedPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
            index_base=100.0,
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
            IndexFixedPeriod(
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

        with pytest.raises(ValueError, match=err.VE_HAS_ND_CURRENCY_PARAMS[:15]):
            IndexFixedPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                index_base=100.0,
                pair="eurusd",
            )


class TestNonDeliverableIndexFixedPeriod:
    def test_init(self):
        NonDeliverableIndexFixedPeriod(
            pair="eurusd",
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
            index_base=100.0,
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
            NonDeliverableIndexFixedPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                index_base=100.0,
            )

        with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
            NonDeliverableIndexFixedPeriod(
                pair="eurusd",
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


class TestNonDeliverableCashflow:
    def test_init(self):
        NonDeliverableCashflow(currency="usd", pair="brlusd", notional=2e6, payment=dt(2000, 1, 1))
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
            NonDeliverableCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))

        with pytest.raises(ValueError, match=err.VE_HAS_INDEX_PARAMS[:15]):
            NonDeliverableCashflow(
                currency="usd",
                pair="eurusd",
                notional=2e6,
                payment=dt(2000, 1, 1),
                index_base=100.0,
            )


class TestIndexCashflow:
    def test_init(self):
        IndexCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1), index_base=100.0)
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
            IndexCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))

        with pytest.raises(ValueError, match=err.VE_HAS_ND_CURRENCY_PARAMS[:15]):
            IndexCashflow(
                currency="usd",
                pair="eurusd",
                notional=2e6,
                payment=dt(2000, 1, 1),
                index_base=100.0,
            )


class TestNonDeliverableIndexCashflow:
    def test_init(self):
        NonDeliverableIndexCashflow(
            currency="usd", pair="eurusd", notional=2e6, payment=dt(2000, 1, 1), index_base=100.0
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
            NonDeliverableIndexCashflow(currency="usd", notional=2e6, payment=dt(2000, 1, 1))

        with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
            NonDeliverableIndexCashflow(
                currency="usd",
                notional=2e6,
                payment=dt(2000, 1, 1),
                index_base=100.0,
            )


class TestNonDeliverableFixedPeriod:
    def test_init(self):
        NonDeliverableFixedPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
            pair="brlusd",
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
            NonDeliverableFixedPeriod(
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

        with pytest.raises(ValueError, match=err.VE_HAS_INDEX_PARAMS[:15]):
            NonDeliverableFixedPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                pair="brlusd",
                index_base=100.0,
            )


class TestNonDeliverableFloatPeriod:
    def test_init(self):
        NonDeliverableFloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
            pair="brlusd",
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
            NonDeliverableFloatPeriod(
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

        with pytest.raises(ValueError, match=err.VE_HAS_INDEX_PARAMS[:15]):
            NonDeliverableFloatPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                pair="brlusd",
                index_base=100.0,
            )


class TestIndexFloatPeriod:
    def test_init(self):
        IndexFloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
            index_base=100.0,
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
            IndexFloatPeriod(
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

        with pytest.raises(ValueError, match=err.VE_HAS_ND_CURRENCY_PARAMS[:15]):
            IndexFloatPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                index_base=100.0,
                pair="eurusd",
            )


class TestNonDeliverableIndexFloatPeriod:
    def test_init(self):
        NonDeliverableIndexFloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            frequency="M",
            notional=2e6,
            currency="usd",
            convention="act365f",
            calendar="tgt",
            adjuster="mf",
            index_base=100.0,
            pair="eurusd",
        )
        pass

    def test_errors(self):
        with pytest.raises(ValueError, match=err.VE_NEEDS_INDEX_PARAMS[:15]):
            NonDeliverableIndexFloatPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                pair="eurusd",
            )

        with pytest.raises(ValueError, match=err.VE_NEEDS_ND_CURRENCY_PARAMS[:15]):
            NonDeliverableIndexFloatPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 2, 1),
                payment=dt(2000, 2, 1),
                frequency="M",
                notional=2e6,
                currency="usd",
                convention="act365f",
                calendar="tgt",
                adjuster="mf",
                index_base=100.0,
            )
