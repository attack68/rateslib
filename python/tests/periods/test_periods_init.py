from datetime import datetime as dt

from rateslib.periods.components import (
    Cashflow,
    FixedPeriod,
    FloatPeriod,
    IndexFixedPeriod,
    IndexFloatPeriod,
    NonDeliverableCashflow,
    NonDeliverableFixedPeriod,
    NonDeliverableFloatPeriod,
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


class TestNonDeliverableCashflow:
    def test_init(self):
        NonDeliverableCashflow(currency="usd", pair="brlusd", notional=2e6, payment=dt(2000, 1, 1))
        pass


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
