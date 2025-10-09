from typing import TypeAlias

from rateslib.periods.components.cashflow import (
    Cashflow,
    IndexCashflow,
    MtmCashflow,
    NonDeliverableCashflow,
    NonDeliverableIndexCashflow,
)
from rateslib.periods.components.credit import CreditPremiumPeriod, CreditProtectionPeriod
from rateslib.periods.components.fixed_period import (
    FixedPeriod,
    IndexFixedPeriod,
    NonDeliverableFixedPeriod,
    NonDeliverableIndexFixedPeriod,
)
from rateslib.periods.components.float_period import (
    FloatPeriod,
    IndexFloatPeriod,
    NonDeliverableFloatPeriod,
    NonDeliverableIndexFloatPeriod,
)
from rateslib.periods.components.fx_volatility import FXCallPeriod, FXOptionPeriod, FXPutPeriod
from rateslib.periods.components.protocols import _WithNPV, _WithNPVStatic

Period: TypeAlias = (
    "FixedPeriod | IndexFixedPeriod | NonDeliverableFixedPeriod | "
    "NonDeliverableIndexFixedPeriod | FloatPeriod | IndexFloatPeriod | "
    "NonDeliverableFloatPeriod | NonDeliverableIndexFloatPeriod | "
    "Cashflow | IndexCashflow | NonDeliverableCashflow | NonDeliverableIndexCashflow | "
    "MtmCashflow | CreditPremiumPeriod | CreditProtectionPeriod"
)

__all__ = [
    "_WithNPV",
    "_WithNPVStatic",
    "Cashflow",
    "IndexCashflow",
    "NonDeliverableCashflow",
    "NonDeliverableIndexCashflow",
    "MtmCashflow",
    "FixedPeriod",
    "IndexFixedPeriod",
    "NonDeliverableFixedPeriod",
    "NonDeliverableIndexFixedPeriod",
    "FloatPeriod",
    "IndexFloatPeriod",
    "NonDeliverableFloatPeriod",
    "NonDeliverableIndexFloatPeriod",
    "FXOptionPeriod",
    "FXCallPeriod",
    "FXPutPeriod",
    "CreditPremiumPeriod",
    "CreditProtectionPeriod",
]
