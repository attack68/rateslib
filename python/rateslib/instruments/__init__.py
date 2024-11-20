# Sphinx substitutions

"""
.. ipython:: python
   :suppress:

   from rateslib import *
"""

from __future__ import annotations

from rateslib.instruments.bonds import (
    BILL_MODE_MAP,
    BOND_MODE_MAP,
    BillCalcMode,
    BondCalcMode,
    FixedRateBond,
    IndexFixedRateBond,
    BondFuture,
    Bill,
    FloatRateNote,
)
from rateslib.instruments.core import (
    BaseMixin,
    Sensitivities,
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _push,
    _update_with_defaults,
)
from rateslib.instruments.fx_volatility import (
    FXBrokerFly,
    FXCall,
    FXOption,
    FXOptionStrat,
    FXPut,
    FXRiskReversal,
    FXStraddle,
    FXStrangle,
)
from rateslib.instruments.generics import Fly, Portfolio, Spread, Value, VolValue
from rateslib.instruments.rates_derivatives import (
    CDS,
    FRA,
    IIRS,
    IRS,
    SBS,
    ZCIS,
    ZCS,
    BaseDerivative,
    STIRFuture,
)
from rateslib.instruments.rates_multi_ccy import (
    XCS,
    FXExchange,
    FXSwap,
)

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

__all__ = [
    "BaseDerivative",
    "BaseMixin",
    "Bill",
    "BondMixin",
    "BondCalcMode",
    "BillCalcMode",
    "BondFuture",
    "CDS",
    "FRA",
    "FXBrokerFly",
    "FXCall",
    "FXExchange",
    "FXOption",
    "FXOptionStrat",
    "FXPut",
    "FXRiskReversal",
    "FXStraddle",
    "FXStrangle",
    "FXSwap",
    "FixedRateBond",
    "FloatRateNote",
    "Fly",
    "IIRS",
    "IRS",
    "IndexFixedRateBond",
    "Portfolio",
    "SBS",
    "STIRFuture",
    "Sensitivities",
    "Spread",
    "Value",
    "VolValue",
    "XCS",
    "ZCIS",
    "ZCS",
]
