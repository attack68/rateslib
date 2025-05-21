from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.instruments.bonds.conventions.accrued import ACC_FRAC_FUNCS
from rateslib.instruments.bonds.conventions.discounting import C_FUNCS, V1_FUNCS, V2_FUNCS, V3_FUNCS

if TYPE_CHECKING:
    from rateslib.instruments.bonds.conventions.accrued import AccrualFunction
    from rateslib.instruments.bonds.conventions.discounting import (
        CashflowFunction,
        YtmDiscountFunction,
        YtmStubDiscountFunction,
    )
    from rateslib.typing import Security


class BondCalcMode:
    """
    Define calculation conventions for :class:`~rateslib.instruments.FixedRateBond`,
    :class:`~rateslib.instruments.IndexFixedRateBond` and
    :class:`~rateslib.instruments.FloatRateNote` types.

    For a list of :class:`~rateslib.instruments.BondCalcMode` that have already
    been pre-defined see :ref:`Securities Defaults <defaults-securities-input>`.
    
    Parameters
    ----------
    settle_accrual: str or Callable
        The calculation type for accrued interest for physical settlement. See notes.
    ytm_accrual: str or Callable
        The calculation method for accrued interest used in the YTM formula. Often the same
        as above but not always (e.g. Canadian GBs). See notes.
    v1: str or Callable
        The calculation function that defines discounting of the first period of the YTM formula.
    v2: str or Callable
        The calculation function that defines discounting of the regular periods of the YTM formula.
    v3: str or Callable
        The calculation function that defines discounting of the last period of the YTM formula.
    c1: str or Callable
        The calculation function that determines the cashflow amount in the first period of the
        YTM formula.
    ci: str or Callable
        The calculation function that determines the cashflow amount in the interim periods of the
        YTM formula.
    cn: str or Callable
        The calculation function that determines the cashflow amount in the final period of the
        YTM formula.

    Notes
    -------
    For an example custom implementation of a *BondCalcMode* see the cookbook article:
    :ref:`Understanding and Customising FixedRateBond Conventions <cook-bond_convs>`

    **Notation**
    
    The following notation is used in this section:
    
    - :math:`\\xi`: The **accrual fraction** is a float, typically, in [0, 1] which defines the
      amount of a bond's current cashflow period that is paid at *settlement* as accrued interest.
    - :math:`\\xi_y`: The **accrual fraction** determined in a secondary method, used only in YTM
      calculations and **not** for physical settlement.
      (Almost always :math:`\\xi_y` and :math:`\\xi` are the same, for an exception see
      Canadian GBs)
    - :math:`r_u`: The number of calendar days between the last (unadjusted) coupon date and
      settlement. If a **long stub** this is either; zero if settlement falls before the
      (unadjusted) quasi-coupon date, or the number of calendar days between
      those dates.
    - :math:`s_u`: The number of calendar days between the last (unadjusted) coupon date and the
      next (unadjusted) coupon date, i.e the number of calendar days in the (unadjusted) coupon
      period. If a **long stub** this is the number of calendar days in the (unadjusted)
      quasi-coupon period.
    - :math:`\\bar{r}_u`: If a **long stub**, the number of calendar days between the accrual
      effective date and either; the next (unadjusted) quasi-coupon date, or settlement date,
      whichever is earliest.
    - :math:`\\bar{s}_u`: If a **long stub**, the number of calendar days between the prior
      (unadjusted) quasi-coupon date and the (unadjusted) next quasi-coupon date surrounding the
      accrual effective date.
    - :math:`d_i`: The full DCF of coupon period, *i*, calculated with the convention which
      determines the physical cashflows.
    - :math:`f`: The frequency of the coupon as integer, 1-annually, 2-semi, 3-tertiary,
      4-quarterly, 6-bi-monthly, 12-monthly.
    - :math:`\\bar{d}_u`: The DCF between settlement and the next (unadjusted) coupon date
      determined with the convention of the accrual function (which may be different to the
      convention for determining physical bond cashflows)
    - :math:`c_i`: A coupon cashflow monetary amount, **per 100 nominal**, for coupon period, *i*.
    - :math:`p_d`: Number of days between unadjusted coupon date and payment date in a coupon
      period, i.e. the pay delay.
    - :math:`p_D` = Number of days between previous payment date and current payment date, in a
      coupon period.
    - :math:`C`: The nominal annual coupon rate for the bond.
    - :math:`y`: The yield-to-maturity for a given bond. The expression of which, i.e. annually
      or semi-annually is derived from the calculation context.

    **Accrual Functions**

    Accrual functions must be supplied to the ``settle_accrual`` and ``ytm_accrual``
    arguments, and must output **accrual fractions**. The available values are:

    - ``linear_days``: A calendar day, linear proportion used in any period.
      (Used by UK and German GBs).
    
      .. math::
      
         \\xi = r_u / s_u
    
    - ``linear_days_long_front_split``: A modified version of the above which, **only for long
      stub** periods, uses a different formula treating the first quasi period as part of the
      long stub differently. This adjustment is then scaled according to the length of the period.
      (Treasury method for US Treasuries, see Section 31B ii A.356, Code of Federal Regulations)
      
      .. math::
      
         \\xi = (\\bar{r}_u / \\bar{s}_u + r_u / s_u) / ( d_i * f )
      
    - ``30u360_backward``: For **stubs** this method reverts to ``linear_days``. Otherwise,
      determines the DCF, under the required convention, of the remaining part of the coupon
      period from settlement and deducts this from the full accrual fraction.
      
      .. math::
      
         \\xi = 1 - \\bar{d_u} f
      
    - ``30u360_forward``: Calculates the DCF between last (unadjusted) coupon and settlement,
      and compares this with DCF between (unadjusted) coupon dates, both measured using *'30u360'*
      (See MSRB Rule G-33):
      
      .. math::
      
         \\xi = DCF(prior, settlement) / DCF(prior, next)
           
    - ``act365f_1y``: For **stubs** this method reverts to ``linear_days``. Otherwise,
      determines the accrual fraction using an approach that uses ACT365F convention.
      (Used by Canadian GBs)
      
      .. math::
      
         \\xi = \\left \\{ \\begin{matrix} 1.0 & \\text{if, } r_u = s_u \\\\ 1.0 - f(s_u - r_u) / 365 & \\text{if, } r_u \\ge 365 / f \\\\ fr_u / 365 & \\text{if, } r_u < 365 / f \\\\ \\end{matrix} \\right .

    **Custom accrual functions** can also be supplied where the input arguments signature should
    accept the bond object, the settlement date, and the index relating to the period in which
    the relevant coupon period falls. It should return an accrual fraction upto settlement.
    As an example the code below shows the implementation of the
    *"linear_days"* accrual function:

    .. ipython:: python

       def _linear_days(obj, settlement, acc_idx, *args) -> float:
            sch = obj.leg1.schedule
            r_u = (settlement - sch.uschedule[acc_idx]).days
            s_u = (sch.uschedule[acc_idx + 1] - sch.uschedule[acc_idx]).days
            return r_u / s_u
            
    **Calculation of Accrued Interest**
    
    Accrued interest, *AI*, is then calculated according to the following:
    
    .. math::
    
       &AI = \\xi c_i \\qquad \\text{if not ex-dividend} \\\\
       &AI = (\\xi - 1) c_i \\qquad \\text{if ex-dividend} \\\\
       
    And accrued interest for the purpose of YTM calculations, :math:`AI_y`, is:
    
    .. math::
    
       &AI_y = \\xi_y c_i \\qquad \\text{if not ex-dividend} \\\\
       &AI_y = (\\xi_y - 1) c_i \\qquad \\text{if ex-dividend} \\\\

    Where in these formula :math:`c_i` currently always uses the ``cashflow`` method (see below).

    **YTM Calculation and Required Functions**

    Yield-to-maturity is calculated using the below formula, where specific discounting and
    cashflow functions must be provided to determine values based on the conventions of a given
    bond. The below formula outlines the
    cases where the number of remaining coupons are 1, 2, or generically >2.

    .. math::

       P &= v_1 \\left ( c_1 + 100 \\right ), \\quad n = 1 \\\\
       P &= v_1 \\left ( c_1 + v3(c_n + 100) \\right ), \\quad n = 2 \\\\
       P &= v_1 \\left ( c_1 + \\sum_{i=2}^{n-1} c_i v_2^{i-2} v_{2,i} + c_nv_2^{n-2}v_3 + 100 v_2^{n-2}v_3 \\right ), \\quad n > 2  \\\\
       Q &= P - AI_y

    where,

    .. math::

       P &= \\text{Dirty price}, \\; Q = \\text{Clean Price} \\\\
       n &= \\text{Coupon periods remaining} \\\\
       c_1 &= \\text{Cashflow (per 100) on next coupon date (may be zero if ex-dividend)} \\\\
       c_i &= i \\text{'th cashflow (per 100) on subsequent coupon dates} \\\\
       v_1 &= \\text{Discount value for the initial, possibly stub, period} \\\\
       v_2 &= \\text{General discount value for the interim regular periods} \\\\
       v_{2,i} &= \\text{Specific discount value for the i'th interim regular period} \\\\
       v_3 &= \\text{Discount value for the final, possibly stub, period} \\\\

    **v2 Functions**
    
    *v2* forms the core, regular part of discounting the cashflows. *v2* functions are required when
    a bond has more than two coupon remaining. This reflects coupon periods that are
    never stubs. The available functions are described below:

    - ``regular``: uses the traditional discounting function matching the actual frequency of
      coupons:

      .. math::

         v_2 = \\frac{1}{1 + y/f}

    - ``annual``: assumes an annually expressed YTM disregarding the actual coupon frequency:

      .. math::

         v_2 = \\left ( \\frac{1}{1 + y} \\right ) ^ {1/f}
         
    - ``annual_pay_adjust``: an extension to ``annual`` that adjusts the period in scope to
      account for a delay between its unadjusted coupon end date and the actual payment date. (Used
      by Italian BTPs)
      
      .. math::
      
         v_2 = \\left ( \\frac{1}{1 + y} \\right ) ^ {1/f}, \\qquad \\text{and in the current period} \\qquad v_{2,i} = v_2 ^ {(1 + p_d / p_D)}
              
    **v1 Functions**

    *v1* functions are required for every bond. Its value may, or may not, be dependent upon *v2*.
    *v1* functions have to handle the cases whereby the coupon period in which *settlement* falls
    is
    
    - The first coupon period, **and** it may be a **stub**,
    - A regular interim coupon period,
    - The final coupon period **and** it may be a **stub**.
     
    The two most common functions for determining *v1* are described below:
    
    - ``compounding``: If a **stub** then scaled by the length of
      the stub. At issue, or on a coupon date, for a regular period, *v1* converges to *v2*.
         
      .. math::
      
         v_1 =  v_2^{g(\\xi_y)}  \\quad \\text{where,} \\quad g(\\xi_y) = \\left \\{ \\begin{matrix} 1-\\xi_y & \\text{if regular,} \\\\ (1-\\xi_y) f d_i & \\text{if stub,} \\\\ \\end{matrix} \\right . \\\\

    - ``simple``: calculation uses a simple interest formula. At issue, or on a coupon date,
      for a regular period, *v1* converges to a *'regular'* style *v2*.
    
      .. math::
      
         v_1 = \\frac{1}{1 + g(\\xi_y) y / f}  \\quad \\text{where, } g(\\xi_y) \\text{ defined as above}

    Combinations, or extensions, of the two above functions are also required for some
    bond conventions:

    - ``compounding_final_simple``: uses ``compounding``, unless settlement occurs in the final
      period of the bond (and in which case n=1) and then the ``simple`` method is applied.
    - ``compounding_stub_act365f``: uses ``compounding``, unless settlement occurs in a stub
      period in which case Act365F convention derives the exponent.
      
      .. math::
      
         v_1 = v_2^{\\bar{d}_u} \\qquad \\text{if stub.}

    - ``simple_long_stub_compounding``: uses ``simple`` formula **except** for long stubs,
      and the calculation is only different if settlement falls before the quasi-coupon.
      If settlement occurs before the quasi-coupon date then the entire quasi-coupon period
      applies regular *v2* discounting, and the preliminary component has *simple* method
      applied.
      
      .. math::

         v_1 = v_2 \\frac{1}{1 + [f d_i(1 - \\xi_y) - 1] y / f} \\qquad \\text{if settlement before quasi-coupon in long stub}

    - ``simple_pay_adjust``: adjusts the *'simple'* method to account for the payment date.
    
      .. math::
      
         v_1 = \\frac{1}{1 + g_p(\\xi_y) y / f} \\quad \\text{where,} \\quad g_p(\\xi_y) = \\left \\{ \\begin{matrix} 1-\\xi_y + p_d / p_D & \\text{if regular,} \\\\ (1-\\xi_y + p_d / p_D) f d_i & \\text{if stub,} \\\\ \\end{matrix} \\right .
    
    - ``compounding_pay_adjust``: adjusts the *'compounding'* method to account for payment date.
    
      .. math::
      
         v_1 = v_2^{g_p(\\xi_y)}  \\quad \\text{where, } g_p(\\xi_y) \\text{ defined as above}
         
    - ``compounding_final_simple_pay_adjust``: uses ``compounding`` unless settlement
      occurs in the final period of the bond (and in which case n=1) and then the
      ``simple_pay_adjust`` method is applied.
    
    
    **v3 Functions**

    *v3* functions will never have a settlement mid period, and are only used in the case
    of 2 or more remaining coupon periods. The available functions are:

    - ``compounding``: is identical to *v1 'compounding'* where :math:`\\xi_y` is set to zero.
    - ``compounding_pay_adjust``: is identical to *v1 'compounding_pay_adjust'* where :math:`\\xi_y` is set to zero.
    - ``simple``: is identical to *v1 'simple'* where :math:`\\xi_y` is set to zero.
    - ``simple_pay_adjust``: is identical to *v1 'simple_pay_adjust'* where :math:`\\xi_y`
      is set to zero.
    - ``simple_30e360``: uses simple interest with a DCF calculated
      under 30e360 convention, irrespective of the bond's underlying convention.
      
      .. math::

         v_3 = \\frac{1}{1+\\bar{d}_n y}

    **Custom discount functions** can also be supplied where the input arguments signature
    is shown in the below example. It should return a discount factor. The example
    shows the implementation of the *"regular"* discount function:

    .. ipython:: python

       def _v2_(
           obj,         # the bond object
           ytm,         # y as defined
           f,           # f as defined
           settlement,  # datetime
           acc_idx,     # the index of the period in which settlement occurs
           v2,          # the numeric value of v2 already calculated
           accrual,     # the ytm_accrual function to return accrual fractions
       ):
           return 1 / (1 + ytm / (100 * f))
           
    **Cashflow Generating Functions**
    
    Most of the time, for the cashflows shown above in the YTM formula, the actual cashflows, as
    determined by the native *schedule* and *convention* on the bond itself, can be used.
    
    This is because the cashflow often aligns with a *typical* expected amount,
    i.e. *coupon / frequency*. Since this is by definition under the *ActActICMA* convention
    and unadjusted *30360* will also tend to return standardised coupons.
    
    However, some bonds use a *convention* which does not lead to standardised
    coupons, but have YTM formula definitions which do require standardised coupons. An example
    is Thai Government Bonds.
    
    The available functions here are:
    
    - ``cashflow``: determine the cashflow for the period by using the native cashflow calculation
      under the *schedule* and *convention* on the bond.
    - ``full_coupon``: determine the cashflow as a full coupon payment, irrespective of period
      dates, based on the notional of the period and the coupon rate of the bond. This method is
      only for fixed rate bonds.
      
      .. math::
      
         c_i = \\frac{-N_i C}{f}
    
    """  # noqa: E501, W293

    _settle_accrual: AccrualFunction
    _ytm_accrual: AccrualFunction
    _v1: YtmStubDiscountFunction
    _v2: YtmDiscountFunction
    _v3: YtmStubDiscountFunction
    _c1: CashflowFunction
    _ci: CashflowFunction
    _cn: CashflowFunction

    def __init__(
        self,
        settle_accrual: str | AccrualFunction,
        ytm_accrual: str | AccrualFunction,
        v1: str | YtmStubDiscountFunction,
        v2: str | YtmDiscountFunction,
        v3: str | YtmStubDiscountFunction,
        c1: str | CashflowFunction,
        ci: str | CashflowFunction,
        cn: str | CashflowFunction,
    ):
        self._kwargs: dict[str, str] = {}
        for name, func, _map in zip(
            ["settle_accrual", "ytm_accrual", "v1", "v2", "v3", "c1", "ci", "cn"],
            [settle_accrual, ytm_accrual, v1, v2, v3, c1, ci, cn],
            [
                ACC_FRAC_FUNCS,
                ACC_FRAC_FUNCS,
                V1_FUNCS,
                V2_FUNCS,
                V3_FUNCS,
                C_FUNCS,
                C_FUNCS,
                C_FUNCS,
            ],
            strict=False,
        ):
            if isinstance(func, str):
                setattr(self, f"_{name}", _map[func.lower()])  # type: ignore[index]
                self._kwargs[name] = func
            else:
                setattr(self, f"_{name}", func)
                self._kwargs[name] = "custom"

    @property
    def kwargs(self) -> dict[str, str]:
        """String representation of the parameters for the calculation convention."""
        return self._kwargs


class BillCalcMode:
    """
    Define calculation conventions for :class:`~rateslib.instruments.Bill` type.

    Parameters
    ----------
    price_type: str in {"simple", "discount"}
        The default calculation convention for the rate of the bill.
    ytm_clone_kwargs: dict | str,
        A list of bond keyword arguments, or the ``spec`` for a given bond for which
        a replicable zero coupon bond is constructed and its YTM calculated as comparison.

    Notes
    ------

    - *"simple"*: uses simple interest formula:

      .. math::

         P = \\frac{100}{1+r_{simple}d}

    - *"discount*": uses a discount rate:

      .. math::

         P = 100 ( 1 - r_{discount} d )
    """

    def __init__(
        self,
        price_type: str,
        # price_accrual_type: str,
        # accrual type uses "linear days" by default. This correctly scales ACT365f and ACT360
        # DCF conventions and prepares for any non-standard DCFs.
        # currently no identified cases where anything else is needed. Revise as necessary.
        ytm_clone_kwargs: dict[str, str] | str,
    ):
        self._price_type = price_type
        price_accrual_type = "linear_days"
        self._settle_accrual = ACC_FRAC_FUNCS[price_accrual_type.lower()]
        if isinstance(ytm_clone_kwargs, dict):
            self._ytm_clone_kwargs = ytm_clone_kwargs
        else:
            self._ytm_clone_kwargs = defaults.spec[ytm_clone_kwargs]
        self._kwargs: dict[str, str] = {
            "price_type": price_type,
            "price_accrual_type": price_accrual_type,
            "ytm_clone": "Custom dict" if isinstance(ytm_clone_kwargs, dict) else ytm_clone_kwargs,
        }

    @property
    def kwargs(self) -> dict[str, str]:
        """String representation of the parameters for the calculation convention."""
        return self._kwargs


UK_GB = BondCalcMode(
    # UK government bond conventions
    settle_accrual="linear_days",
    ytm_accrual="linear_days",
    v1="compounding",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

US_GB = BondCalcMode(
    # US Treasury street convention
    settle_accrual="linear_days_long_front_split",
    ytm_accrual="linear_days_long_front_split",
    v1="compounding",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

US_GB_TSY = BondCalcMode(
    # US Treasury treasury convention
    settle_accrual="linear_days_long_front_split",
    ytm_accrual="linear_days_long_front_split",
    v1="simple_long_stub_compounding",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

US_CORP = BondCalcMode(
    # US Corporate bond street convention
    settle_accrual="30u360_forward",
    ytm_accrual="30u360_forward",
    v1="compounding_final_simple",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

US_MUNI = BondCalcMode(
    # US Corporate bond street convention
    settle_accrual="30u360_forward",
    ytm_accrual="30u360_forward",
    v1="compounding_final_simple",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

SE_GB = BondCalcMode(
    # Swedish government bonds
    settle_accrual="30e360_backward",
    ytm_accrual="30e360_backward",
    v1="compounding_final_simple",
    v2="regular",
    v3="simple_30e360",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

CA_GB = BondCalcMode(
    # Canadian government bonds
    settle_accrual="act365f_1y",
    ytm_accrual="linear_days",
    v1="compounding",
    v2="regular",
    v3="simple_30e360",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

DE_GB = BondCalcMode(
    # German government bonds
    settle_accrual="linear_days",
    ytm_accrual="linear_days",
    v1="compounding_final_simple",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

FR_GB = BondCalcMode(
    # French OATs
    settle_accrual="linear_days",
    ytm_accrual="linear_days",
    v1="compounding",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

IT_GB = BondCalcMode(
    # Italian GBs
    settle_accrual="linear_days",
    ytm_accrual="linear_days",
    v1="compounding_final_simple_pay_adjust",
    v2="annual_pay_adjust",
    v3="compounding_pay_adjust",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

NO_GB = BondCalcMode(
    # Norwegian GBs
    settle_accrual="act365f_1y",
    ytm_accrual="act365f_1y",
    v1="compounding_stub_act365f",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

NL_GB = BondCalcMode(
    # Dutch GBs
    settle_accrual="linear_days_long_front_split",
    ytm_accrual="linear_days_long_front_split",
    v1="compounding_final_simple",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

CH_GB = BondCalcMode(
    # Swiss GBs
    settle_accrual="30e360_backward",
    ytm_accrual="30e360_backward",
    v1="compounding",
    v2="regular",
    v3="compounding",
    c1="cashflow",
    ci="cashflow",
    cn="cashflow",
)

UK_GBB = BillCalcMode(
    # UK T-bills
    price_type="simple",
    # price_accrual_type="linear_days",
    ytm_clone_kwargs="uk_gb",
)

US_GBB = BillCalcMode(
    # US T-bills
    price_type="discount",
    # price_accrual_type="linear_days",
    ytm_clone_kwargs="us_gb",
)

SE_GBB = BillCalcMode(
    # Swedish T-bills
    price_type="simple",
    # price_accrual_type="linear_days",
    ytm_clone_kwargs="se_gb",
)

BOND_MODE_MAP = {
    "uk_gb": UK_GB,
    "us_gb": US_GB,
    "de_gb": DE_GB,
    "fr_gb": FR_GB,
    "nl_gb": NL_GB,
    "ch_gb": CH_GB,
    "no_gb": NO_GB,
    "se_gb": SE_GB,
    "us_gb_tsy": US_GB_TSY,
    "us_corp": US_CORP,
    "us_muni": US_MUNI,
    "it_gb": IT_GB,
    "ca_gb": CA_GB,
    # aliases
    "ukg": UK_GB,
    "cadgb": CA_GB,
    "ust": US_GB,
    "ust_31bii": US_GB_TSY,
    "sgb": SE_GB,
}

BILL_MODE_MAP = {
    "uk_gbb": UK_GBB,
    "us_gbb": US_GBB,
    "se_gbb": SE_GBB,
    # aliases
    "ustb": US_GBB,
    "uktb": UK_GBB,
    "sgbb": SE_GBB,
}


def _get_bond_calc_mode(calc_mode: str | BondCalcMode) -> BondCalcMode:
    if isinstance(calc_mode, str):
        return BOND_MODE_MAP[calc_mode.lower()]
    return calc_mode


def _get_calc_mode_for_class(
    obj: Security, calc_mode: str | BondCalcMode | BillCalcMode
) -> BondCalcMode | BillCalcMode:
    if isinstance(calc_mode, str):
        map_: dict[str, dict[str, BondCalcMode] | dict[str, BillCalcMode]] = {
            "FixedRateBond": BOND_MODE_MAP,
            "Bill": BILL_MODE_MAP,
            "FloatRateNote": BOND_MODE_MAP,
            "IndexFixedRateBond": BOND_MODE_MAP,
        }
        klass: str = type(obj).__name__
        return map_[klass][calc_mode.lower()]
    return calc_mode
