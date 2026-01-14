#############################################################
# COPYRIGHT 2022 Siffrorna Technology Limited
# This code may not be copied, modified, used or distributed
# except with the express permission and licence to
# do so, provided by the copyright holder.
# See: https://rateslib.com/py/en/latest/i_licence.html
#############################################################

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from pytz import UTC

from rateslib import defaults
from rateslib.curves import (
    _BaseCurve,
    _CurveMeta,
    _CurveNodes,
    _CurveType,
    _WithMutability,
)
from rateslib.dual import dual_exp, dual_log
from rateslib.enums.generics import NoInput, _drb
from rateslib.mutability import _new_state_post
from rateslib.scheduling import Convention, get_calendar
from rateslib.scheduling.convention import _get_convention

if TYPE_CHECKING:
    from numpy import float64 as Nf64  # noqa: N812
    from numpy import object_ as Nobject  # noqa: N812
    from numpy.typing import NDArray

    from rateslib.typing import (  # pragma: no cover
        CalInput,
        DualTypes,
        Variable,
        datetime,
        float_,
        int_,
        str_,
    )


class _NullInterpolator:
    def _csolve(self, curve_type: _CurveType, nodes: _CurveNodes, ad: int) -> None:
        pass


def _dual_sinh(x: DualTypes) -> DualTypes:
    return (dual_exp(x) - dual_exp(-x)) * 0.5


class SmithWilsonCurve(_WithMutability, _BaseCurve):
    r"""
    A Smith-Wilson style *Curve* defined by discount factors.

    The discount factors on this curve are defined by:

    .. math::

       v(t) = e^{-wt} + \mathbf{W}[t, \mathbf{u}] \mathbf{\hat{b}}

    where,

    .. math::

       W(t, u) &= e^{-w(t+u)} \left ( \alpha \min(t, u) - e^{\alpha max(t, u)} sinh(\alpha min(t, u)) \right )  \\
       w &= \ln ( 1 + UFR)

    and :math:`\alpha` and :math:`UFR` are parameters controlling convergence to some rate in the
    long term, and :math:`\mathbf{\hat{b}}` are calibration parameters. All 'time' quantities are
    derived under an effective '*Act/365.25*' day count convention.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    nodes: dict[datetime, float]
        The parameters of the *Curve*. The value associated with the *initial node date* is
        treated as :math:`\alpha`. All subsequent key-value pairs define the (Mx1) vectors
        :math:`\mathbf{u}` and :math:`\mathbf{\hat{b}}` respectively.
    ufr: float, :red:`required`
        The rates that is denoted by the *'ultimate forward rate'*.
    solve_alpha: bool, :green:`optional (set as False)`
        Define whether :math:`\alpha` is to be treated as a parameter in the solver process
        simultaneously with :math:`\mathbf{\hat{b}}`.
    id : str, :green:`optional (set randomly)`
        The unique identifier to distinguish between curves in a multicurve framework.
    convention : Convention, str, :green:`optional (set as Act365_25)`
        The convention of the curve for determining rates. Please see
        :meth:`dcf()<rateslib.scheduling.dcf>` for all available options.
    modifier : str, :green:`optional (set by 'defaults')`
        The modification rule, in {"F", "MF", "P", "MP"}, for determining rates when input as
        a tenor, e.g. "3M".
    calendar : calendar, str, :green:`optional (set as 'all')`
        The holiday calendar object to use. If str, looks up named calendar from
        static data. Used for determining rates.
    ad : int in {0, 1, 2}, :green:`optional`
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.
    index_base: float, :green:`optional`
        The initial index value at the initial node date of the curve. Used for
        forecasting future index values.
    index_lag : int, :green:`optional (set by 'defaults')`
        Number of months of by which the index lags the date. For example if the initial
        curve node date is 1st Sep 2021 based on the inflation index published
        17th June 2023 then the lag is 3 months. Best practice is to use 0 months.
    collateral : str, :green:`optional (set as None)`
        A currency identifier to denote the collateral currency against which the discount factors
        for this *Curve* are measured.
    credit_discretization : int, :green:`optional (set by 'defaults')`
        A parameter for numerically solving the integral for credit protection legs and default
        events. Expressed in calendar days. Only used by *Curves* functioning as *hazard Curves*.
    credit_recovery_rate : Variable | float, :green:`optional (set by 'defaults')`
        A parameter used in pricing credit protection legs and default events.

    Notes
    -----

    **EIOPA's Approach**

    The Smith-Wilson *Curve* as `defined by EIOPA <https://www.eiopa.europa.eu/system/files/2022-09/eiopa-bos-22-409-technical-documentation.pdf>`__
    is a *Curve* designed with the following properties:

    - A matrix-type formulation to solve calibration parameters using linear algebra.
    - An *'ultra-forward-rate (UFR)'* and convergence parameter :math:`\alpha` to control
      the curve beyond points at which there might be priced market instruments.

    The official version of the Smith-Wilson discount factor function is:

    .. math::

       v(t) = e^{-wt} + \mathbf{W}[t, \mathbf{u}]\mathbf{C} \mathbf{b}

    In this equation a set of *N* bonds (likely coupon bearing) are selected from the market and
    the vector :math:`\mathbf{u}`, of length *M*, contains ordered times to each cashflow of any
    bond. The *(MxN)* matrix :math:`\mathbf{C}_{i,j}` structures individual cashflows attributable
    to each bond, *j*, at cashflow date, :math:`u_i`. And :math:`\mathbf{b}`, the calibration
    parameters, must have length *N*.

    The Smith-Wilson concept is to use that same equation replacing, *t*, with each :math:`u_i`,
    and then multiplying each cashflow of any bond by those relevant discount factors to return the
    market price, :math:`\mathbf{p}`, i.e.

    .. math::

       \mathbf{p} = \mathbf{C^T} v[\mathbf{u}] = \mathbf{C^T} e^{-w\mathbf{u}} + \mathbf{C^T W[u,u] C b}

    After this is rearranged it yields,

    .. math::

       \mathbf{b} = \left ( \mathbf{C^T W[u,u] C} \right )^{-1} ( \mathbf{p} - \mathbf{C^T} e^{-w \mathbf{u}} )

    which is transformable into the equations recognisable in the EIOPA document using their same
    substitutions,

    .. math::

       \mathbf{b} &= \left ( \mathbf{Q^T H[u,u] Q} \right )^{-1} ( \mathbf{p} - \mathbf{q} )  \\
       \mathbf{d} &= e^{-w \mathbf{u}} \\
       \mathbf{Q} &= \mathbf{d_\Delta C} \\
       \mathbf{W[u,u]} &= \mathbf{d_\Delta H[u,u] d_\Delta} \\
       \mathbf{q} &= \mathbf{C^T d} \\

    **Rateslib's Approach**

    *Rateslib* makes two key changes. Firstly it recognises that for an unchanged :math:`\mathbf{u}`
    vector, i.e. the cashflow dates remain the same, and unchanged discount factors at those dates,
    i.e. unchanged :math:`v[\mathbf{u}]` the system can be equivalently formulated
    in terms of zero coupon bonds, so that:

    .. math::

       \underbrace{\mathbf{C b}}_{(M \times N) (N \times 1)} = \underbrace{\mathbf{I \hat{b}}}_{(M \times M) (M \times 1)}

    Since the market prices of the bonds are known and the discount factors of these synthesised
    zero coupon bonds are not known apriori this transformation does not allow the linear
    algebraic solution (EIOPA's approach) to remain viable. That leads to the second change.

    *Rateslib* does not bootstrap or algebraically solve *Curves*. It uses a global solver.
    This is why the above change is permissible because even under the
    reformulation it will still converge on *a* solution for :math:`\mathbf{\hat{b}}` which
    reprices the bonds.

    **Implication**

    The general rules for *Curve* solving remain applicable; if M > N then the system is
    underspecified and may result in spurious behavior. If M = N and maturities are all
    appropriately chosen the solution is exact and unique.

    Because *rateslib* treats *Curve* parameterization and *Instrument* calibration as two
    separate processes there is increased flexibility in both aspects. The calibrating bonds
    do not necessarily have to match the *nodes* of the Smith-Wilson *Curve*. Under EIOPA's
    approach this is obviously not possible because the framework of equations relies on
    setting up the appropriate cashflow matrix and array of cashflow dates.

    .. note::

       *Rateslib* will not determine the matrices :math:`\mathbf{W[u,u], H[u,u], Q, C}` etc.
       becuase its methods does not require them

    Examples
    --------
    The `standard EIOPA example <https://register.eiopa.europa.eu/Publications/Consultations/Consultation_RFR_Example_Extrapolation.xlsx>`__
    happens to include a 20x20 cashflow matrix, each bond valued at par with increasing coupon
    rates, implying increasing YTM.

    .. image:: ../_static/eiopa_c.png
       :align: center
       :alt: EIOPA Example of Smith-Wilson Curve
       :height: 304
       :width: 597

    Because this is a square matrix and satisfies the criteria above the *rateslib* solution
    will match EIOPA's.

    .. ipython:: python
       :suppress:

       from rateslib import FixedRateBond, Solver, SmithWilsonCurve, dt

    .. ipython:: python

        sw = SmithWilsonCurve(
            nodes={
                dt(2000, 1, 1): 0.12376,       #  <--  alpha value used in EIOPA file
                **{dt(2000+i, 1, 1): 0.1 for i in range(1, 21)}
            },
            solve_alpha=False,
            ufr= 4.2,
            id="academic_curve",
        )
        coupons = [0.2, 0.225, 0.3, 0.425, 0.55, 0.7, 0.85, 1.0, 1.15, 1.275, 1.4, 1.475, 1.575, 1.65, 1.7, 1.75, 1.8, 1.825, 1.85, 1.875]
        bonds = [
            FixedRateBond(
                effective=dt(2000, 1, 1),
                termination=f"{i}Y",         #  <-  1Y to 20Y
                fixed_rate=coupons[i-1],     #  <-  Coupons as specified
                calendar="all",
                ex_div=1,
                convention="actacticma",
                frequency="A",
                curves="academic_curve",
                metric="dirty_price"
            )
            for i in range(1, 21)
        ]
        prices = [100.0] * 20                #  <-  All bonds priced to par
        Solver(curves=[sw], instruments=bonds, s=prices)

    We can plot the resultant curves, which can be compared directly with the EIOPA file.

    .. ipython:: python

       sw.plot("Z")
       sw.plot("1b")

    .. plot::

       from rateslib import SmithWilsonCurve, Solver, dt, FixedRateBond
       import matplotlib.pyplot as plt

       sw = SmithWilsonCurve(
           nodes={
               dt(2000, 1, 1): 0.12376,
               **{dt(2000+i, 1, 1): 0.1 for i in range(1, 21)}
           },
           solve_alpha=False,
           ufr= 4.2,
           id="academic_curve",
       )
       coupons = [0.2, 0.225, 0.3, 0.425, 0.55, 0.7, 0.85, 1.0, 1.15, 1.275, 1.4, 1.475, 1.575, 1.65, 1.7, 1.75, 1.8, 1.825, 1.85, 1.875]
       bonds = [
           FixedRateBond(
               effective=dt(2000, 1, 1),
               termination=f"{i}Y",         #  <-  1Y to 20Y
               fixed_rate=coupons[i-1],     #  <-  Coupons as specified
               calendar="all",
               ex_div=1,
               convention="actacticma",
               frequency="A",
               curves="academic_curve",
               metric="dirty_price"
           )
           for i in range(1, 21)
       ]
       prices = [100.0] * 20                #  <-  All bonds priced to par
       Solver(curves=[sw], instruments=bonds, s=prices)

       fig1, ax1, lines = sw.plot("z")
       del fig1, ax1
       plt.close()
       fig, ax, _ = sw.plot("1b")
       ax.plot(lines[0]._x, lines[0]._y)
       plt.show()
       plt.close()

    """  # noqa: E501

    # ABC properties

    _ini_solve = 0
    _base_type: _CurveType = _CurveType.dfs
    _id: str = None  # type: ignore[assignment]
    _ad: int = None  # type: ignore[assignment]
    _meta: _CurveMeta = None  # type: ignore[assignment]
    _nodes: _CurveNodes = None  # type: ignore[assignment]
    _interpolator = _NullInterpolator()  # type: ignore[assignment]

    @_new_state_post
    def __init__(
        self,
        nodes: dict[datetime, DualTypes],
        ufr: DualTypes,
        solve_alpha: bool = False,
        id: str_ = NoInput(0),  # noqa: A002
        *,
        convention: Convention | str | NoInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        ad: int = 0,
        index_base: Variable | float_ = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        collateral: str_ = NoInput(0),
        credit_discretization: int_ = NoInput(0),
        credit_recovery_rate: Variable | float_ = NoInput(0),
    ):
        self._nodes = _CurveNodes(_nodes=nodes)
        if not solve_alpha:
            self._ini_solve = 1

        self._ufr = ufr

        self._meta = _CurveMeta(
            _calendar=get_calendar(calendar),
            _convention=_get_convention(_drb(Convention.Act365_25, convention)),
            _modifier=_drb(defaults.modifier, modifier).upper(),
            _index_base=index_base,
            _index_lag=_drb(defaults.index_lag_curve, index_lag),
            _collateral=_drb(None, collateral),
            _credit_discretization=_drb(
                defaults.cds_protection_discretization, credit_discretization
            ),
            _credit_recovery_rate=_drb(defaults.cds_recovery_rate, credit_recovery_rate),
        )

        self._id = _drb(uuid4().hex[:5], id)  # 1 in a million clash
        self._set_ad_order(order=ad)  # will also clear and initialise the cache

    @property
    def alpha(self) -> DualTypes:
        r"""The :math:`\alpha` value of the *Curve*."""
        return self.nodes.nodes[self.nodes.initial]

    @property
    def b(self) -> NDArray[Nobject]:
        r"""The :math:`\mathbf{\hat{b}}` parameters value of the *Curve*."""
        return np.array(self.nodes.values[1:])

    @property
    def ufr(self) -> DualTypes:
        """The UFR value of the *Curve*."""
        return self._ufr

    @property
    def k(self) -> DualTypes:
        r"""
        The :math:`\kappa` value as defined in the EIOPA documentation.

        Under EIOPA:

        .. math::

           \kappa = \frac{ 1 + \alpha \mathbf{u^T Q b} }{ sinh[\alpha \mathbf{u^T}] \mathbf{Q b} }

        """
        Q = np.diag([dual_exp(-self.w * _) for _ in self.u])  # Q is d_delta
        numerator: DualTypes = 1 + self.alpha * np.matmul(
            np.matmul(self.u[None, :], Q), self.b[:, None]
        )
        denominator: DualTypes = np.matmul(
            np.matmul(np.array([_dual_sinh(self.alpha * _) for _ in self.u])[None, :], Q),
            self.b[:, None],
        )
        return numerator / denominator

    @cached_property
    def w(self) -> DualTypes:
        """The :math:`w` value of the *Curve* derived from the UFR."""
        return dual_log(1 + self.ufr / 100.0)

    @cached_property
    def u(self) -> NDArray[Nf64]:
        r"""The :math:`\mathbf{u}` vector of the *Curve* derived from the node dates."""
        # 31557600 = 365.25 days * 86400 seconds per day
        return (np.array(self.nodes.posix_keys[1:]) - self.nodes.posix_keys[0]) / 31557600.0

    def __getitem__(self, date: datetime) -> DualTypes:
        if defaults.curve_caching and date in self._cache:
            return self._cache[date]

        if date < self.nodes.initial:
            return 0.0
        elif date == self.nodes.initial:
            return 1.0

        # 31557600 = 365.25 days * 86400 seconds per day
        t = (date.replace(tzinfo=UTC).timestamp() - self.nodes.posix_keys[0]) / 31557600.0
        a = self.alpha
        w = self.w

        v = dual_exp(-t * w)

        mins = [min(t, _) for _ in self.u]
        maxs = [max(t, _) for _ in self.u]
        ww = np.array(
            [
                dual_exp(-u * w) * (a * min_ - dual_exp(-a * max_) * _dual_sinh(a * min_))
                for (u, min_, max_) in zip(self.u, mins, maxs, strict=False)
            ]
        )

        v += np.inner(self.b, ww) * v
        return self._cached_value(date, v)
