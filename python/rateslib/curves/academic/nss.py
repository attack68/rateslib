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

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from rateslib import defaults
from rateslib.curves import _BaseCurve, _CurveMeta, _CurveNodes, _CurveType, _WithMutability
from rateslib.dual import Dual, Dual2, dual_exp, set_order_convert
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.mutability import _clear_cache_post, _new_state_post
from rateslib.scheduling import Convention, dcf, get_calendar
from rateslib.scheduling.convention import _get_convention

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        DualTypes,
        Variable,
        datetime,
        float_,
        int_,
        str_,
    )


class NelsonSiegelSvenssonCurve(_WithMutability, _BaseCurve):
    r"""
    A Nelson-Siegel-Svensson curve defined by discount factors.

    The continuously compounded rate to maturity, :math:`r(T)`, is given by the following
    equation of **six** parameters, :math:`[\beta_0, \beta_1, \beta_2, \lambda_0, \beta_3, \lambda_1]`

    .. math::

       r(T) = \begin{bmatrix} \beta_0 & \beta_1 & \beta_2 & \beta_3 \end{bmatrix} \begin{bmatrix} 1 \\ \lambda_0 (1- e^{-T/ \lambda_0}) / T \\ \lambda_0 (1- e^{-T/ \lambda_0})/ T - e^{-T/ \lambda_0} \\ \lambda_1 (1- e^{-T/ \lambda_1})/ T - e^{-T/ \lambda_1} \end{bmatrix}

    The **discount factors** on that curve equaling:

    .. math::

       v(T) = e^{-T r(T)}

    *T* is determined as the day count fraction between the start of the curve and the maturity
    under the given the ``convention`` and ``calendar``.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    dates: 2-tuple of datetime, :red:`required`
        The dates defining the eval date and final date of the *Curve*.
    parameters: 6-tuple of Dual, Dual2, Variable, float, :red:`required`
        The parameters associated with the *Curve*. In order these are
        :math:`[\beta_0, \beta_1, \beta_2, \lambda_0, \beta_3, \lambda_1]`.
    id : str, :green:`optional (set randomly)`
        The unique identifier to distinguish between curves in a multicurve framework.
    convention : Convention, str, :green:`optional (set as ActActISDA)`
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

    """  # noqa: E501

    # ABC properties

    _ini_solve = 0
    _base_type = _CurveType.dfs
    _id = None  # type: ignore[assignment]
    _meta = None  # type: ignore[assignment]
    _nodes = None  # type: ignore[assignment]
    _ad = None  # type: ignore[assignment]
    _interpolator = None  # type: ignore[assignment]
    _n = 6

    @_new_state_post
    def __init__(
        self,
        dates: tuple[datetime, datetime],
        parameters: tuple[DualTypes, DualTypes, DualTypes, DualTypes, DualTypes, DualTypes],
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
        self._nodes = _CurveNodes({dates[0]: 0.0, dates[1]: 0.0})
        self._params = parameters
        self._meta = _CurveMeta(
            _calendar=get_calendar(calendar),
            _convention=_get_convention(_drb(Convention.ActActISDA, convention)),
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
    def params(self) -> tuple[DualTypes, DualTypes, DualTypes, DualTypes, DualTypes, DualTypes]:
        r"""
        The parameters associated with the *Curve*.
        In order these are :math:`[\beta_0, \beta_1, \beta_2, \lambda_0, \beta_3, \lambda_1]`.
        """
        return self._params

    def __getitem__(self, date: datetime) -> DualTypes:
        if defaults.curve_caching and date in self._cache:
            return self._cache[date]

        if date < self.nodes.initial:
            return 0.0
        elif date == self.nodes.initial:
            return 1.0
        b0, b1, b2, l0, b3, l1 = self._params
        T = dcf(
            self.nodes.initial, date, convention=self.meta.convention, calendar=self.meta.calendar
        )
        a1 = l0 * (1 - dual_exp(-T / l0)) / T
        a2 = a1 - dual_exp(-T / l0)
        x1 = l1 * (1 - dual_exp(-T / l1)) / T
        x2 = x1 - dual_exp(-T / l1)
        r = b0 + a1 * b1 + a2 * b2 + x2 * b3

        return self._cached_value(date, dual_exp(-T * r))

    # Solver mutability methods

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
        return np.array(self._params)

    def _get_node_vars(self) -> tuple[str, ...]:
        return tuple(f"{self._id}{i}" for i in range(self._ini_solve, self._n))

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(self, vector: list[DualTypes], ad: int) -> None:
        if ad == 0:
            self._params = tuple(_dual_float(_) for _ in vector)  # type: ignore[assignment]
        elif ad == 1:
            self._params = tuple(  # type: ignore[assignment]
                Dual(_dual_float(_), [f"{self._id}{i}"], []) for i, _ in enumerate(vector)
            )
        else:  # ad == 2
            self._params = tuple(  # type: ignore[assignment]
                Dual2(_dual_float(_), [f"{self._id}{i}"], [], []) for i, _ in enumerate(vector)
            )

    @_clear_cache_post
    def _set_ad_order(self, order: int) -> None:
        if self.ad == order:
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order
        self._params = tuple(  # type: ignore[assignment]
            set_order_convert(_, order, [f"{self._id}{i}"]) for i, _ in enumerate(self.params)
        )
