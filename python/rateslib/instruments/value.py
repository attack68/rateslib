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

from typing import TYPE_CHECKING, NoReturn

from rateslib.curves.utils import _CurveType
from rateslib.dual.utils import dual_log
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import IndexMethod
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
)
from rateslib.periods.utils import _validate_base_curve
from rateslib.scheduling import dcf

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        datetime,
        datetime_,
        str_,
    )


class Value(_BaseInstrument):
    """
    A pseudo *Instrument* used to calibrate a *Curve* within a :class:`~rateslib.solver.Solver`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Value
       from datetime import datetime as dt
       from rateslib import Curve, Solver

    The below :class:`~rateslib.curves.Curve` is solved directly
    from a calibrating DF value on 1st Nov 2022.

    .. ipython:: python

       val = Value(dt(2022, 11, 1), curves=["v"], metric="curve_value")
       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="v")
       solver = Solver(curves=[curve], instruments=[val], s=[0.99])
       curve[dt(2022, 11, 1)]

    .. rubric:: Pricing

    A *Value* requires, and will calibrate, just one *Curve*. This *Curve*, appropriating
    a *rate curve* or an *index curve*, is dependent upon the ``metric``.
    Allowable inputs are:

    .. code-block:: python

       curves = curve | [curve]           #  a single curve is repeated for all required curves
       curves = {"rate_curve": rate_curve} | {"index_curve": index_curve}  # dict form is explicit

    The various *rate* ``metric`` that can be calculated for a *Curve* are as follows;

    - *'curve_value'*: returns the discount factor or a value from a DF-based or value-based
      *rate curve*.
    - *'index_value'*: returns a daily interpolated index value using an index lag derived from the
      *index curve*.
    - *'cc_zero_rate'*: returns a continuously compounded zero rate to the provided *effective*
      date from a DF based *rate curve*.
    - *'o/n_rate'*: returns a 1 calendar day rate starting on the effective date with the provided
      *convention* from a *rate curve*.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    effective : datetime, :red:`required`
        The datetime index for which the `rate`, which is just the curve value, is
        returned.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    metric : str, :green:`optional` (set as 'curve_value')
        The pricing metric returned by :meth:`~rateslib.instruments.Value.rate`. See
        **Pricing**.

    """

    _rate_scalars = {
        "curve_value": 100.0,
        "index_value": 100.0,
        "cc_zero_rate": 1.0,
        "o/n_rate": 1.0,
    }

    def __init__(
        self,
        effective: datetime,
        *,
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> None:
        user_args = dict(
            effective=effective,
            curves=self._parse_curves(curves),
            metric=metric,
        )
        default_args = dict(metric="curve_value")
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=user_args,
            default_args=default_args,
            meta_args=["curves", "metric"],
        )

        self._rate_scalar = self._rate_scalars.get(self.kwargs.meta["metric"], 1.0)

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        A Value requires only one 1 curve, which is set as all element values
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
            )
        elif isinstance(curves, list | tuple):
            if len(curves) != 1:
                raise ValueError(
                    f"{type(self).__name__} requires only 1 curve types. Got {len(curves)}."
                )
            else:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[0],
                    index_curve=curves[0],
                )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input
            return _Curves(
                rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
                index_curve=curves,  # type: ignore[arg-type]
            )

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        effective: datetime = self.kwargs.leg1["effective"]
        _curves = self._parse_curves(curves)
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "curve_value":
            curve = _validate_base_curve(
                _maybe_get_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "rate_curve", solver
                )
            )
            ret: DualTypes = curve[effective]

        elif metric_ == "cc_zero_rate":
            curve = _validate_base_curve(
                _maybe_get_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "rate_curve", solver
                )
            )
            if curve._base_type != _CurveType.dfs:
                raise TypeError(
                    "`curve` used with `metric`='cc_zero_rate' must be discount factor based.",
                )
            dcf_ = dcf(start=curve.nodes.initial, end=effective, convention=curve.meta.convention)
            ret = (dual_log(curve[effective]) / -dcf_) * 100

        elif metric_ == "index_value":
            curve = _validate_base_curve(
                _maybe_get_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "index_curve", solver
                )
            )
            ret = curve.index_value(
                index_date=effective,
                index_lag=curve.meta.index_lag,
                index_method=IndexMethod.Daily,
            )

        elif metric_ == "o/n_rate":
            curve = _validate_base_curve(
                _maybe_get_curve_maybe_from_solver(
                    self.kwargs.meta["curves"], _curves, "rate_curve", solver
                )
            )
            ret = curve.rate(effective, "1D")  # type: ignore[assignment]

        else:
            raise ValueError(
                "`metric`must be in {'curve_value', 'cc_zero_rate', 'index_value', 'o/n_rate'}."
            )
        return ret

    def npv(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`Value` instrument has no concept of NPV.")

    def cashflows(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`Value` instrument has no concept of cashflows.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`Value` instrument has no concept of analytic delta.")
