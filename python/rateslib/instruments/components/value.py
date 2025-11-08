from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from rateslib import defaults
from rateslib.curves._parsers import _CurveType
from rateslib.dual.utils import dual_log
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import IndexMethod
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.kwargs import _KWArgs
from rateslib.instruments.components.protocols.pricing import (
    _Curves,
    _get_maybe_curve_maybe_from_solver,
)
from rateslib.scheduling import dcf

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        Convention,
        Curves_,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Solver_,
        datetime,
        datetime_,
        str_,
    )


class Value(_BaseInstrument):
    """
    A null *Instrument* which can be used within a :class:`~rateslib.solver.Solver`
    to directly parametrise a *Curve* node, via some calculated value.

    Parameters
    ----------
    effective : datetime
        The datetime index for which the `rate`, which is just the curve value, is
        returned.
    curves : Curve, LineCurve, str or list of such, optional
        A single :class:`~rateslib.curves.Curve`,
        :class:`~rateslib.curves.LineCurve` or id or a
        list of such. Only uses the first *Curve* in a list.
    convention : str, optional,
        Day count convention used with certain ``metric``.
    metric : str in {"curve_value", "index_value", "cc_zero_rate", "o/n_rate"}, optional
        Configures which value to extract from the *Curve*.

    Examples
    --------
    The below :class:`~rateslib.curves.Curve` is solved directly
    from a calibrating DF value on 1st Nov 2022.

    .. ipython:: python
       :suppress:

       from rateslib import Value

    .. ipython:: python

       curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="v")
       instruments = [(Value(dt(2022, 11, 1)), (curve,), {})]
       solver = Solver([curve], [], instruments, [0.99])
       curve[dt(2022, 1, 1)]
       curve[dt(2022, 11, 1)]
       curve[dt(2023, 1, 1)]
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
        convention: Convention | str_ = NoInput(0),
        metric: str_ = NoInput(0),
        curves: Curves_ = NoInput(0),
    ) -> None:
        user_args = dict(
            effective=effective,
            convention=convention,
            curves=self._parse_curves(curves),
            metric=metric,
        )
        default_args = dict(convention=defaults.convention, metric="curve_value")
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=user_args,
            default_args=default_args,
            meta_args=["curves", "metric"],
        )

        self._rate_scalar = self._rate_scalars.get(self.kwargs.meta["metric"], 1.0)

    def _parse_curves(self, curves: Curves_) -> _Curves:
        """
        A Value requires only one 1 curve, which is set as all element values
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            raise ValueError("`curves` supplied to a Value should be a single _BaseCurve object.")
        elif isinstance(curves, list | tuple):
            if len(curves) != 1:
                raise ValueError(
                    "`curves` supplied to a Value should be a single _BaseCurve object."
                )
            else:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[0],
                    index_curve=curves[0],
                )
        else:  # `curves` is just a single input
            return _Curves(
                rate_curve=curves,
                disc_curve=curves,
                index_curve=curves,
            )

    def rate(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        effective: datetime = self.kwargs.leg1["effective"]
        convention: Convention | str = self.kwargs.leg1["convention"]
        _curves = self._parse_curves(curves)
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "curve_value":
            curve = _get_maybe_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "rate_curve", solver
            )
            ret: DualTypes = curve[effective]

        elif metric_ == "cc_zero_rate":
            curve = _get_maybe_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "rate_curve", solver
            )
            if curve._base_type != _CurveType.dfs:
                raise TypeError(
                    "`curve` used with `metric`='cc_zero_rate' must be discount factor based.",
                )
            dcf_ = dcf(start=curve.nodes.initial, end=effective, convention=convention)
            ret = (dual_log(curve[effective]) / -dcf_) * 100

        elif metric_ == "index_value":
            curve = _get_maybe_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "index_curve", solver
            )
            ret = curve.index_value(
                index_date=effective,
                index_lag=curve.meta.index_lag,
                index_method=IndexMethod.Daily,
            )

        elif metric_ == "o/n_rate":
            curve = _get_maybe_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "rate_curve", solver
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
