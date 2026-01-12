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

from functools import partial
from typing import TYPE_CHECKING, Protocol

from rateslib import defaults
from rateslib.curves._parsers import (
    _maybe_set_ad_order,
    _validate_obj_not_no_input,
)
from rateslib.dual import ift_1dim
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.protocols import _WithAccrued
from rateslib.instruments.protocols.pricing import (
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        Solver_,
        VolT_,
        _BaseCurve,
        _BaseCurveOrDict_,
        _Curves,
        datetime_,
        float_,
        str_,
    )


class _WithOASpread(_WithAccrued, Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    def _parse_curves(self, curves: CurvesT_) -> _Curves: ...

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
    ) -> DualTypes: ...

    def oaspread(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        price: DualTypes_ = NoInput(0),
        metric: str_ = NoInput(0),
        func_tol: float_ = NoInput(0),
        conv_tol: float_ = NoInput(0),
    ) -> DualTypes:
        """
        The option adjusted spread added to the discounting *Curve* to value the security
        at ``price``.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        price : float, Dual, Dual2
            The price of the bond to match.
        metric : str, optional
            The metric to use when evaluating the price/rate of the instrument. If not
            given uses the instrument's :meth:`~rateslib.instruments.FixedRateBond.rate` method
            default.
        func_tol: float, optional
            The tolerance for the objective function value when iteratively solving. If not given
            uses `defaults.oaspread_func_tol`.
        conv_tol: float, optional
            The tolerance used for stopping criteria of successive iteration values. If not
            given uses `defaults.oaspread_conv_tol`.

        Returns
        -------
        float, Dual, Dual2

        Notes
        ------
        The discount curve must be of type :class:`~rateslib.curves._BaseCurve` with a
        provided :meth:`~rateslib.curves._BaseCurve.shift` method available.

        .. warning::
           The sensitivity of variables is preserved for the input argument ``price``, but this
           function does **not** preserve AD towards variables associated with the ``curves`` or
           ``solver``.

        Examples
        --------

        .. ipython:: python
           :suppress:

           from rateslib import Variable

        .. ipython:: python

           bond = FixedRateBond(dt(2000, 1, 1), "3Y", fixed_rate=2.5, spec="us_gb")
           curve = Curve({dt(2000, 7, 1): 1.0, dt(2005, 7, 1): 0.80})
           # Add AD variables to the curve without a Solver
           curve._set_ad_order(1)

           bond.oaspread(curves=curve, price=Variable(95.0, ["price"], []))

        This result excludes curve sensitivities but includes sensitivity to the
        constructed *'price'* variable. Accuracy can be observed through numerical simulation.

        .. ipython:: python

           bond.oaspread(curves=curve, price=96.0)
           bond.oaspread(curves=curve, price=94.0)

        """
        if isinstance(price, NoInput):
            raise ValueError("`price` must be supplied in order to derive the `oaspread`.")

        _curves = self._parse_curves(curves)

        def s_with_args(
            g: DualTypes, curve: _BaseCurveOrDict_, disc_curve: _BaseCurve, metric: str_
        ) -> DualTypes:
            """
            Return the price of a bond given an OASpread.

            Parameters
            ----------
            g: DualTypes
                The OASpread value in basis points.
            curve:
                The forecasting curve.
            disc_curve:
                The discount curve.

            Returns
            -------
            DualTypes
            """
            _shifted_discount_curve = disc_curve.shift(g)
            return self.rate(curves=[curve, _shifted_discount_curve], metric=metric)  # type: ignore[list-item]

        disc_curve_ = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                self.kwargs.meta["curves"], _curves, "disc_curve", solver
            ),
            "disc_curve",
        )

        _ad_disc = _maybe_set_ad_order(disc_curve_, 0)
        rate_curve_ = _maybe_get_curve_or_dict_maybe_from_solver(
            self.kwargs.meta["curves"], _curves, "rate_curve", solver
        )

        _ad_fore = _maybe_set_ad_order(rate_curve_, 0)

        s = partial(
            s_with_args,
            curve=rate_curve_,
            disc_curve=disc_curve_,
            metric=metric,
        )

        result = ift_1dim(
            s,
            price,
            "ytm_quadratic",
            (-300, 200, 1200),
            func_tol=_drb(defaults.oaspread_func_tol, func_tol),
            conv_tol=_drb(defaults.oaspread_conv_tol, conv_tol),
        )

        _maybe_set_ad_order(disc_curve_, _ad_disc)
        _maybe_set_ad_order(rate_curve_, _ad_fore)
        ret: DualTypes = result["g"]
        return ret
