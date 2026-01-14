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

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _get_fx_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _WithPricingObjs,
)
from rateslib.periods.utils import _maybe_fx_converted

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        _Curves,
        _SettlementParams,
        _Vol,
        datetime_,
        str_,
    )


class _WithNPV(_WithPricingObjs, Protocol):
    """
    Protocol to establish value of any *Instrument* type.
    """

    _kwargs: _KWArgs

    @property
    def settlement_params(self) -> _SettlementParams:
        """
        The default :class:`~rateslib.periods.parameters._SettlementParams` of the *Instrument*.

        This is used to define a ``base`` currency when one is not specified.
        """
        if hasattr(self, "legs"):
            return self.legs[0].settlement_params  # type: ignore[no-any-return]
        elif hasattr(self, "instruments"):
            return self.instruments[0].settlement_params  # type: ignore[no-any-return]
        else:
            raise NotImplementedError(
                f"`settlement_params` not implemented for type {type(self).__name__}"
            )

    @property
    def kwargs(self) -> _KWArgs:
        """The :class:`~rateslib.instruments.protocols._KWArgs` container for
        the *Instrument*."""
        return self._kwargs

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Calculate the NPV of the *Instrument* converted to any other *base* accounting currency.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import dt, Curve, IRS

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.75})
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", fixed_rate=1.0, curves=[curve])
           irs.npv()
           irs.npv(local=True)

        .. role:: red

        .. role:: green

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        local: bool, :green:`optional (set as False)`
            An override flag to return a dict of NPV values indexed by string currency.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        float, Dual, Dual2, Variable or dict of such indexed by string currency.

        Notes
        -----
        If ``base`` is not given then this function will return the value obtained from
        determining the PV in local *settlement currency*.

        If ``base`` is provided this then an :class:`~rateslib.fx.FXForwards` object may be
        required to perform conversions. An :class:`~rateslib.fx.FXRates` object is also allowed
        for this conversion although best practice does not recommend it due to possible
        settlement date conflicts.
        """
        # this is a generalist implementation of an NPV function for an instrument with 2 legs.
        # most instruments may be likely to implement NPV directly to benefit from optimisations
        # specific to that instrument
        assert hasattr(self, "legs")  # noqa: S101

        _curves: _Curves = self._parse_curves(curves)
        _vol: _Vol = self._parse_vol(vol)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _vol_meta: _Vol = self.kwargs.meta["vol"]
        _fx_maybe_from_solver = _get_fx_maybe_from_solver(fx=fx, solver=solver)
        fx_vol = _maybe_get_fx_vol_maybe_from_solver(_vol_meta, _vol, solver)

        local_npv: dict[str, DualTypes] = {}
        for leg, names in zip(
            self.legs,
            [
                ("rate_curve", "disc_curve", "index_curve"),
                ("leg2_rate_curve", "leg2_disc_curve", "leg2_index_curve"),
            ],
            strict=False,
        ):
            leg_local_npv = leg.local_npv(
                rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                    _curves_meta, _curves, names[0], solver
                ),
                disc_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                    _curves_meta, _curves, names[1], solver
                ),
                index_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                    _curves_meta, _curves, names[2], solver
                ),
                fx=_fx_maybe_from_solver,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
            )
            if leg.settlement_params.currency in local_npv:
                local_npv[leg.settlement_params.currency] += leg_local_npv
            else:
                local_npv[leg.settlement_params.currency] = leg_local_npv

        if not local:
            single_value: DualTypes = 0.0
            base_ = _drb(self.settlement_params.currency, base)
            for k, v in local_npv.items():
                single_value += _maybe_fx_converted(
                    value=v,
                    currency=k,
                    fx=_fx_maybe_from_solver,
                    base=base_,
                    forward=forward,
                )
            return single_value
        else:
            return local_npv

    def _npv_single_core(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, DualTypes]:
        """
        Private NPV summation function used with a single thread, over all `self.instruments`.

        Returns a dict type: local = True
        """
        assert hasattr(self, "instruments")  # noqa: S101

        local_npv: dict[str, DualTypes] = {}
        for instrument in self.instruments:
            inst_local_npv = instrument.npv(
                curves=curves,
                solver=solver,
                fx=fx,
                vol=vol,
                base=base,
                local=True,
                settlement=settlement,
                forward=forward,
            )

            for k, v in inst_local_npv.items():
                if k in local_npv:
                    local_npv[k] += v
                else:
                    local_npv[k] = v
        return local_npv
