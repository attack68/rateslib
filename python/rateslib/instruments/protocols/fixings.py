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

from pandas import DataFrame, Series

from rateslib.enums.generics import NoInput
from rateslib.periods.protocols.fixings import (
    _replace_fixings_with_ad_variables,
    _reset_fixings_data,
    _structure_sensitivity_data,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Sequence,
        Solver_,
        VolT_,
        datetime_,
        int_,
        str_,
    )


class _WithFixings(Protocol):
    """
    Protocol for determining fixing sensitivity for a *Period* with AD.

    .. rubric:: Provided methods

    .. autosummary::

       ~_WithFixings.reset_fixings

    """

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
    ) -> DualTypes | dict[str, DualTypes]: ...

    def reset_fixings(self, state: int_ = NoInput(0)) -> None:
        """
        Resets any fixings values of the *Instrument* derived using the given data state.

        .. role:: green

        Parameters
        ----------
        state: int, :green:`optional`
            The *state id* of the data series that set the fixing. Only fixings determined by this
            data will be reset. If not given resets all fixings.

        Returns
        -------
        None
        """
        if hasattr(self, "legs"):
            for leg in self.legs:
                leg.reset_fixings(state)
        elif hasattr(self, "instruments"):
            for inst in self.instruments:
                inst.reset_fixings(state)

    def local_fixings(
        self,
        identifiers: Sequence[tuple[str, Series]],
        scalars: Sequence[float] | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Calculate the sensitivity to fixings of the *Instrument*, expressed in local
        settlement currency.

        .. role:: red

        .. role:: green

        Parameters
        ----------
        identifiers: Sequence of tuple[str, Series], :red:`required`
            These are the series string identifiers and the data values that will be used in each
            Series to determine the sensitivity against.
        scalars: Sequence of floats, :green:`optional (each set as 1.0)`
            A sequence of scalars to multiply the sensitivities by for each on of the
            ``identifiers``.
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        original_data, index, state = _replace_fixings_with_ad_variables(identifiers)
        # Extract sensitivity data
        pv: dict[str, DualTypes] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
            local=True,
        )
        df = _structure_sensitivity_data(pv, index, identifiers, scalars)
        _reset_fixings_data(self, original_data, state, identifiers)
        return df
