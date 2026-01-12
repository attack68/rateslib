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

import warnings
from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, concat

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        FXForwards_,
        Sequence,
        _BaseCurve_,
        _BasePeriod,
        _FXVolOption_,
        datetime_,
    )


class _WithAnalyticRateFixings(Protocol):
    """
    Protocol to calculate analytical rate fixing sensitivities of any *Leg* type.

    """

    @property
    def periods(self) -> Sequence[_BasePeriod]: ...

    def local_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of financial sensitivity to published interest rate fixings,
        expressed in local **settlement currency** of the *Period*.

        If the *Period* has no sensitivity to rates fixings this *DataFrame* is empty.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        dfs = []
        for period in self.periods:
            dfs.append(
                period.local_analytic_rate_fixings(
                    rate_curve=rate_curve,
                    index_curve=index_curve,
                    disc_curve=disc_curve,
                    fx=fx,
                    fx_vol=fx_vol,
                    settlement=settlement,
                    forward=forward,
                )
            )

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            return concat(dfs)
