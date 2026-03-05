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


from __future__ import annotations  # type hinting

from typing import TYPE_CHECKING, NoReturn, TypeAlias

from rateslib.default import PlotOutput, plot
from rateslib.dual import Dual, Dual2, Variable
from rateslib.enums.generics import NoInput, _drb
from rateslib.mutability import _WithCache, _WithState
from rateslib.volatility.ir.utils import _IRSmileMeta

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        CurvesT_,
    )

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure


class _BaseIRSmile(_WithState, _WithCache[float, DualTypes]):
    """Abstract base class for implementing *IR Smiles*."""

    _ad: int
    _default_plot_x_axis: str
    meta: _IRSmileMeta

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Smile*."""
        return self._ad

    def __iter__(self) -> NoReturn:
        raise TypeError("`Smile` types are not iterable.")

    def plot(
        self,
        comparators: list[_BaseIRSmile] | NoInput = NoInput(0),
        labels: list[str] | NoInput = NoInput(0),
        x_axis: str | NoInput = NoInput(0),
        y_axis: str | NoInput = NoInput(0),
        f: DualTypes | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> PlotOutput:
        """
        Plot volatilities associated with the *Smile*.

        .. role:: green

        .. role:: red

        Parameters
        ----------
        comparators: list[Smile], :green:`optional`
            A list of Smiles which to include on the same plot as comparators.
        labels : list[str], :green:`optional`
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.
        x_axis : str in {"strike", "moneyness"}, :green:`optional (set by object)`
            *'strike'* is the natural option for this *SabrSmile*.
            If *'moneyness'* the strikes are converted using ``f``.
        y_axis : str in {"black_vol", "normal_vol"}, :green:`optional (set by object)`
            Convert the y-axis to a different representation using the Hagan approximation from
            `Managing Smile Risk <https://www.researchgate.net/publication/235622441_Managing_Smile_Risk>`__
        f: DualTypes, :green:`optional`
            The mid-market IRS rate. If ``curves`` are not given then ``f`` is required.
        curves: Curves, :green:`optional`
            The *Curves* in the required form for an :class:`~rateslib.instruments.IRS`. If ``f``
            is not given then ``curves`` are required.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        # reversed for intuitive strike direction
        comparators = _drb([], comparators)
        labels = _drb([], labels)

        x_axis_: str = _drb(self.meta.plot_x_axis, x_axis)
        y_axis_: str = _drb(self.meta.plot_y_axis, y_axis)
        del x_axis, y_axis

        x_, y_ = self._plot(x_axis_, f, y_axis_, curves)  # type: ignore[attr-defined]

        x = [x_]
        y = [y_]
        if not isinstance(comparators, NoInput):
            for smile in comparators:
                if not isinstance(smile, _BaseIRSmile):
                    raise ValueError("A `comparator` must be a valid IR Smile type.")
                x_, y_ = smile._plot(x_axis_, f, y_axis_, curves)  # type: ignore[attr-defined]
                x.append(x_)
                y.append(y_)

        return plot(x, y, labels)
