from __future__ import annotations  # type hinting

from typing import TYPE_CHECKING, NoReturn, TypeAlias

from rateslib.default import NoInput, PlotOutput, _drb, plot
from rateslib.dual import Dual, Dual2, Variable
from rateslib.fx_volatility.utils import _FXDeltaVolSmileMeta, _FXSabrSmileMeta
from rateslib.mutability import _WithCache, _WithState

if TYPE_CHECKING:
    from rateslib.typing import FXForwards

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure


class _BaseSmile(_WithState, _WithCache[float, DualTypes]):
    _ad: int
    _default_plot_x_axis: str
    meta: _FXDeltaVolSmileMeta | _FXSabrSmileMeta

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Smile*."""
        return self._ad

    def __iter__(self) -> NoReturn:
        raise TypeError("`Smile` types are not iterable.")

    def plot(
        self,
        comparators: list[_BaseSmile] | NoInput = NoInput(0),
        labels: list[str] | NoInput = NoInput(0),
        x_axis: str | NoInput = NoInput(0),
        f: DualTypes | FXForwards | NoInput = NoInput(0),
    ) -> PlotOutput:
        """
        Plot volatilities associated with the *Smile*.

        .. warning::

           The *'delta'* ``x_axis`` type for a *SabrSmile* is calculated based on a
           **forward, unadjusted** delta and is expressed as a negated put option delta
           consistent with the definition for a :class:`~rateslib.fx_volatility.FXDeltaVolSmile`.

        Parameters
        ----------
        comparators: list[Smile]
            A list of Smiles which to include on the same plot as comparators.
            Note the comments on
            :meth:`FXDeltaVolSmile.plot <rateslib.fx_volatility.FXDeltaVolSmile.plot>`.
        labels : list[str]
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.
        x_axis : str in {"strike", "moneyness", "delta"}
            *'strike'* is the natural option for this *SabrSmile* types while *'delta'* is the
            natural choice for *DeltaVolSmile* types.
            If *'delta'* see the warning. If *'moneyness'* the strikes are converted using ``f``.
        f: DualTypes
            The FX forward rate at delivery.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D
        """
        # reversed for intuitive strike direction
        comparators = _drb([], comparators)
        labels = _drb([], labels)

        x_axis_: str = _drb(self.meta.plot_x_axis, x_axis)

        x_, y_ = self._plot(x_axis_, f)  # type: ignore[attr-defined]

        x = [x_]
        y = [y_]
        if not isinstance(comparators, NoInput):
            for smile in comparators:
                if not isinstance(smile, _BaseSmile):
                    raise ValueError("A `comparator` must be a valid FX Smile type.")
                x_, y_ = smile._plot(x_axis_, f)  # type: ignore[attr-defined]
                x.append(x_)
                y.append(y_)

        return plot(x, y, labels)
