from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        _Curves,
    )


class _WithCurves(Protocol):
    """
    Protocol to determine individual *curve* inputs for each *Instrument*, possibly from a
    :class:`~rateslib.solver.Solver` mapping, from a generic ``curves`` argument.
    """

    def _parse_curves(self, curves: CurveOption_) -> _Curves:
        """Method is needed to map the `curves` argument input for any individual Instrument into
        the more defined :class:`~rateslib.curves._parsers._Curves` structure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_parse_curves` of class `_WithCurves`."
        )
