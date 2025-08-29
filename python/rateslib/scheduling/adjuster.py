from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.enums import NoInput
from rateslib.rs import Adjuster

if TYPE_CHECKING:
    from rateslib.typing import str_

_A = {  # Provides the map of all available string to Adjuster conversions.
    "NONESETTLE": Adjuster.Actual(),
    "NONE": Adjuster.Actual(),
    "F": Adjuster.Following(),
    "P": Adjuster.Previous(),
    "MF": Adjuster.ModifiedFollowing(),
    "MP": Adjuster.ModifiedPrevious(),
    "FSETTLE": Adjuster.FollowingSettle(),
    "PSETTLE": Adjuster.PreviousSettle(),
    "MFSETTLE": Adjuster.ModifiedFollowingSettle(),
    "MPSETTLE": Adjuster.ModifiedPreviousSettle(),
    "FEX": Adjuster.FollowingExLast(),
    "FEXSETTLE": Adjuster.FollowingExLastSettle(),
}


def _get_adjuster_none(adjuster: Adjuster | str_) -> Adjuster | None:
    if isinstance(adjuster, NoInput):
        return None
    else:
        return _get_adjuster(adjuster)


def _get_adjuster(adjuster: str | Adjuster) -> Adjuster:
    """Convert a str such as 'F', 'MF' or '2B' or '5D' to an Adjuster."""
    if isinstance(adjuster, Adjuster):
        return adjuster
    else:
        adjuster = adjuster.upper()
        if adjuster[-1] == "B":
            return Adjuster.BusDaysLagSettle(int(adjuster[:-1]))
        elif adjuster[-1] == "D":
            return Adjuster.CalDaysLagSettle(int(adjuster[:-1]))
        else:
            return _A[adjuster]


def _convert_to_adjuster(modifier: str | Adjuster, settlement: bool, mod_days: bool) -> Adjuster:
    """Convert a legacy `modifier` to an Adjuster with additional options."""
    if isinstance(modifier, Adjuster):
        return modifier
    modifier = modifier.upper()
    if not mod_days and modifier[0] == "M":
        modifier = modifier[1:]
    if settlement:
        modifier = modifier + "SETTLE"
    return _get_adjuster(modifier)
