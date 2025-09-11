from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.enums.generics import NoInput
from rateslib.rs import Adjuster, NamedCal
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.convention import Convention, _get_convention
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import CalTypes, Frequency, str_


class FloatRateIndex:
    """
    Define the parameters of a specific interest rate index.

    Parameters
    ----------
    frequency : Frequency or str
        The specific tenor of the interest rate index.
    series : Series or str
        The general parameters applied to any tenor of this particular interest rate series.

    Examples
    --------
    None
    """

    _frequency: Frequency
    _series: FloatRateSeries

    def __init__(
        self,
        frequency: Frequency | str,
        series: FloatRateSeries | str,
    ) -> None:
        self._series = _get_float_rate_series(series)
        self._frequency = _get_frequency(frequency, NoInput(0), self.calendar)

    @property
    def frequency(self) -> Frequency:
        """The specific tenor of the interest rate index."""
        return self._frequency

    @property
    def series(self) -> FloatRateSeries:
        """The general parameters applied to any tenor of this particular interest rate series."""
        return self._series

    @property
    def lag(self) -> int:
        """The lag for the determining the publication date of the interest rate index."""
        return self.series.lag

    @property
    def calendar(self) -> CalTypes:
        """The calendar associated with the publication of the interest rate index."""
        return self.series.calendar

    @property
    def modifier(self) -> Adjuster:
        """The :class:`Adjuster` associated with the end accrual day of the interest rate index."""
        return self.series.modifier

    @property
    def eom(self) -> bool:
        """Whether the interest rate index adopts an end of month convention."""
        return self.series.eom

    @property
    def convention(self) -> Convention:
        """The :class:`Convention` associated with the publication of the interest rate index."""
        return self.series.convention


class FloatRateSeries:
    """
    Define the general parameters of multiple tenors of an interest rate series.

    Parameters
    ----------
    lag: int
        The

    Examples
    --------
    None
    """

    _lag: int
    _calendar: CalTypes
    _modifier: Adjuster
    _convention: Convention
    _eom: bool

    def __init__(
        self,
        lag: int,
        calendar: CalTypes | str,
        modifier: Adjuster | str,
        convention: Convention | str,
        eom: bool,
    ) -> None:
        self._lag = lag
        self._calendar = get_calendar(calendar)
        self._modifier = _get_adjuster(modifier)
        self._convention = _get_convention(convention)
        self._eom = eom

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def calendar(self) -> CalTypes:
        return self._calendar

    @property
    def convention(self) -> Convention:
        return self._convention

    @property
    def modifier(self) -> Adjuster:
        return self._modifier

    @property
    def eom(self) -> bool:
        return self._eom


_SERIES_MAP = {
    "usd_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("nyc"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act360,
        eom=False,
    ),
    "usd_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("nyc"),
        modifier=Adjuster.Following(),
        convention=Convention.Act360,
        eom=False,
    ),
    "gbp_ibor": FloatRateSeries(
        lag=0,
        calendar=NamedCal("ldn"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act365F,
        eom=True,
    ),
    "gbp_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("ldn"),
        modifier=Adjuster.Following(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "sek_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("ldn"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act360,
        eom=True,
    ),
    "sek_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("ldn"),
        modifier=Adjuster.Following(),
        convention=Convention.Act360,
        eom=False,
    ),
    "eur_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("tgt"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act360,
        eom=False,
    ),
    "eur_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("tgt"),
        modifier=Adjuster.Following(),
        convention=Convention.Act360,
        eom=False,
    ),
    "nok_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("osl"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act360,
        eom=False,
    ),
    "nok_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("osl"),
        modifier=Adjuster.Following(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "chf_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("zur"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act360,
        eom=False,
    ),
    "chf_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("zur"),
        modifier=Adjuster.Following(),
        convention=Convention.Act360,
        eom=False,
    ),
    "cad_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("tro"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "cad_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("tro"),
        modifier=Adjuster.Following(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "jpy_ibor": FloatRateSeries(
        lag=2,
        calendar=NamedCal("tyo"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "jpy_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("tyo"),
        modifier=Adjuster.Following(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "aud_ibor": FloatRateSeries(
        lag=0,
        calendar=NamedCal("syd"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act365F,
        eom=True,
    ),
    "aud_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("syd"),
        modifier=Adjuster.Following(),
        convention=Convention.Act365F,
        eom=False,
    ),
    "nzd_ibor": FloatRateSeries(
        lag=0,
        calendar=NamedCal("wlg"),
        modifier=Adjuster.ModifiedFollowing(),
        convention=Convention.Act365F,
        eom=True,
    ),
    "nzd_rfr": FloatRateSeries(
        lag=0,
        calendar=NamedCal("wlg"),
        modifier=Adjuster.Following(),
        convention=Convention.Act365F,
        eom=False,
    ),
}


def _get_float_rate_series(val: FloatRateSeries | str) -> FloatRateSeries:
    if isinstance(val, FloatRateSeries):
        return val
    else:
        return _SERIES_MAP[val.lower()]


def _get_float_rate_series_or_blank(val: FloatRateSeries | str_) -> FloatRateSeries | NoInput:
    if isinstance(val, NoInput):
        return val
    else:
        return _get_float_rate_series(val)
