from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from packaging import version
from pandas import Series, read_csv
from pandas import __version__ as pandas_version

from rateslib.enums import Err

if TYPE_CHECKING:
    from rateslib.typing import Adjuster, CalTypes, DualTypes, Result


class _BaseFixingsLoader(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, name: str) -> Series[float]:
        """
        Get item method to load a fixing series from a custom container object.

        Parameters
        ----------
        name: str
            The name of the fixing series to load.

        Returns
        -------
        pandas Series

        Notes
        -----
        If a valid Series object cannot be loaded this method **must** raise an `Exception`,
        preferably a `ValueError`.
        """
        pass

    def get_stub_ibor_fixings(
        self,
        value_start_date: datetime,
        value_end_date: datetime,
        fixing_date: datetime,
        fixing_calendar: CalTypes,
        fixing_modifier: Adjuster,
        fixing_identifier: str,
    ) -> tuple[list[str], list[datetime], list[float | None]]:
        """
        Return the tenors available in the :class:`~rateslib.defaults.Fixings` object for
        determining an IBOR type stub period.

        Parameters
        ----------
        value_start_date: datetime
            The value start date of the IBOR period.
        value_end_date: datetime
            The value end date of the current stub period.
        fixing_date: datetime
            The index date to examine from the fixing series.
        fixing_calendar: Cal, UnionCal, NamedCal,
            The calendar to derive IBOR value end dates.
        fixing_modifier: Adjuster
            The date adjuster to derive IBOR value end dates.
        fixing_identifier: str
            The fixing name, prior to the addition of tenor, e.g. "EUR_EURIBOR"

        Returns
        -------
        tuple of list[string tenors] and list[evaluated end dates]
        """
        from rateslib.scheduling import add_tenor

        left: tuple[str | None, datetime] = (None, datetime(1, 1, 1))
        right: tuple[str | None, datetime] = (None, datetime(9999, 1, 1))

        for tenor in [
            "1D",
            "1B",
            "2B",
            "1W",
            "2W",
            "3W",
            "4W",
            "1M",
            "2M",
            "3M",
            "4M",
            "5M",
            "6M",
            "7M",
            "8M",
            "9M",
            "10M",
            "11M",
            "12M",
            "1Y",
        ]:
            try:
                _ = self.__getitem__(f"{fixing_identifier.upper()}_{tenor}")
            except Exception:  # noqa: S112
                continue
            else:
                sample_end = add_tenor(
                    start=value_start_date,
                    tenor=tenor,
                    modifier=fixing_modifier,
                    calendar=fixing_calendar,
                )
                if sample_end <= value_end_date and sample_end > left[1]:
                    left = (tenor, sample_end)
                if sample_end >= value_end_date and sample_end < right[1]:
                    right = (tenor, sample_end)
                    break

        ret: tuple[list[str], list[datetime], list[float | None]] = ([], [], [])
        if left[0] is not None:
            s = self[f"{fixing_identifier.upper()}_{left[0]}"]
            try:
                val: float = s[fixing_date]
            except KeyError:
                ret[2].append(None)
            else:
                ret[2].append(val)
            ret[0].append(left[0])
            ret[1].append(left[1])
        if right[0] is not None:
            s = self[f"{fixing_identifier.upper()}_{right[0]}"]
            try:
                val = s[fixing_date]
            except KeyError:
                ret[2].append(None)
            else:
                ret[2].append(val)
            ret[0].append(right[0])
            ret[1].append(right[1])
        return ret

    def get_index_value_from_fixings(
        self,
        index_lag: int,
        index_method: str,
        index_fixings: str,
        index_date: datetime,
    ) -> Result[DualTypes]:
        """
        Derive a value from a Series only, detecting cases where the errors might be raised.
        """
        try:
            fixings_series = self.__getitem__(index_fixings)
        except Exception as e:
            return Err(e)

        from rateslib.curves.curves import _index_value_from_series_no_curve

        return _index_value_from_series_no_curve(
            index_lag=index_lag,
            index_method=index_method,
            index_fixings=fixings_series,  # type: ignore[arg-type]
            index_date=index_date,
        )


class Fixings(_BaseFixingsLoader):
    """
    Object to store and load fixing data to populate *Leg* and *Period* calculations.

    .. warning::

       You must maintain and populate your own fixing data.

       *Rateslib* does not come pre-packaged with accurate, nor upto date fixing data.
       1) It does not have data licensing to distribute such data.
       2) It is a statically uploaded code package will become immediately out of date.

    .. attention::

       This object is loaded once by *rateslib* and is associated with its global
       :class:`~rateslib.default.Defaults` object under the attribute `defaults.fixings`.
       Only this object is referenced internally and other instantiations of this class
       will be ignored.

    Notes
    -----
    This class maintains a dictionary of financial fixing Series indexed by string identifiers.

    **Fixing Population**

    This dictionary can be populated in one of two ways:

    - Either by maintaining a set of CSV files in the source lookup directory (whose path is
      visible/settable by calling `defaults.fixings.directory`)
    - Or creating a pandas *Series* and using the :meth:`~rateslib.default.Fixings.add_series` to
      add this object to the dictionary.

    **Fixing Lookup**

    Lookup of a fixing *Series* is performed, for example using the get item pattern. If an
    object does not already exist in the dictionary it will be attempted to load from source CSV
    file. If neither exists it will raise a `ValueError`.

    .. ipython:: python
       :suppress:

       from pandas import Series
       from datetime import datetime as dt
       from rateslib import defaults

    .. ipython:: python

       cpi = Series(
           index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)],
           data=[100.0, 101.2, 102.2]
       )
       defaults.fixings.add_series("MY_CPI", cpi)
       defaults.fixings["MY_CPI"]

    .. ipython:: python

       try:
           defaults.fixings["NON_EXISTENT_SERIES"]
       except ValueError as e:
           print(e)

    The `Fixings` object can be overladed by a user customised implementation.
    For further info see :ref:`working with fixings <cook-fixings-doc>`.
    """

    @staticmethod
    def _load_csv(directory: str, path: str) -> Series[float]:
        target = os.path.join(directory, path)
        if version.parse(pandas_version) < version.parse("2.0"):  # pragma: no cover
            # this is tested by the minimum version gitflow actions.
            # TODO (low:dependencies) remove when pandas min version is bumped to 2.0
            df = read_csv(target)
            df["reference_date"] = df["reference_date"].map(
                lambda x: datetime.strptime(x, "%d-%m-%Y"),
            )
            df = df.set_index("reference_date")
        else:
            df = read_csv(target, index_col=0, parse_dates=[0], date_format="%d-%m-%Y")
        return df["rate"].sort_index(ascending=True)

    def __getitem__(self, name: str) -> Series[float]:
        if name.upper() in self.loaded:
            return self.loaded[name.upper()]

        try:
            s = self._load_csv(self.directory, f"{name}.csv")
        except FileNotFoundError:
            raise ValueError(
                f"Fixing data for the index '{name}' has been attempted, but there is no file:\n"
                f"'{name}.csv' located in the search directory.\n"
                "For further info see the documentation for the `Fixings` class and/or the "
                "cookbook article  'Working with Fixings'.",
            )

        self.loaded[name.upper()] = s
        return s

    def __init__(self) -> None:
        self.directory = os.path.dirname(os.path.abspath(__file__)) + "/data"
        self.loaded: dict[str, Series[float]] = {}

    def add_series(self, name: str, series: Series[float]) -> None:
        if name in self.loaded:
            raise ValueError(f"Fixing data for the index '{name}' has already been loaded.")
        s = series.sort_index(ascending=True)
        s.index.name = "reference_date"
        s.name = "rate"
        self.loaded[name.upper()] = s

    def remove_series(self, name: str) -> Series[float] | None:
        return self.loaded.pop(name.upper(), None)
