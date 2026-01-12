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

import os
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import rateslib.errors as err
from packaging import version
from pandas import Series, read_csv
from pandas import __version__ as pandas_version
from rateslib.enums.generics import Err, NoInput, Ok

if TYPE_CHECKING:
    from rateslib.typing import (
        Adjuster,
        CalTypes,
        DualTypes,
        FloatRateSeries,
        Result,
        datetime_,
        int_,
    )


class _BaseFixingsLoader(metaclass=ABCMeta):
    """
    Abstract base class to allow custom implementations of a fixings data loader.

    Notes
    -----
    This class requires an implementation of ``__getitem__``, which should accept an
    ``identifier`` and return a 3-tuple. The 3-tuple should include;

    - an integer representing the state id of the loaded data, i.e. its hash or pseudo-hash.
    - the data itself as a Series indexed by daily datetimes.
    - a 2-tuple of datetimes indicating the min and max of the timeseries index.

    If a valid Series object cannot be loaded for the ``identifier`` then this method
    is required to raise a `ValeuError`.
    """

    @abstractmethod
    def __getitem__(self, name: str) -> tuple[int, Series[DualTypes], tuple[datetime, datetime]]:  # type: ignore[type-var]
        """
        Get item method to load a fixing series and ist state id from a custom container object.

        Parameters
        ----------
        name: str
            The name of the fixing series to load.

        Returns
        -------
        tuple of int, pandas Series, and tuple of datetime

        Notes
        -----
        The first tuple element is a hash integer which represents the state of the Series object.
        This is used to determine if the Series object has changed since it was last loaded,
        and makes for more efficient fixings lookup calculations in *Periods*.

        The second element is the timeseries object itself.

        The third tuple element is a cached record of the first and last dates in the Series index.

        If a valid Series object cannot be loaded this method **must** raise an `Exception`,
        preferably a `ValueError`.
        """
        pass

    @abstractmethod
    def add(self, name: str, series: Series[DualTypes], state: int_ = NoInput(0)) -> None:  # type: ignore[type-var]
        """
        Add a timeseries to the data loader directly from Python.

        Parameters
        ----------
        name: str
            The string identifier for the timeseries.
        series: Series[DualTypes]
            The timeseries to add to static data.

        Returns
        -------
        None

        Examples
        --------

        .. ipython:: python
           :suppress:

           from rateslib import fixings, dt
           from pandas import Series

        .. ipython:: python

           ts = Series(index=[dt(2000, 1, 1)], data=[666.0])
           fixings.add("my_timeseries", ts)
           fixings["my_timeseries"]
           fixings.pop("my_timeseries")

        """
        pass

    @abstractmethod
    def pop(self, name: str) -> Series[DualTypes] | None:  # type: ignore[type-var]
        """
        Remove a timeseries from the data loader.

        Parameters
        ----------
        name: str
            The string identifier for the timeseries.

        Returns
        -------
        Series[DualTypes] or None

        Notes
        -----
        If the ``name`` does not exist None will be returned.
        """
        pass

    def __try_getitem__(
        self, name: str
    ) -> Result[tuple[int, Series[DualTypes], tuple[datetime, datetime]]]:  # type: ignore[type-var]
        try:
            tuple_value = self.__getitem__(name)
        except Exception as e:
            return Err(e)
        else:
            return Ok(tuple_value)

    def __base_lookup__(
        self,
        fixing_series: Series[DualTypes],  # type: ignore[type-var]
        lookup_date: datetime_,
        bounds: tuple[datetime, datetime] | None = None,
    ) -> Result[DualTypes]:
        if bounds is not None:
            left, right = bounds
        else:
            # default to slower mechanism of lookup
            left, right = fixing_series.index[0], fixing_series.index[-1]

        if isinstance(lookup_date, NoInput):
            # program break, raise directly
            raise ValueError("A `lookup_date` must be provided for fetching fixings from Series.")
        if lookup_date < left or lookup_date > right:
            return Err(FixingRangeError(lookup_date, (left, right)))

        if lookup_date not in fixing_series.index:
            return Err(FixingMissingDataError(lookup_date, (left, right)))
        else:
            return Ok(fixing_series.loc[lookup_date])

    def get_stub_ibor_fixings(
        self,
        value_start_date: datetime,
        value_end_date: datetime,
        fixing_date: datetime,
        fixing_calendar: CalTypes,
        fixing_modifier: Adjuster,
        fixing_identifier: str,
    ) -> tuple[list[str], list[datetime], list[DualTypes | None]]:
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

        def _is_available(tenor: str) -> bool:
            try:
                self.__getitem__(f"{fixing_identifier.upper()}_{tenor}")
            except Exception:  # noqa: S112
                return False
            else:
                return True

        tenors = ["1D", "1B", "2B", "1W", "2W", "3W", "4W"] + [
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
        ]

        available_tenors = [tenor for tenor in tenors if _is_available(tenor)]
        from rateslib.data.fixings import FloatRateSeries

        neighbouring_tenors = _find_neighbouring_tenors(
            end=value_end_date,
            start=value_start_date,
            tenors=available_tenors,
            rate_series=FloatRateSeries(
                lag=0, calendar=fixing_calendar, convention="1", modifier=fixing_modifier, eom=False
            ),
        )

        values: list[DualTypes | None] = []
        for tenor in neighbouring_tenors[0]:
            try:
                val: DualTypes = self.__getitem__(f"{fixing_identifier.upper()}_{tenor}")[1][
                    fixing_date
                ]
            except KeyError:
                values.append(None)
            else:
                values.append(val)
        return neighbouring_tenors + (values,)


class DefaultFixingsLoader(_BaseFixingsLoader):
    """
    The :class:`~rateslib.data.loader._BaseFixingsLoader` implemented by default.

    This loader searches a particular local directory for CSV files.
    """

    def __init__(self) -> None:
        self._directory = os.path.dirname(os.path.abspath(__file__)) + "/historical"
        self._loaded: dict[str, tuple[int, Series[DualTypes], tuple[datetime, datetime]]] = {}  # type: ignore[type-var]

    @property
    def directory(self) -> str:
        """The local directory in which data CSV files may be located."""
        return self._directory

    @directory.setter
    def directory(self, val: str) -> None:
        self._directory = val

    @property
    def loaded(self) -> dict[str, tuple[int, Series[DualTypes], tuple[datetime, datetime]]]:  # type: ignore[type-var]
        """A dictionary of the (state id, timeseries, data range) keyed by identifiers."""
        return self._loaded

    @staticmethod
    def _load_csv(directory: str, path: str) -> Series[DualTypes]:  # type: ignore[type-var]
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

    def __getitem__(self, name: str) -> tuple[int, Series[DualTypes], tuple[datetime, datetime]]:  # type: ignore[type-var]
        name_ = name.upper()
        if name_ in self.loaded:
            return self.loaded[name_]

        try:
            s: Series[DualTypes] = self._load_csv(self.directory, f"{name}.csv")  # type: ignore[type-var]
        except FileNotFoundError:
            raise ValueError(
                f"Fixing data for the index '{name}' has been attempted, but there is no file:\n"
                f"'{name}.csv' located in the search directory.\n"
                "For further info see the documentation section regarding `Fixings`.",
            )

        data = (hash(os.urandom(8)), s, (s.index[0], s.index[-1]))
        self.loaded[name_] = data
        return data

    def add(self, name: str, series: Series[DualTypes], state: int_ = NoInput(0)) -> None:  # type: ignore[type-var]
        if name in self.loaded:
            raise ValueError(f"Fixing data for the index '{name}' has already been loaded.")
        s = series.sort_index(ascending=True)
        s.index.name = "reference_date"
        s.name = "rate"
        name_ = name.upper()
        if isinstance(state, NoInput):
            state_: int = hash(os.urandom(64))
        else:
            state_ = state
        self.loaded[name_] = (state_, s, (s.index[0], s.index[-1]))

    def pop(self, name: str) -> Series[DualTypes] | None:  # type: ignore[type-var]
        name_ = name.upper()
        popped = self.loaded.pop(name_, None)
        if popped is not None:
            return popped[1]  # return the Series object only
        else:
            return None


class Fixings(_BaseFixingsLoader):
    """
    Object to store and load fixing data to populate *Leg* and *Period* calculations.

    .. warning::

       You must maintain and populate your own fixing data.

       *Rateslib* does not come pre-packaged with accurate, nor upto date fixing data.
       1) It does not have data licensing to distribute such data.
       2) It is a statically uploaded code package will become immediately out of date.

    .. attention::

       This object is loaded **once** by *rateslib* and in its global module,
       under the attribute `fixings`.
       Only this object is referenced internally and other instantiations of this class
       will be ignored.

    Notes
    -----
    The ``loader`` is initialised as the :class:`DefaultFixingsLoader`. This can be set as
    a user implemented :class:`_BaseFixingsLoader`.

    This class maintains a dictionary of financial fixing Series indexed by string identifiers.

    **Fixing Population**

    This dictionary can be populated in one of two ways:

    - Either by maintaining a set of CSV files in the source lookup directory (whose path is
      visible/settable by calling `fixings.directory`)
    - Or creating a pandas *Series* and using the :meth:`~rateslib.default.Fixings.add` to
      add this object to the dictionary.

    **Fixing Lookup**

    Lookup of a fixing *Series* is performed, for example using the get item pattern. If an
    object does not already exist in the dictionary it will be attempted to load from source CSV
    file. If neither exists it will raise a `ValueError`.

    .. ipython:: python
       :suppress:

       from pandas import Series
       from datetime import datetime as dt
       from rateslib import fixings

    .. ipython:: python

       cpi = Series(
           index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)],
           data=[100.0, 101.2, 102.2]
       )
       fixings.add("MY_CPI", cpi)
       fixings["MY_CPI"]

    .. ipython:: python

       try:
           fixings["NON_EXISTENT_SERIES"]
       except ValueError as e:
           print(e)

    """

    _instance = None

    def __new__(cls) -> Fixings:
        if cls._instance is None:
            # Singleton pattern creates only one instance: TODO (low) might not be thread safe
            cls._instance = super(_BaseFixingsLoader, cls).__new__(cls)  # noqa: UP008

            cls._loader: _BaseFixingsLoader = DefaultFixingsLoader()

        return cls._instance

    def __getitem__(self, name: str) -> tuple[int, Series[DualTypes], tuple[datetime, datetime]]:  # type: ignore[type-var]
        return self.loader.__getitem__(name)

    @property
    def loader(self) -> _BaseFixingsLoader:
        """
        Object responsible for fetching data from external sources.
        """
        return self._loader

    @loader.setter
    def loader(self, loader: _BaseFixingsLoader) -> None:
        self._loader = loader

    def add(self, name: str, series: Series[DualTypes], state: int_ = NoInput(0)) -> None:  # type: ignore[type-var]
        """
        Add a Series to the Fixings object directly from Python

        .. role:: red

        .. role:: green

        Parameters
        ----------
        name: str, :red:`required`
            The string identifier key for the timeseries.
        series: Series, :red:`required`
            The timeseries indexed by datetime.
        state: int, :green:`optional`
            The state id to be used upon insertion of the Series.

        Returns
        -------
        None
        """
        return self.loader.add(name, series, state)

    def pop(self, name: str) -> Series[DualTypes] | None:  # type: ignore[type-var]
        """
        Remove a Series from the Fixings object.

        .. role:: red

        Parameters
        ----------
        name: str, :red:`required`
            The string identifier key for the timeseries.

        Returns
        -------
        Series, or None (if name not found)
        """
        return self.loader.pop(name)


class FixingRangeError(Exception):
    def __init__(self, date: datetime, boundary: tuple[datetime, datetime]) -> None:
        super().__init__(
            f"Fixing lookup for date '{date}' failed.\n"
            f"The fixings series has range [{boundary[0]}, {boundary[1]}]"
        )
        self.date = date
        self.boundary = boundary


class FixingMissingDataError(Exception):
    def __init__(self, date: datetime, boundary: tuple[datetime, datetime]) -> None:
        super().__init__(
            f"Fixing lookup for date '{date}' failed.\n"
            f"The requested date falls within the fixings series range "
            f"[{boundary[0]}, {boundary[1]}] but was not found."
        )
        self.date = date
        self.boundary = boundary


class FixingMissingForecasterError(Exception):
    def __init__(self) -> None:
        super().__init__(err.VE_NEEDS_RATE_TO_FORECAST_RFR)


def _find_neighbouring_tenors(
    end: datetime,
    start: datetime,
    tenors: list[str],
    rate_series: FloatRateSeries,
) -> tuple[list[str], list[datetime]]:
    """
    Given a list of string tenors find the two, measured from `start`, that encompass `end`
    on neighbouring sides. If outside, find the closest single tenor.
    """
    from rateslib.scheduling import add_tenor

    left: tuple[str | None, datetime] = (None, datetime(1, 1, 1))
    right: tuple[str | None, datetime] = (None, datetime(9999, 1, 1))

    for tenor in tenors:
        sample_end = add_tenor(
            start=start,
            tenor=tenor,
            modifier=rate_series.modifier,
            calendar=rate_series.calendar,
        )
        if sample_end <= end and sample_end > left[1]:
            left = (tenor, sample_end)
        if sample_end >= end and sample_end < right[1]:
            right = (tenor, sample_end)
            break

    ret: tuple[list[str], list[datetime]] = ([], [])
    if left[0] is not None:
        ret[0].append(left[0].upper())
        ret[1].append(left[1])
    if right[0] is not None:
        ret[0].append(right[0].upper())
        ret[1].append(right[1])
    return ret


__all__ = ["Fixings", "DefaultFixingsLoader", "_BaseFixingsLoader"]
