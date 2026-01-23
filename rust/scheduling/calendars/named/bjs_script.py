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

from datetime import datetime, timedelta

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
)
from pandas.tseries.offsets import CustomBusinessDay

"""
The Chinese holiday system is quite complex. This script focuses on 2025 and later.

Most holidays are defined relative to lunar or solar events whose dates must be known or tabulated
in advance. Some of the data here is collected versus external, cited sources.

A system of bridging and repaying holidays is in force meaning generic week masks cannot be
applied becuase some Saturdays or Sundays will be official work days.

Regulation in 2025 aimed to reduce the number of "owed" days but these still exist throughout the
year.

For each holiday this script aims to return a list of holiday dates and a list of compensation
days, if required.

"""


def weekday_mask(
    weekdays: list[int], years: tuple[int, int], exclude: list[datetime]
) -> list[datetime]:
    iterate = datetime(years[0], month=1, day=1)
    end = datetime(years[1], month=12, day=31)
    holidays = []
    while iterate < end:
        if iterate.weekday() in weekdays and iterate not in exclude:
            holidays.append(
                Holiday(
                    f"Weekday Mask {iterate.strftime('yymmdd')}",
                    year=iterate.year,
                    month=iterate.month,
                    day=iterate.day,
                )
            )
        iterate = iterate + timedelta(days=1)
    return holidays


def new_years_holidays(year: int) -> list[datetime]:
    dt = datetime(year=year, month=1, day=1)
    if dt.weekday() in [3, 4, 5]:
        # Th, Fr, Sa roll forwards
        idx = 0
    elif dt.weekday() in [0, 1]:
        # Mo, Tu roll backwards
        idx = -2
    elif dt.weekday() in [2]:
        # We is single day
        return [dt]
    elif dt.weekday() in [6]:
        # Su roll backwards
        idx = -1
    return [dt + timedelta(days=i) for i in range(idx, idx + 3)]


def new_years_compensations(year: int) -> list[datetime]:
    dt = datetime(year=year, month=1, day=1)
    if dt.weekday() in [0, 2, 4, 5, 6]:
        return []
    elif dt.weekday() in [1]:
        return [datetime(year=year - 1, month=12, day=29)]
    else:
        return [datetime(year=year, month=1, day=4)]


# parsed with re: from https://taiwan-database.net/PDFs/WTFpdf23.pdf
lunar_new_year_dates = [
    "1800-01-25",
    "1801-02-13",
    "1802-02-03",
    "1803-01-23",
    "1804-02-11",
    "1805-01-31",
    "1806-02-18",
    "1807-02-07",
    "1808-01-28",
    "1809-02-14",
    "1810-02-04",
    "1811-01-25",
    "1812-02-13",
    "1813-02-01",
    "1814-01-21",
    "1815-02-09",
    "1816-01-29",
    "1817-02-16",
    "1818-02-05",
    "1819-01-26",
    "1820-02-14",
    "1821-02-03",
    "1822-01-23",
    "1823-02-11",
    "1824-01-31",
    "1825-02-18",
    "1826-02-07",
    "1827-01-27",
    "1828-02-15",
    "1829-02-04",
    "1830-01-25",
    "1831-02-13",
    "1832-02-02",
    "1833-02-20",
    "1834-02-09",
    "1835-01-29",
    "1836-02-17",
    "1837-02-05",
    "1838-01-26",
    "1839-02-14",
    "1840-02-03",
    "1841-01-23",
    "1842-02-10",
    "1843-01-30",
    "1844-02-18",
    "1845-02-07",
    "1846-01-27",
    "1847-02-15",
    "1848-02-05",
    "1849-01-24",
    "1850-02-12",
    "1851-02-01",
    "1852-02-20",
    "1853-02-08",
    "1854-01-29",
    "1855-02-17",
    "1856-02-06",
    "1857-01-26",
    "1858-02-14",
    "1859-02-03",
    "1860-01-23",
    "1861-02-10",
    "1862-01-30",
    "1863-02-18",
    "1864-02-08",
    "1865-01-27",
    "1866-02-15",
    "1867-02-05",
    "1868-01-25",
    "1869-02-11",
    "1870-01-31",
    "1871-02-19",
    "1872-02-09",
    "1873-01-29",
    "1874-02-17",
    "1875-02-06",
    "1876-01-26",
    "1877-02-13",
    "1878-02-02",
    "1879-01-22",
    "1880-02-10",
    "1881-01-30",
    "1882-02-18",
    "1883-02-08",
    "1884-01-28",
    "1885-02-15",
    "1886-02-04",
    "1887-01-24",
    "1888-02-12",
    "1889-01-31",
    "1890-01-21",
    "1891-02-09",
    "1892-01-30",
    "1893-02-17",
    "1894-02-06",
    "1895-01-26",
    "1896-02-13",
    "1897-02-02",
    "1898-01-22",
    "1899-02-10",
    "1900-01-31",
    "1901-02-19",
    "1902-02-08",
    "1903-01-29",
    "1904-02-16",
    "1905-02-04",
    "1906-01-25",
    "1907-02-13",
    "1908-02-02",
    "1909-01-22",
    "1910-02-10",
    "1911-01-30",
    "1912-02-18",
    "1913-02-06",
    "1914-01-26",
    "1915-02-14",
    "1916-02-03",
    "1917-01-23",
    "1918-02-11",
    "1919-02-01",
    "1920-02-20",
    "1921-02-08",
    "1922-01-28",
    "1923-02-16",
    "1924-02-05",
    "1925-01-24",
    "1926-02-13",
    "1927-02-02",
    "1928-01-23",
    "1929-02-10",
    "1930-01-30",
    "1931-02-17",
    "1932-02-06",
    "1933-01-26",
    "1934-02-14",
    "1935-02-04",
    "1936-01-24",
    "1937-02-11",
    "1938-01-31",
    "1939-02-19",
    "1940-02-08",
    "1941-01-27",
    "1942-02-15",
    "1943-02-05",
    "1944-01-25",
    "1945-02-13",
    "1946-02-02",
    "1947-01-22",
    "1948-02-10",
    "1949-01-29",
    "1950-02-17",
    "1951-02-06",
    "1952-01-27",
    "1953-02-14",
    "1954-02-03",
    "1955-01-24",
    "1956-02-12",
    "1957-01-31",
    "1958-02-18",
    "1959-02-08",
    "1960-01-28",
    "1961-02-15",
    "1962-02-05",
    "1963-01-25",
    "1964-02-13",
    "1965-02-02",
    "1966-01-21",
    "1967-02-09",
    "1968-01-30",
    "1969-02-17",
    "1970-02-06",
    "1971-01-27",
    "1972-02-15",
    "1973-02-03",
    "1974-01-23",
    "1975-02-11",
    "1976-01-31",
    "1977-02-18",
    "1978-02-07",
    "1979-01-28",
    "1980-02-16",
    "1981-02-05",
    "1982-01-25",
    "1983-02-13",
    "1984-02-02",
    "1985-02-20",
    "1986-02-09",
    "1987-01-29",
    "1988-02-17",
    "1989-02-06",
    "1990-01-27",
    "1991-02-15",
    "1992-02-04",
    "1993-01-23",
    "1994-02-10",
    "1995-01-31",
    "1996-02-19",
    "1997-02-07",
    "1998-01-28",
    "1999-02-16",
    "2000-02-05",
    "2001-01-24",
    "2002-02-12",
    "2003-02-01",
    "2004-01-22",
    "2005-02-09",
    "2006-01-29",
    "2007-02-18",
    "2008-02-07",
    "2009-01-26",
    "2010-02-14",
    "2011-02-03",
    "2012-01-23",
    "2013-02-10",
    "2014-01-31",
    "2015-02-19",
    "2016-02-08",
    "2017-01-28",
    "2018-02-16",
    "2019-02-05",
    "2020-01-25",
    "2021-02-12",
    "2022-02-01",
    "2023-01-22",
    "2024-02-10",
    "2025-01-29",
    "2026-02-17",
    "2027-02-06",
    "2028-01-26",
    "2029-02-13",
    "2030-02-02",
    "2031-01-23",
    "2032-02-11",
    "2033-01-31",
    "2034-02-19",
    "2035-02-08",
    "2036-01-28",
    "2037-02-15",
    "2038-02-04",
    "2039-01-24",
    "2040-02-12",
    "2041-02-01",
    "2042-01-22",
    "2043-02-10",
    "2044-01-30",
    "2045-02-17",
    "2046-02-06",
    "2047-01-26",
    "2048-02-14",
    "2049-02-02",
    "2050-01-23",
    "2051-02-11",
    "2052-02-01",
    "2053-02-19",
    "2054-02-08",
    "2055-01-28",
    "2056-02-15",
    "2057-02-04",
    "2058-01-24",
    "2059-02-12",
    "2060-02-02",
    "2061-01-21",
    "2062-02-09",
    "2063-01-29",
    "2064-02-17",
    "2065-02-05",
    "2066-01-26",
    "2067-02-14",
    "2068-02-03",
    "2069-01-23",
    "2070-02-11",
    "2071-01-31",
    "2072-02-19",
    "2073-02-07",
    "2074-01-27",
    "2075-02-15",
    "2076-02-05",
    "2077-01-24",
    "2078-02-12",
    "2079-02-02",
    "2080-01-22",
    "2081-02-09",
    "2082-01-29",
    "2083-02-17",
    "2084-02-06",
    "2085-01-26",
    "2086-02-14",
    "2087-02-03",
    "2088-01-24",
    "2089-02-10",
    "2090-01-30",
    "2091-02-18",
    "2092-02-07",
    "2093-01-27",
    "2094-02-15",
    "2095-02-05",
    "2096-01-25",
    "2097-02-12",
    "2098-02-01",
    "2099-01-21",
    "2100-02-09",
]
lunar_new_year_dict = {
    k + 1800: datetime.strptime(v, "%Y-%m-%d") for k, v in enumerate(lunar_new_year_dates)
}


def lunar_new_year_holidays(year: int) -> list[datetime]:
    try:
        dt = lunar_new_year_dict[year]
    except KeyError:
        return []

    if dt.weekday() in [0, 2, 3, 4, 5, 6]:
        idx = (-1, 7)
    elif dt.weekday() in [1]:
        idx = (-2, 7)

    return [dt + timedelta(days=i) for i in range(*idx)]


def lunar_new_year_compensations(year: int) -> list[datetime]:
    try:
        dt = lunar_new_year_dict[year]
    except KeyError:
        return []

    if dt.weekday() in [0]:
        # previous and post saturday
        return [dt + timedelta(days=-2), dt + timedelta(days=12)]
    elif dt.weekday() in [1]:
        # previous and post saturday
        return [dt + timedelta(days=-3), dt + timedelta(days=11)]
    elif dt.weekday() in [2]:
        # previous Sunday and post saturday
        return [dt + timedelta(days=-3), dt + timedelta(days=10)]
    elif dt.weekday() in [3]:
        # previous Sunday and post saturday
        return [dt + timedelta(days=-4), dt + timedelta(days=9)]
    elif dt.weekday() in [4]:
        return [dt + timedelta(days=-5), dt + timedelta(days=8)]
    elif dt.weekday() in [5]:
        return [dt + timedelta(days=-6), dt + timedelta(days=7)]
    elif dt.weekday() in [6]:
        return [dt + timedelta(days=-7), dt + timedelta(days=7)]


# parsed with re: from https://taiwan-database.net/PDFs/WTFpdf23.pdf
dragon_boat_days = [
    "1900-06-01",
    "1901-06-20",
    "1902-06-10",
    "1903-05-31",
    "1904-06-18",
    "1905-06-07",
    "1906-06-26",
    "1907-06-15",
    "1908-06-03",
    "1909-06-22",
    "1910-06-11",
    "1911-06-01",
    "1912-06-19",
    "1913-06-09",
    "1914-05-29",
    "1915-06-17",
    "1916-06-05",
    "1917-06-23",
    "1918-06-13",
    "1919-06-02",
    "1920-06-20",
    "1921-06-10",
    "1922-05-31",
    "1923-06-18",
    "1924-06-06",
    "1925-06-25",
    "1926-06-14",
    "1927-06-04",
    "1928-06-22",
    "1929-06-11",
    "1930-06-01",
    "1931-06-20",
    "1932-06-08",
    "1933-05-28",
    "1934-06-16",
    "1935-06-05",
    "1936-06-23",
    "1937-06-12",
    "1938-06-02",
    "1939-06-21",
    "1940-06-10",
    "1941-05-30",
    "1942-06-18",
    "1943-06-07",
    "1944-06-25",
    "1945-06-14",
    "1946-06-04",
    "1947-06-23",
    "1948-06-11",
    "1949-06-01",
    "1950-06-19",
    "1951-06-09",
    "1952-05-28",
    "1953-06-15",
    "1954-06-05",
    "1955-06-24",
    "1956-06-13",
    "1957-06-02",
    "1958-06-21",
    "1959-06-10",
    "1960-05-29",
    "1961-06-17",
    "1962-06-06",
    "1963-06-25",
    "1964-06-14",
    "1965-06-04",
    "1966-06-23",
    "1967-06-12",
    "1968-05-31",
    "1969-06-19",
    "1970-06-08",
    "1971-05-28",
    "1972-06-15",
    "1973-06-05",
    "1974-06-24",
    "1975-06-14",
    "1976-06-02",
    "1977-06-21",
    "1978-06-10",
    "1979-05-30",
    "1980-06-17",
    "1981-06-06",
    "1982-06-25",
    "1983-06-15",
    "1984-06-04",
    "1985-06-22",
    "1986-06-11",
    "1987-05-31",
    "1988-06-18",
    "1989-06-08",
    "1990-05-28",
    "1991-06-16",
    "1992-06-05",
    "1993-06-24",
    "1994-06-13",
    "1995-06-02",
    "1996-06-20",
    "1997-06-09",
    "1998-05-30",
    "1999-06-18",
    "2000-06-06",
    "2001-06-25",
    "2002-06-15",
    "2003-06-04",
    "2004-06-22",
    "2005-06-11",
    "2006-05-31",
    "2007-06-19",
    "2008-06-08",
    "2009-05-28",
    "2010-06-16",
    "2011-06-06",
    "2012-06-23",
    "2013-06-13",
    "2014-06-02",
    "2015-06-20",
    "2016-06-09",
    "2017-05-30",
    "2018-06-18",
    "2019-06-07",
    "2020-06-25",
    "2021-06-14",
    "2022-06-03",
    "2023-06-22",
    "2024-06-10",
    "2025-05-31",
    "2026-06-19",
    "2027-06-09",
    "2028-05-28",
    "2029-06-16",
    "2030-06-05",
    "2031-06-24",
    "2032-06-12",
    "2033-06-01",
    "2034-06-20",
    "2035-06-10",
    "2036-05-30",
    "2037-06-18",
    "2038-06-07",
    "2039-05-27",
    "2040-06-14",
    "2041-06-03",
    "2042-06-22",
    "2043-06-11",
    "2044-05-31",
    "2045-06-19",
    "2046-06-08",
    "2047-05-29",
    "2048-06-15",
    "2049-06-04",
    "2050-06-23",
    "2051-06-13",
    "2052-06-01",
    "2053-06-20",
    "2054-06-10",
    "2055-05-30",
    "2056-06-17",
    "2057-06-06",
    "2058-06-25",
    "2059-06-14",
    "2060-06-03",
    "2061-06-22",
    "2062-06-11",
    "2063-06-01",
    "2064-06-19",
    "2065-06-08",
    "2066-05-28",
    "2067-06-16",
    "2068-06-04",
    "2069-06-23",
    "2070-06-13",
    "2071-06-02",
    "2072-06-20",
    "2073-06-10",
    "2074-05-30",
    "2075-06-17",
    "2076-06-06",
    "2077-06-24",
    "2078-06-14",
    "2079-06-04",
    "2080-06-22",
    "2081-06-11",
    "2082-06-01",
    "2083-06-19",
    "2084-06-07",
    "2085-05-27",
    "2086-06-15",
    "2087-06-05",
    "2088-06-23",
    "2089-06-13",
    "2090-06-02",
    "2091-06-21",
    "2092-06-09",
    "2093-05-29",
    "2094-06-17",
    "2095-06-06",
    "2096-06-24",
    "2097-06-14",
    "2098-06-04",
    "2099-06-23",
    "2100-06-12",
]
dragon_boat_dict = {
    k + 1900: datetime.strptime(v, "%Y-%m-%d") for k, v in enumerate(dragon_boat_days)
}


def three_day_holidays(dt: datetime) -> list[datetime]:
    if dt.weekday() in [3, 4]:
        # holidays on the weekend after
        return [dt + timedelta(days=i) for i in range(3)]
    elif dt.weekday() in [0, 1]:
        # holidays on the weekend before
        return [dt + timedelta(days=i) for i in range(-2, 1)]
    elif dt.weekday() in [2]:
        # only one mid-week holiday day
        return [dt + timedelta(days=i) for i in range(1)]
    elif dt.weekday() in [5]:
        # sat, sun and mon
        start = 0
    elif dt.weekday() in [6]:
        # sat, sun, mon
        start = -1
    return [dt + timedelta(days=i) for i in range(start, start + 3)]


def three_day_compensations(dt: datetime) -> list[datetime]:
    if dt.weekday() in [1]:
        # compensate by preceding Su
        return [dt + timedelta(days=-9)]
    if dt.weekday() in [3]:
        # following Su
        return [dt + timedelta(days=10)]
    else:
        return []


def dragon_boat_holidays(year: int) -> list[datetime]:
    try:
        dt = dragon_boat_dict[year]
    except KeyError:
        return []

    return three_day_holidays(dt)


def dragon_boat_compensations(year: int) -> list[datetime]:
    try:
        dt = dragon_boat_dict[year]
    except KeyError:
        return []

    return three_day_compensations(dt)


vernal_equinox_date = [
    "1970-03-21",
    "1971-03-21",
    "1972-03-20",
    "1973-03-20",
    "1974-03-21",
    "1975-03-21",
    "1976-03-20",
    "1977-03-20",
    "1978-03-20",
    "1979-03-21",
    "1980-03-20",
    "1981-03-20",
    "1982-03-20",
    "1983-03-21",
    "1984-03-20",
    "1985-03-20",
    "1986-03-20",
    "1987-03-21",
    "1988-03-20",
    "1989-03-20",
    "1990-03-20",
    "1991-03-21",
    "1992-03-20",
    "1993-03-20",
    "1994-03-20",
    "1995-03-21",
    "1996-03-20",
    "1997-03-20",
    "1998-03-20",
    "1999-03-21",
    "2000-03-20",
    "2001-03-20",
    "2002-03-20",
    "2003-03-21",
    "2004-03-20",
    "2005-03-20",
    "2006-03-20",
    "2007-03-21",
    "2008-03-20",
    "2009-03-20",
    "2010-03-20",
    "2011-03-20",
    "2012-03-20",
    "2013-03-20",
    "2014-03-20",
    "2015-03-20",
    "2016-03-21",  # Manual edit
    "2017-03-20",
    "2018-03-21",  # Manual edit
    "2019-03-20",
    "2020-03-20",
    "2021-03-21",  # Manual edit
    "2022-03-21",  # Manual edit
    "2023-03-21",  # Manual edit
    "2024-03-20",
    "2025-03-20",
    "2026-03-20",
    "2027-03-20",
    "2028-03-20",
    "2029-03-20",
    "2030-03-20",
    "2031-03-20",
    "2032-03-20",
    "2033-03-20",
    "2034-03-20",
    "2035-03-20",
    "2036-03-20",
    "2037-03-20",
    "2038-03-20",
    "2039-03-20",
    "2040-03-20",
    "2041-03-20",
    "2042-03-20",
    "2043-03-20",
    "2044-03-19",
    "2045-03-20",
    "2046-03-20",
    "2047-03-20",
    "2048-03-19",
    "2049-03-20",
    "2050-03-20",
    "2051-03-20",
    "2052-03-19",
    "2053-03-20",
    "2054-03-20",
    "2055-03-20",
    "2056-03-19",
    "2057-03-20",
    "2058-03-20",
    "2059-03-20",
    "2060-03-19",
    "2061-03-20",
    "2062-03-20",
    "2063-03-20",
    "2064-03-19",
    "2065-03-20",
    "2066-03-20",
    "2067-03-20",
    "2068-03-19",
    "2069-03-20",
    "2070-03-20",
    "2071-03-20",
    "2072-03-19",
    "2073-03-20",
    "2074-03-20",
    "2075-03-20",
    "2076-03-19",
    "2077-03-19",
    "2078-03-20",
    "2079-03-20",
    "2080-03-19",
    "2081-03-19",
    "2082-03-20",
    "2083-03-20",
    "2084-03-19",
    "2085-03-19",
    "2086-03-20",
    "2087-03-20",
    "2088-03-19",
    "2089-03-19",
    "2090-03-20",
    "2091-03-20",
    "2092-03-19",
    "2093-03-19",
    "2094-03-20",
    "2095-03-20",
    "2096-03-19",
    "2097-03-19",
    "2098-03-20",
    "2099-03-20",
    "2100-03-20",
    "2101-03-20",
    "2102-03-21",
    "2103-03-21",
    "2104-03-20",
    "2105-03-20",
    "2106-03-21",
    "2107-03-21",
    "2108-03-20",
    "2109-03-20",
    "2110-03-20",
    "2111-03-21",
    "2112-03-20",
    "2113-03-20",
    "2114-03-20",
    "2115-03-21",
    "2116-03-20",
    "2117-03-20",
    "2118-03-20",
    "2119-03-21",
    "2120-03-20",
    "2121-03-20",
    "2122-03-20",
    "2123-03-21",
    "2124-03-20",
    "2125-03-20",
    "2126-03-20",
    "2127-03-21",
    "2128-03-20",
    "2129-03-20",
    "2130-03-20",
    "2131-03-21",
    "2132-03-20",
    "2133-03-20",
    "2134-03-20",
    "2135-03-21",
    "2136-03-20",
    "2137-03-20",
    "2138-03-20",
    "2139-03-20",
    "2140-03-20",
    "2141-03-20",
    "2142-03-20",
    "2143-03-20",
    "2144-03-20",
    "2145-03-20",
    "2146-03-20",
    "2147-03-20",
    "2148-03-20",
    "2149-03-20",
    "2150-03-20",
    "2151-03-20",
    "2152-03-20",
    "2153-03-20",
    "2154-03-20",
    "2155-03-20",
    "2156-03-20",
    "2157-03-20",
    "2158-03-20",
    "2159-03-20",
    "2160-03-20",
    "2161-03-20",
    "2162-03-20",
    "2163-03-20",
    "2164-03-20",
    "2165-03-20",
    "2166-03-20",
    "2167-03-20",
    "2168-03-20",
    "2169-03-20",
    "2170-03-20",
    "2171-03-20",
    "2172-03-19",
    "2173-03-20",
    "2174-03-20",
    "2175-03-20",
    "2176-03-19",
    "2177-03-20",
    "2178-03-20",
    "2179-03-20",
    "2180-03-19",
    "2181-03-20",
    "2182-03-20",
    "2183-03-20",
    "2184-03-19",
    "2185-03-20",
    "2186-03-20",
    "2187-03-20",
    "2188-03-19",
    "2189-03-20",
    "2190-03-20",
    "2191-03-20",
    "2192-03-19",
    "2193-03-20",
    "2194-03-20",
    "2195-03-20",
    "2196-03-19",
    "2197-03-20",
    "2198-03-20",
    "2199-03-20",
    "2200-03-20",
]
vernal_equinox_dict = {
    k + 1970: datetime.strptime(v, "%Y-%m-%d") for k, v in enumerate(vernal_equinox_date)
}

mid_autumn_festival_dates = [
    "1900-09-08",
    "1901-09-27",
    "1902-09-16",
    "1903-10-05",
    "1904-09-24",
    "1905-09-13",
    "1906-10-02",
    "1907-09-22",
    "1908-09-10",
    "1909-09-28",
    "1910-09-18",
    "1911-10-06",
    "1912-09-25",
    "1913-09-15",
    "1914-10-04",
    "1915-09-23",
    "1916-09-12",
    "1917-09-30",
    "1918-09-19",
    "1919-10-08",
    "1920-09-26",
    "1921-09-16",
    "1922-10-05",
    "1923-09-25",
    "1924-09-13",
    "1925-10-02",
    "1926-09-21",
    "1927-09-10",
    "1928-09-28",
    "1929-09-17",
    "1930-10-06",
    "1931-09-26",
    "1932-09-15",
    "1933-10-04",
    "1934-09-23",
    "1935-09-12",
    "1936-09-30",
    "1937-09-18",
    "1938-10-08",
    "1939-09-27",
    "1940-09-16",
    "1941-10-05",
    "1942-09-24",
    "1943-09-14",
    "1944-10-01",
    "1945-09-20",
    "1946-09-10",
    "1947-09-29",
    "1948-09-17",
    "1949-10-06",
    "1950-09-26",
    "1951-09-15",
    "1952-10-03",
    "1953-09-22",
    "1954-09-11",
    "1955-09-30",
    "1956-09-19",
    "1957-09-08",
    "1958-09-27",
    "1959-09-17",
    "1960-10-05",
    "1961-09-24",
    "1962-09-13",
    "1963-10-02",
    "1964-09-20",
    "1965-09-10",
    "1966-09-29",
    "1967-09-18",
    "1968-10-06",
    "1969-09-26",
    "1970-09-15",
    "1971-10-03",
    "1972-09-22",
    "1973-09-11",
    "1974-09-30",
    "1975-09-20",
    "1976-09-08",
    "1977-09-27",
    "1978-09-17",
    "1979-10-05",
    "1980-09-23",
    "1981-09-12",
    "1982-10-01",
    "1983-09-21",
    "1984-09-10",
    "1985-09-29",
    "1986-09-18",
    "1987-10-07",
    "1988-09-25",
    "1989-09-14",
    "1990-10-03",
    "1991-09-22",
    "1992-09-11",
    "1993-09-30",
    "1994-09-20",
    "1995-09-09",
    "1996-09-27",
    "1997-09-16",
    "1998-10-05",
    "1999-09-24",
    "2000-09-12",
    "2001-10-01",
    "2002-09-21",
    "2003-09-11",
    "2004-09-28",
    "2005-09-18",
    "2006-10-06",
    "2007-09-25",
    "2008-09-14",
    "2009-10-03",
    "2010-09-22",
    "2011-09-12",
    "2012-09-30",
    "2013-09-19",
    "2014-09-08",
    "2015-09-27",
    "2016-09-15",
    "2017-10-04",
    "2018-09-24",
    "2019-09-13",
    "2020-10-01",
    "2021-09-21",
    "2022-09-10",
    "2023-09-29",
    "2024-09-17",
    "2025-10-06",
    "2026-09-25",
    "2027-09-15",
    "2028-10-03",
    "2029-09-22",
    "2030-09-12",
    "2031-10-01",
    "2032-09-19",
    "2033-09-08",
    "2034-09-27",
    "2035-09-16",
    "2036-10-04",
    "2037-09-24",
    "2038-09-13",
    "2039-10-02",
    "2040-09-20",
    "2041-09-10",
    "2042-09-28",
    "2043-09-17",
    "2044-10-05",
    "2045-09-25",
    "2046-09-15",
    "2047-10-04",
    "2048-09-22",
    "2049-09-11",
    "2050-09-30",
    "2051-09-19",
    "2052-09-07",
    "2053-09-26",
    "2054-09-16",
    "2055-10-05",
    "2056-09-24",
    "2057-09-13",
    "2058-10-02",
    "2059-09-21",
    "2060-09-09",
    "2061-09-28",
    "2062-09-17",
    "2063-10-06",
    "2064-09-25",
    "2065-09-15",
    "2066-10-03",
    "2067-09-23",
    "2068-09-11",
    "2069-09-29",
    "2070-09-19",
    "2071-09-08",
    "2072-09-26",
    "2073-09-16",
    "2074-10-05",
    "2075-09-24",
    "2076-09-12",
    "2077-10-01",
    "2078-09-20",
    "2079-09-10",
    "2080-09-28",
    "2081-09-17",
    "2082-10-06",
    "2083-09-26",
    "2084-09-14",
    "2085-10-03",
    "2086-09-22",
    "2087-09-11",
    "2088-09-29",
    "2089-09-18",
    "2090-09-08",
    "2091-09-27",
    "2092-09-16",
    "2093-10-05",
    "2094-09-24",
    "2095-09-13",
    "2096-09-30",
    "2097-09-20",
    "2098-09-09",
    "2099-09-29",
    "2100-09-18",
]
mid_autumn_dict = {
    k + 1900: datetime.strptime(v, "%Y-%m-%d") for k, v in enumerate(mid_autumn_festival_dates)
}


def tomb_sweeping_holidays(year: int) -> list[datetime]:
    try:
        dt = vernal_equinox_dict[year]
    except KeyError:
        return []
    # add 15 days to get to the holiday
    return three_day_holidays(dt + timedelta(days=15))


def tomb_sweeping_compensations(year: int) -> list[datetime]:
    try:
        dt = vernal_equinox_dict[year]
    except KeyError:
        return []
    # add 15 days to get to the holiday
    return three_day_compensations(dt + timedelta(days=15))


def mid_autumn_holidays(year: int) -> list[datetime]:
    try:
        dt = mid_autumn_dict[year]
    except KeyError:
        return []

    return three_day_holidays(dt)


def mid_autumn_compensations(year: int) -> list[datetime]:
    try:
        dt = mid_autumn_dict[year]
    except KeyError:
        return []

    return three_day_compensations(dt)


def labour_day_holidays(year: int) -> list[datetime]:
    # golden 5 day holiday
    dt = datetime(year=year, month=5, day=1)

    if dt.weekday() == 0:
        idx = (-2, 3)
    elif dt.weekday() == 1:
        idx = (-3, 2)
    elif dt.weekday() == 2:
        idx = (0, 2)
    elif dt.weekday() == 3 or dt.weekday() == 4 or dt.weekday() == 5:
        idx = (0, 5)
    elif dt.weekday() == 6:
        idx = (-1, 4)

    return [dt + timedelta(days=i) for i in range(*idx)]


def labour_day_compensations(year: int) -> list[datetime]:
    dt = datetime(year=year, month=5, day=1)

    if dt.weekday() == 0:
        # following Su
        return [dt + timedelta(days=6)]
    elif dt.weekday() == 1:
        # Previous Sa
        return [dt + timedelta(days=-3)]
    elif dt.weekday() in [2]:
        # no compensation for Wednesday
        return []
    elif dt.weekday() in [3]:
        # Previous Su
        return [dt + timedelta(days=-4)]
    elif dt.weekday() in [4]:
        # Following Sa
        return [dt + timedelta(days=8)]
    elif dt.weekday() in [5]:
        # Following Su
        return [dt + timedelta(days=8)]
    elif dt.weekday() in [6]:
        # Following Su
        return [dt + timedelta(days=7)]


def national_day_holidays(year: int) -> list[datetime]:
    return [datetime(year=year, month=10, day=1) + timedelta(days=i) for i in range(7)]


def national_day_compensations(year: int) -> list[datetime]:
    dt = datetime(year=year, month=10, day=1)

    if dt.weekday() == 0:
        return [dt + timedelta(days=-2), dt + timedelta(days=12)]
    if dt.weekday() == 1:
        return [dt + timedelta(days=-2), dt + timedelta(days=11)]
    if dt.weekday() == 2:
        return [dt + timedelta(days=-3), dt + timedelta(days=10)]
    if dt.weekday() == 3:
        return [dt + timedelta(days=-4), dt + timedelta(days=9)]
    if dt.weekday() == 4:
        return [dt + timedelta(days=-5), dt + timedelta(days=8)]
    if dt.weekday() == 5:
        return [dt + timedelta(days=-6), dt + timedelta(days=8)]
    if dt.weekday() == 6:
        return [dt + timedelta(days=-7), dt + timedelta(days=7)]


def national_day_and_mid_autumn_holidays(year: int) -> list[datetime]:
    # broad stroke approximations to merge National holiday and Mid Autumn overlaps
    mu_holidays = mid_autumn_holidays(year)
    nat_holidays = national_day_holidays(year)

    if mu_holidays[-1] <= nat_holidays[-1] and mu_holidays[0] >= nat_holidays[0]:
        # then mu holidays are contained within nat holidays so extend
        return nat_holidays + [datetime(year=year, month=10, day=8)]
    else:
        return mu_holidays + nat_holidays


def national_day_and_mid_autumn_compensations(year: int) -> list[datetime]:
    # broad stroke approximations to merge National holiday and Mid Autumn overlaps
    mu_holidays = mid_autumn_holidays(year)
    nat_holidays = national_day_holidays(year)
    mu_compensations = mid_autumn_compensations(year)  # can only be at most 1 date
    nat_compensations = national_day_compensations(year)  # will be 2 dates

    if len(mu_compensations) > 0 and mu_compensations in nat_holidays:
        mu_compensations = [mu_compensations[0] - timedelta(days=7)]

    if nat_compensations[0] in mu_holidays:
        nat_compensations[0] = nat_compensations[0] - timedelta(days=7)

    return mu_compensations + nat_compensations


def apply_years(years: tuple[int, int], func) -> list[datetime]:
    h = []
    for year in range(years[0], years[1] + 1):
        h.extend(func(year))
    return h


def apply_years_H(years: tuple[int, int], func) -> list[Holiday]:
    return [
        Holiday("Date: {_.strftime('%Y-%m-%d')}", year=_.year, month=_.month, day=_.day)
        for _ in apply_years(years, func)
    ]


COMPENSATIONS = [
    *apply_years((2025, 2100), new_years_compensations),
    *apply_years((2025, 2100), lunar_new_year_compensations),
    *apply_years((2025, 2100), dragon_boat_compensations),
    *apply_years((2025, 2100), tomb_sweeping_compensations),
    *apply_years((2025, 2100), labour_day_compensations),
    *apply_years((2025, 2100), national_day_and_mid_autumn_compensations),
]

RULES = [
    # these provide a custom saturday sunday weekmask but add back specific trading days at weekend
    *weekday_mask(weekdays=[5, 6], years=(1970, 2125), exclude=COMPENSATIONS),
    *apply_years_H((2025, 2100), new_years_holidays),
    *apply_years_H((2025, 2100), lunar_new_year_holidays),
    *apply_years_H((2025, 2100), dragon_boat_holidays),
    *apply_years_H((2025, 2100), tomb_sweeping_holidays),
    *apply_years_H((2025, 2100), labour_day_holidays),
    *apply_years_H((2025, 2100), national_day_and_mid_autumn_holidays),
]

CALENDAR = CustomBusinessDay(  # type: ignore[call-arg]
    calendar=AbstractHolidayCalendar(rules=RULES),
    weekmask="Mon Tue Wed Thu Fri Sat Sun",
)

### RUN THE SCRIPT TO EXPORT HOLIDAY LIST
ts = pd.to_datetime(CALENDAR.holidays)
strings = ['"' + _.strftime("%Y-%m-%d %H:%M:%S") + '"' for _ in ts]
line = ",\n".join(list(dict.fromkeys(strings)))
print(line)


# [
#             datetime(2022, 1, 29),
#             datetime(2022, 1, 30),
#             datetime(2022, 4, 2),
#             datetime(2022, 4, 24),
#             datetime(2022, 5, 7),
#             datetime(2022, 10, 8),
#             datetime(2022, 10, 9),
#             datetime(2022, 12, 31),
#             datetime(2023, 1, 28),
#             datetime(2023, 1, 29),
#             datetime(2023, 4, 23),
#             datetime(2023, 5, 6),
#             datetime(2023, 6, 25),
#             datetime(2023, 10, 7),
#             datetime(2023, 10, 8),
#             datetime(2023, 12, 31),
#             datetime(2024, 2, 4),
#             datetime(2024, 2, 18),
#             datetime(2024, 4, 7),
#             datetime(2024, 4, 28),
#             datetime(2024, 5, 11),
#             datetime(2024, 9, 14),
#             datetime(2024, 9, 29),
#             datetime(2024, 10, 12),
#             datetime(2025, 1, 26),
#             datetime(2025, 2, 8),
#             datetime(2025, 4, 27),
#             datetime(2025, 9, 28),
#             datetime(2025, 10, 11),
#             datetime(2026, 1, 4),
#             datetime(2026, 2, 14),
#             datetime(2026, 2, 28),
#             datetime(2026, 5, 9),
#             datetime(2026, 9, 20),
#             datetime(2026, 10, 10),
#         ],
