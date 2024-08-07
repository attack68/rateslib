from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset


def sunday_to_monday_or_tuesday(dt: datetime) -> datetime:
    """
    Used for Greenery Day
    If holiday falls on Sunday, use following Monday instead;
    if holiday falls on Monday, use the following Tuesday;
    """
    dow = dt.weekday()
    if dow in (6, 0):
        return dt + timedelta(1)
    return dt


def sunday_to_monday_or_tuesday_or_wednesday(dt: datetime) -> datetime:
    """
    Used for Children's Day
    If holiday falls on Sunday, use following Monday instead;
    if holiday falls on Monday, use the following Tuesday;
    if holiday falls on Tuesday, use the following Wednesday
    """
    dow = dt.weekday()
    if dow in (6, 0, 1):
        return dt + timedelta(1)
    return dt


# # The below list of equinoxes were exported using the python library Ephem
# import ephem
# for i in range(1970, 2201):
#     dt = ephem.next_equinox(f"{i}/1/1")
#     print(f"'{dt.datetime().strftime('%Y-%m-%d')}',")

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

autumn_equinox_date = [
    "1970-09-23",
    "1971-09-23",
    "1972-09-22",
    "1973-09-23",
    "1974-09-23",
    "1975-09-23",
    "1976-09-22",
    "1977-09-23",
    "1978-09-23",
    "1979-09-23",
    "1980-09-22",
    "1981-09-23",
    "1982-09-23",
    "1983-09-23",
    "1984-09-22",
    "1985-09-23",
    "1986-09-23",
    "1987-09-23",
    "1988-09-22",
    "1989-09-23",
    "1990-09-23",
    "1991-09-23",
    "1992-09-22",
    "1993-09-23",
    "1994-09-23",
    "1995-09-23",
    "1996-09-22",
    "1997-09-22",
    "1998-09-23",
    "1999-09-23",
    "2000-09-22",
    "2001-09-22",
    "2002-09-23",
    "2003-09-23",
    "2004-09-22",
    "2005-09-22",
    "2006-09-23",
    "2007-09-23",
    "2008-09-22",
    "2009-09-22",
    "2010-09-23",
    "2011-09-23",
    "2012-09-22",
    "2013-09-22",
    "2014-09-23",
    "2015-09-23",
    "2016-09-22",
    "2017-09-23",  # Manual edit
    "2018-09-24",  # Manual edit
    "2019-09-23",
    "2020-09-22",
    "2021-09-23",  # Manual edit
    "2022-09-23",
    "2023-09-23",
    "2024-09-22",
    "2025-09-22",
    "2026-09-23",
    "2027-09-23",
    "2028-09-22",
    "2029-09-22",
    "2030-09-22",
    "2031-09-23",
    "2032-09-22",
    "2033-09-22",
    "2034-09-22",
    "2035-09-23",
    "2036-09-22",
    "2037-09-22",
    "2038-09-22",
    "2039-09-23",
    "2040-09-22",
    "2041-09-22",
    "2042-09-22",
    "2043-09-23",
    "2044-09-22",
    "2045-09-22",
    "2046-09-22",
    "2047-09-23",
    "2048-09-22",
    "2049-09-22",
    "2050-09-22",
    "2051-09-23",
    "2052-09-22",
    "2053-09-22",
    "2054-09-22",
    "2055-09-23",
    "2056-09-22",
    "2057-09-22",
    "2058-09-22",
    "2059-09-23",
    "2060-09-22",
    "2061-09-22",
    "2062-09-22",
    "2063-09-22",
    "2064-09-22",
    "2065-09-22",
    "2066-09-22",
    "2067-09-22",
    "2068-09-22",
    "2069-09-22",
    "2070-09-22",
    "2071-09-22",
    "2072-09-22",
    "2073-09-22",
    "2074-09-22",
    "2075-09-22",
    "2076-09-22",
    "2077-09-22",
    "2078-09-22",
    "2079-09-22",
    "2080-09-22",
    "2081-09-22",
    "2082-09-22",
    "2083-09-22",
    "2084-09-22",
    "2085-09-22",
    "2086-09-22",
    "2087-09-22",
    "2088-09-22",
    "2089-09-22",
    "2090-09-22",
    "2091-09-22",
    "2092-09-21",
    "2093-09-22",
    "2094-09-22",
    "2095-09-22",
    "2096-09-21",
    "2097-09-22",
    "2098-09-22",
    "2099-09-22",
    "2100-09-22",
    "2101-09-23",
    "2102-09-23",
    "2103-09-23",
    "2104-09-22",
    "2105-09-23",
    "2106-09-23",
    "2107-09-23",
    "2108-09-22",
    "2109-09-23",
    "2110-09-23",
    "2111-09-23",
    "2112-09-22",
    "2113-09-23",
    "2114-09-23",
    "2115-09-23",
    "2116-09-22",
    "2117-09-23",
    "2118-09-23",
    "2119-09-23",
    "2120-09-22",
    "2121-09-22",
    "2122-09-23",
    "2123-09-23",
    "2124-09-22",
    "2125-09-22",
    "2126-09-23",
    "2127-09-23",
    "2128-09-22",
    "2129-09-22",
    "2130-09-23",
    "2131-09-23",
    "2132-09-22",
    "2133-09-22",
    "2134-09-23",
    "2135-09-23",
    "2136-09-22",
    "2137-09-22",
    "2138-09-23",
    "2139-09-23",
    "2140-09-22",
    "2141-09-22",
    "2142-09-23",
    "2143-09-23",
    "2144-09-22",
    "2145-09-22",
    "2146-09-23",
    "2147-09-23",
    "2148-09-22",
    "2149-09-22",
    "2150-09-23",
    "2151-09-23",
    "2152-09-22",
    "2153-09-22",
    "2154-09-22",
    "2155-09-23",
    "2156-09-22",
    "2157-09-22",
    "2158-09-22",
    "2159-09-23",
    "2160-09-22",
    "2161-09-22",
    "2162-09-22",
    "2163-09-23",
    "2164-09-22",
    "2165-09-22",
    "2166-09-22",
    "2167-09-23",
    "2168-09-22",
    "2169-09-22",
    "2170-09-22",
    "2171-09-23",
    "2172-09-22",
    "2173-09-22",
    "2174-09-22",
    "2175-09-23",
    "2176-09-22",
    "2177-09-22",
    "2178-09-22",
    "2179-09-23",
    "2180-09-22",
    "2181-09-22",
    "2182-09-22",
    "2183-09-23",
    "2184-09-22",
    "2185-09-22",
    "2186-09-22",
    "2187-09-22",
    "2188-09-22",
    "2189-09-22",
    "2190-09-22",
    "2191-09-22",
    "2192-09-22",
    "2193-09-22",
    "2194-09-22",
    "2195-09-22",
    "2196-09-22",
    "2197-09-22",
    "2198-09-22",
    "2199-09-22",
    "2200-09-23",
]
autumn_equinox_dict = {
    k + 1970: datetime.strptime(v, "%Y-%m-%d") for k, v in enumerate(autumn_equinox_date)
}


def vernal_equinox_sun_to_mon(dt: datetime) -> datetime:
    try:
        dt = vernal_equinox_dict[dt.year]
    except KeyError:
        return datetime(1900, 1, 1)
    if dt.weekday == 6:
        return dt + timedelta(1)
    return dt


def autumn_equinox_sun_to_mon(dt: datetime) -> datetime:
    try:
        dt = autumn_equinox_dict[dt.year]
    except KeyError:
        return datetime(1900, 1, 1)
    if dt.weekday == 6:
        return dt + timedelta(1)
    return dt


RULES = [
    Holiday("New Year's Day", month=1, day=1),
    Holiday("New Year's Bank holiday", month=1, day=2),
    Holiday("New Year's Bank holiday2", month=1, day=3),
    Holiday("Coming-of-Age Day", month=1, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Foundation Day", month=2, day=11, observance=sunday_to_monday),
    Holiday(
        "Emperor Naruhito Birthday",
        month=2,
        day=23,
        observance=sunday_to_monday,
        start_date=datetime(2020, 1, 1),
    ),
    Holiday("Vernal Equinox Day", month=3, day=20, observance=vernal_equinox_sun_to_mon),
    Holiday("Showa Day", month=4, day=29, observance=sunday_to_monday),
    Holiday("Constitution Day", month=5, day=3, observance=sunday_to_monday),
    Holiday("Greenery Day", month=5, day=4, observance=sunday_to_monday_or_tuesday),
    Holiday("Children's Day", month=5, day=5, observance=sunday_to_monday_or_tuesday_or_wednesday),
    Holiday(
        "Marine Day: Pre olympics",
        month=7,
        day=1,
        offset=DateOffset(weekday=MO(3)),
        end_date=datetime(2019, 12, 31),
    ),
    Holiday(
        "Marine Day: Post olympics",
        month=7,
        day=1,
        offset=DateOffset(weekday=MO(3)),
        start_date=datetime(2022, 1, 1),
    ),
    Holiday(
        "Mountain Day: Pre olympics",
        month=8,
        day=11,
        observance=sunday_to_monday,
        start_date=datetime(2016, 1, 1),
        end_date=datetime(2019, 12, 31),
    ),
    Holiday(
        "Mountain Day: Post olympics",
        month=8,
        day=11,
        observance=sunday_to_monday,
        start_date=datetime(2022, 1, 1),
    ),
    Holiday("Respect the Aged Day", month=9, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Autumn Equinox Day", month=9, day=23, observance=autumn_equinox_sun_to_mon),
    Holiday(
        "Sports Day: Pre Olympics",
        month=10,
        day=1,
        offset=DateOffset(weekday=MO(2)),
        end_date=datetime(2019, 12, 31),
    ),
    Holiday(
        "Sports Day: Post olympics",
        month=10,
        day=1,
        offset=DateOffset(weekday=MO(2)),
        start_date=datetime(2022, 1, 1),
    ),
    Holiday("Culture Day", month=11, day=3, observance=sunday_to_monday),
    Holiday("Labor Thanksgiving Day", month=11, day=23, observance=sunday_to_monday),
    Holiday(
        "Emperor Akihito Birthday",
        month=12,
        day=23,
        observance=sunday_to_monday,
        end_date=datetime(2019, 1, 1),
    ),
    Holiday("End of Year", month=12, day=31),
    # One off
    Holiday("New Emperor Ascention", year=2019, month=5, day=1),
    Holiday("Marine Day: During olympics", year=2020, month=7, day=23),
    Holiday("Marine Day: During paralympics", year=2021, month=7, day=22),
    Holiday("Mountain Day: During olmpypics", year=2020, month=8, day=10),
    Holiday("Mountain Day: During paralympics", year=2021, month=8, day=9),
    Holiday("Sports Day: Olympics", year=2020, month=7, day=24),
    Holiday("Sports Day: Paralympics", year=2021, month=7, day=23),
    Holiday("Citizens Holiday 2015", year=2015, month=9, day=22),
    Holiday("Citizens Holiday 2019", year=2019, month=4, day=30),
    Holiday("Citizens Holiday2 2019", year=2019, month=5, day=2),
    Holiday("New Emperor coronation", year=2019, month=10, day=22),
]

CALENDAR = CustomBusinessDay(  # type: ignore[call-arg]
    calendar=AbstractHolidayCalendar(rules=RULES),
    weekmask="Mon Tue Wed Thu Fri",
)

### RUN THE SCRIPT TO EXPORT HOLIDAY LIST
ts = pd.to_datetime(CALENDAR.holidays)
strings = ['"' + _.strftime("%Y-%m-%d %H:%M:%S") + '"' for _ in ts]
line = ",\n".join(strings)
print(line)
