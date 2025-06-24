from datetime import datetime

import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Day,
    Easter,
    Holiday,
    next_monday,
    next_monday_or_tuesday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

matariki_dates = [
    "2022-06-24",
    "2023-07-14",
    "2024-06-28",
    "2025-06-20",
    "2026-07-10",
    "2027-06-25",
    "2028-07-14",
    "2029-07-06",
    "2030-06-21",
    "2031-07-11",
    "2032-07-02",
    "2033-06-24",
    "2034-07-07",
    "2035-06-29",
    "2036-07-18",
    "2037-07-10",
    "2038-06-25",
    "2039-07-15",
    "2040-07-06",
    "2041-07-19",
    "2042-07-11",
    "2043-07-03",
    "2044-06-24",
    "2045-07-07",
    "2046-06-29",
    "2047-07-19",
    "2048-07-03",
    "2049-06-25",
    "2050-07-15",
    "2051-06-30",
    "2052-06-21",
]

matariki_dict = {k + 2022: datetime.strptime(v, "%Y-%m-%d") for k, v in enumerate(matariki_dates)}


def matariki_hol(dt: datetime) -> datetime:
    try:
        dt = matariki_dict[dt.year]
    except KeyError:
        return datetime(1900, 1, 1)
    return dt


RULES = [
    Holiday("New Year's Day", month=1, day=1),
    Holiday("Day After New Year's Day", month=1, day=2),
    Holiday("Waitangi Day", month=2, day=6, observance=next_monday),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)]),
    Holiday("Anzac Day", month=4, day=25),
    Holiday("King's Birthday", month=6, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Matariki", month=1, day=1, observance=matariki_hol),
    Holiday("Labour Day", month=10, day=31, offset=DateOffset(weekday=MO(-1))),
    Holiday("Christmas Day Holiday", month=12, day=25, observance=next_monday),
    Holiday("Boxing Day Holiday", month=12, day=26, observance=next_monday_or_tuesday),
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
