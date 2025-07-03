from datetime import datetime

import pandas as pd
from pandas.tseries.holiday import (
    MO,
    AbstractHolidayCalendar,
    DateOffset,
    Holiday,
    next_monday,
    next_monday_or_tuesday,
)
from pandas.tseries.offsets import CustomBusinessDay, Day, Easter

RULES = [
    Holiday("New Year's Day Holiday", month=1, day=1, observance=next_monday),
    Holiday(
        "Family Day",
        month=2,
        day=1,
        offset=DateOffset(weekday=MO(3)),
        start_date=datetime(2008, 1, 1),
    ),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Victoria Day", month=5, day=24, offset=DateOffset(weekday=MO(-1))),
    Holiday("Canada Day", month=7, day=1, observance=next_monday),
    Holiday("Civic Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday("CAD Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday("CAD Thanksgiving", month=10, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Remembrance", month=11, day=11, observance=next_monday),
    Holiday("National Truth & Reconciliation", month=9, day=30, start_date=datetime(2021, 1, 1)),
    Holiday("Christmas Day Holiday", month=12, day=25, observance=next_monday),
    Holiday("Boxing Day Holiday", month=12, day=26, observance=next_monday_or_tuesday),
    # Ad hoc dates
    Holiday("adhoc1", year=1997, month=8, day=13),
    Holiday("adhoc2", year=1997, month=8, day=14),
    Holiday("adhoc3", year=1997, month=8, day=15),
    Holiday("adhoc4", year=1997, month=8, day=29),
    Holiday("adhoc5", year=1997, month=12, day=22),
    Holiday("adhoc6", year=1998, month=4, day=9),
    Holiday("adhoc7", year=1998, month=4, day=29),
]

CALENDAR = CustomBusinessDay(
    calendar=AbstractHolidayCalendar(rules=RULES),
    weekmask="Mon Tue Wed Thu Fri",
)

### RUN THE SCRIPT TO EXPORT HOLIDAY LIST
ts = pd.to_datetime(CALENDAR.holidays)
strings = ['"' + _.strftime("%Y-%m-%d %H:%M:%S") + '"' for _ in ts]
line = ",\n".join(strings)
print(line)
