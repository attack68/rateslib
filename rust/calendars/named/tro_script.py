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
    Holiday("Family Day", month=2, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Victoria Day", month=5, day=24, offset=DateOffset(weekday=MO(-1))),
    Holiday("Civic Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday("CAD Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday("CAD Thanksgiving", month=10, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Canada Day", month=7, day=1, observance=next_monday),
    Holiday("Remembrance", month=11, day=11, observance=next_monday),
    Holiday("National Truth & Reconciliation", month=9, day=30, start_date=datetime(2021, 1, 1)),
    Holiday("Christmas Day Holiday", month=12, day=25, observance=next_monday),
    Holiday("Boxing Day Holiday", month=12, day=26, observance=next_monday_or_tuesday),
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
