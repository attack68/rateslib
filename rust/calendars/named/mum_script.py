from datetime import datetime

import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    sunday_to_monday,
    Easter,
    Day
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

RULES = [
    # Days defined by the national stock exchange
    Holiday("Republic Day", month=1, day=26),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Ambedkar Jayanti", month=4, day=14),
    Holiday("May Day", month=5, day=1),
    Holiday("Independence Day", month=8, day=15),
    Holiday("Gandhi Jayanti", month=10, day=2),
    Holiday("Christmas Day", month=12, day=25),
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
