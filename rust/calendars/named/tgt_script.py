import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
)
from pandas.tseries.offsets import CustomBusinessDay, Day, Easter

RULES = [
    Holiday("New Year's Day", month=1, day=1),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)]),
    Holiday("EU Labour Day", month=5, day=1),
    Holiday("Christmas Day", month=12, day=25),
    Holiday("Boxing Day", month=12, day=26),
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
