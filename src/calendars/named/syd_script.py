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

RULES = [
    Holiday("New Year's Day", month=1, day=1, observance=next_monday),
    Holiday("Australia Day", month=1, day=26, observance=next_monday),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)]),
    Holiday("Anzac Day", month=4, day=25),
    Holiday("King's Birthday", month=6, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Christmas Day Holiday", month=12, day=25, observance=next_monday),
    Holiday("Boxing Day Holiday", month=12, day=26, observance=next_monday_or_tuesday),
    # One Off
    Holiday("Memorial", year=2022, month=9, day=22),
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
