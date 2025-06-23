import pandas as pd
from pandas.tseries.holiday import FR, AbstractHolidayCalendar, DateOffset, Holiday
from pandas.tseries.offsets import CustomBusinessDay, Day, Easter

RULES = [
    Holiday("New Year's Day", month=1, day=1),
    Holiday("Epiphany", month=1, day=6),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)]),
    Holiday("EU Labour Day", month=5, day=1),
    Holiday("Ascention Day", month=1, day=1, offset=[Easter(), Day(39)]),
    Holiday("Sweden National Day", month=6, day=6),
    Holiday("Swedish Midsummer", month=6, day=25, offset=DateOffset(weekday=FR(-1))),
    Holiday("Christmas Eve", month=12, day=24),
    Holiday("Christmas Day", month=12, day=25),
    Holiday("Boxing Day", month=12, day=26),
    Holiday("New Year's Eve", month=12, day=31),
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
