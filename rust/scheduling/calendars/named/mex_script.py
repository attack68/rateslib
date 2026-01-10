import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset, Day, Easter

RULES = [
    Holiday("New Year's Day Holiday", month=1, day=1),
    Holiday("Constitution Day", month=2, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday("Birth of Benito Juarez", month=3, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Maundy Thursday", month=1, day=1, offset=[Easter(), Day(-3)]),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Labor Day", month=5, day=1),
    Holiday("Independence Day", month=9, day=16),
    Holiday("All Souls' Day", month=11, day=2),
    Holiday("Revolution Day", month=11, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Day of the Virgin of Guadalupe", month=12, day=12),
    Holiday("Christmas Day", month=12, day=25),
    *[Holiday(f"Election {y}", year=y, month=10, day=1) for y in range(2024, 2075, 6)],
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
