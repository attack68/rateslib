from datetime import datetime

import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset, Day, Easter

RULES = [
    Holiday("New Year's Day Holiday", month=1, day=1, observance=sunday_to_monday),
    Holiday(
        "Dr. Martin Luther King Jr.",
        start_date=datetime(1986, 1, 1),
        month=1,
        day=1,
        offset=DateOffset(weekday=MO(3)),
    ),
    Holiday("US President" "s Day", month=2, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("US Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
    Holiday(
        "Juneteenth Independence Day",
        start_date=datetime(2022, 1, 1),
        month=6,
        day=19,
        observance=sunday_to_monday,
    ),
    Holiday("US Independence Day", month=7, day=4, observance=nearest_workday),
    Holiday("US Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday("US Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Veterans Day", month=11, day=11, observance=sunday_to_monday),
    Holiday("US Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
    Holiday("Christmas Day Sunday Holiday", month=12, day=25, observance=nearest_workday),
    Holiday("GHW Bush Funeral", year=2018, month=12, day=5),
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
