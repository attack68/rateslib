from datetime import datetime

import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    next_monday,
    next_monday_or_tuesday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset, Day, Easter

RULES = [
    Holiday("New Year's Day Holiday", month=1, day=1, observance=next_monday),
    Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)]),
    Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)]),
    Holiday("UK Early May Bank Holiday", month=5, day=1, offset=DateOffset(weekday=MO(1))),
    Holiday(
        "UK Spring Bank Holiday pre 2022",
        end_date=datetime(2022, 5, 1),
        month=5,
        day=31,
        offset=DateOffset(weekday=MO(-1)),
    ),
    Holiday(
        "UK Spring Bank Holiday post 2022",
        start_date=datetime(2022, 7, 1),
        month=5,
        day=31,
        offset=DateOffset(weekday=MO(-1)),
    ),
    Holiday("Queen Elizabeth II Jubilee Thu", year=2022, month=6, day=2),
    Holiday("Queen Elizabeth II Jubilee Fri", year=2022, month=6, day=3),
    Holiday("Queen Elizabeth II Funeral", year=2022, month=9, day=19),
    Holiday("King Charles III Coronation", year=2023, month=5, day=8),
    Holiday("UK Summer Bank Holiday", month=8, day=31, offset=DateOffset(weekday=MO(-1))),
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
