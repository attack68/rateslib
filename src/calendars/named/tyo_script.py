from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset


def sunday_to_monday_or_tuesday(dt: datetime) -> datetime:
    """
    Used for Greenery Day
    If holiday falls on Sunday, use following Monday instead;
    if holiday falls on Monday, use the following Tuesday;
    """
    dow = dt.weekday()
    if dow in (6, 0):
        return dt + timedelta(1)
    return dt


def sunday_to_monday_or_tuesday_or_wednesday(dt: datetime) -> datetime:
    """
    Used for Children's Day
    If holiday falls on Sunday, use following Monday instead;
    if holiday falls on Monday, use the following Tuesday;
    if holiday falls on Tuesday, use the following Wednesday
    """
    dow = dt.weekday()
    if dow in (6, 0, 1):
        return dt + timedelta(1)
    return dt


RULES = [
    Holiday("New Year's Day", month=1, day=1),
    Holiday("New Year's Bank holiday", month=1, day=2),
    Holiday("New Year's Bank holiday2", month=1, day=3),
    Holiday("Coming-of-Age Day", month=1, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Foundation Day", month=2, day=11, observance=sunday_to_monday),
    Holiday("Emperor's Birthday", month=2, day=23, observance=sunday_to_monday),
    Holiday("Vernal Equinox Day", month=3, day=20, observance=sunday_to_monday),
    Holiday("Showa Day", month=4, day=29, observance=sunday_to_monday),
    Holiday("Constitution Day", month=5, day=3, observance=sunday_to_monday),
    Holiday("Greenery Day", month=5, day=4, observance=sunday_to_monday_or_tuesday),
    Holiday("Children's Day", month=5, day=5, observance=sunday_to_monday_or_tuesday_or_wednesday),
    Holiday("Marine Day", month=7, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Mountain Day", month=8, day=11, observance=sunday_to_monday),
    Holiday("Respect the Aged Day", month=9, day=1, offset=DateOffset(weekday=MO(3))),
    Holiday("Autumn Equinox Day", month=9, day=23, observance=sunday_to_monday),
    Holiday("Health and Sports Day", month=10, day=1, offset=DateOffset(weekday=MO(2))),
    Holiday("Culture Day", month=11, day=3, observance=sunday_to_monday),
    Holiday("Labor Thanksgiving Day", month=11, day=23, observance=sunday_to_monday),
    Holiday("End of Year", month=12, day=31),
]

CALENDAR = CustomBusinessDay(  # type: ignore[call-arg]
    calendar=AbstractHolidayCalendar(rules=RULES),
    weekmask="Mon Tue Wed Thu Fri",
)

### RUN THE SCRIPT TO EXPORT HOLIDAY LIST
ts = pd.to_datetime(CALENDAR.holidays)
strings = ['"'+_.strftime("%Y-%m-%d %H:%M:%S")+'"' for _ in ts]
line = ",\n".join(strings)
print(line)
