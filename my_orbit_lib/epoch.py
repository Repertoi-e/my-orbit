import math
from datetime import datetime

import urllib.request

with open('leap-seconds.list') as file:
    leap_seconds_data = [l for l in [line.strip() for line in file.readlines()] if len(l) and l[0] != '#']

LEAP_SECONDS_TABLE = {}
for ntp, seconds in [l.split()[:2] for l in leap_seconds_data]:
    LEAP_SECONDS_TABLE[int(ntp) / 86400 + 15020 + 2400000.5] = int(seconds)

def get_leap_seconds_from_jd(jd):
    jds_sorted = sorted(LEAP_SECONDS_TABLE.keys())

    # First test the extremes 
    if jd <= jds_sorted[0]:
        return 0
    if jd > jds_sorted[-1]:
        return LEAP_SECONDS_TABLE[jds_sorted[-1]]
    
    # Get the last leap second increment.
    # Visible in the jupyter notebook but the table
    # basically contains a map from date to the cumulative 
    # number of leap seconds  e.g. 1 Jan 2017 -> 37
    idx = 0
    while jd > jds_sorted[idx]:
        idx += 1
    return LEAP_SECONDS_TABLE[jds_sorted[idx - 1]]

class Epoch:
    def __init__(self, *args, **kwargs):
        """
        Takes a JDE or a datetime.
        Internal time value is stored as a Julian Ephemeris Day (a float).

        You can pass in either a float or a datetime,
        the float we treat as JDE and the datetime we convert 
        first to terrestrial time (TT), then to JDE. 

        'datetime' is treated as UTC unless it's before 1972.
        If you want to deal with dates before that you should pass
        utc=False as an argument and assume it'll treat the date 
        as if it's in TT.

        Converting from UTC to TT requires some corrections:
        
        1. Taking into account all of the leap seconds from July 1972
        until today (calculated by using an up-to-date online file, but 
        for dates in the future it's unknown). The resulting time is in 
        TAI (atomic time).
        
        2. Assuming TAI has been free of defects since it read 1977-01-01T00:00:00 
        then to convert TAI to TT an addition of 32.184s is needed.
        Source: https://www.ucolick.org/~sla/leapsecs/timescales.html
        """
        self.jde = 0
        if len(args) != 0:
            year, month, day = 0, 0, 0
            if isinstance(args[0], datetime):
                d = args[0]
                year, month, day, hours, minutes, sec = (
                    d.year,
                    d.month,
                    d.day,
                    d.hour,
                    d.minute,
                    d.second + d.microsecond / 1e6
                )
                # Convert hms to fractional day
                day += hours / 24. + minutes / 1440. + sec / 86400.

                jd = date_to_jd(year, month, day)

                utc_to_tt = None
                if 'utc' in kwargs: utc_to_tt = kwargs['utc']
                
                if year < 1972:
                    if utc_to_tt is not None:
                        raise ValueError("Can't do UTC before 1972. Pass utc=False as an argument and it'll treat the date as in TT (terrestrial time).")
                    utc_to_tt = False

                if utc_to_tt is None:
                    utc_to_tt = True

                delta_sec = 0
                if utc_to_tt:
                    delta_sec += 32.184
                    delta_sec += get_leap_seconds_from_jd(jd)
            
                self.jde = jd + delta_sec / 86400.
            elif isinstance(args[0], (int, float)):
                self.jde = float(args[0])
                return
            else:
                raise ValueError("Can't convert from " + str(args[0]))

        # Complain about extra args
        allowed_keywords = ['utc']
        for k, v in kwargs.items():
            if k not in allowed_keywords:
                raise ValueError('Unknown keyword arg ' + str(k) + '=' + str(v))

    def todatetime(self):
        year, month, day = jd_to_date(self.jde)
    
        frac_days, day = math.modf(day)
        day = int(day)
    
        hour, min, sec, micro = days_to_hmsm(frac_days)
        return datetime(year, month, day, hour, min, sec, micro)

    def __str__(self):
        return str(self.jde)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.jde)

def is_date_in_julian_calendar(year, month, day):
    # Pope Gregory XIII introduced the Gregorian calendar in 1582
    if (
        (year < 1582)
        or (year == 1582 and month < 10)
        or (year == 1582 and month == 10 and day < 5.0)
    ):
        return True

    if year >= 1926: # Turkey was the last to switch to the Gregorian calendar on 1 Jan
        return False 
    if year < 1926: 
        raise ValueError("Trying to convert the date " + 
                         str(datetime(year, month, day)) + " "
                         "but we don't know if it's in the Julian or Gregorian " +
                         "calendar, because that's country specific.")

def date_to_jd(year, month, day):
    """
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
        4th ed., Duffet-Smith and Zwart, 2011.

    Found the book in an online library 
    and checked the algorithm. It's 1:1.
    """
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    julian = is_date_in_julian_calendar(year, month, day)
    if julian is not None and julian:
        B = 0
    else:
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
        
    D = math.trunc(30.6001 * (monthp + 1))
    
    jde = B + C + D + day + 1720994.5
    return jde

def jd_to_date(jd):
    """
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
        4th ed., Duffet-Smith and Zwart, 2011.

    Found the book in an online library 
    and checked the algorithm. It's 1:1.
    """
    jd = jd + 0.5
    
    F, I = math.modf(jd)
    I = int(I)
    
    A = math.trunc((I - 1867216.25)/36524.25)
    
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I
        
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    
    day = C - E + F - math.trunc(30.6001 * G)
    
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
        
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
        
    return year, month, day

def days_to_hmsm(days):
    hours = days * 24.
    hours, hour = math.modf(hours)
    
    mins = hours * 60.
    mins, min = math.modf(mins)
    
    secs = mins * 60.
    secs, sec = math.modf(secs)
    
    micro = round(secs * 1.e6)
    
    return int(hour), int(min), int(sec), int(micro)
