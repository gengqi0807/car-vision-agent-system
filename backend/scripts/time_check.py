import datetime
import time
import os

print('Get-Date equivalent:')
print(datetime.datetime.now().isoformat(sep=' '))
print('UTC now:')
print(datetime.datetime.utcnow().isoformat(sep=' '))
print('time.tzname:', time.tzname)
print('time.timezone:', time.timezone)
print('TZ env:', os.environ.get('TZ'))
try:
    import tzlocal
    print('tzlocal:', tzlocal.get_localzone())
except Exception:
    pass
