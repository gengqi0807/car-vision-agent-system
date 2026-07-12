import sqlite3
from datetime import datetime

DB = 'dev.db'

def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    try:
        cur.execute("SELECT id, source, event_type, title, summary, level, status, confidence, created_at FROM monitor_logs ORDER BY created_at DESC LIMIT 50")
        rows = cur.fetchall()
        if not rows:
            print('No monitor logs found in dev.db')
            return
        for r in rows:
            # created_at stored as text; try to print readable
            print(r)
    finally:
        con.close()

if __name__ == '__main__':
    main()
