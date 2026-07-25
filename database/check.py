import sqlite3

conn = sqlite3.connect("hkjc_racing.db")

cur = conn.cursor()

res = cur.execute("SELECT * FROM race")
print(res)