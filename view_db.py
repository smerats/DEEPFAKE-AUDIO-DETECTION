import sqlite3

conn = sqlite3.connect("audio_results.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM results")
rows = cursor.fetchall()

print("\nDATABASE CONTENT:\n")

for row in rows:
    print(row)

conn.close()