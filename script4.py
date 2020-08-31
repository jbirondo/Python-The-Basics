import sqlite3

def create_table():
    conn = sqlite3.connect("lite.db")
    cur= conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS store (item TEXT, quantity INTEGER, price REAL)")
    conn.commit()
    conn.close()

def insert():
    conn = sqlite3.connect("lite.db")
    cur= conn.cursor()
    cur.execute("INSERT INTO store VALUES ('Wine Glass', 8, 10.5)")
    conn.commit()
    conn.close()
