import csv, sqlite3

con = sqlite3.connect("userInteractions.db")

cur = con.cursor()

"""
cur.execute("DROP TABLE IF EXISTS Interactions")
cur.execute("CREATE TABLE Interactions (IP, ChartId, Action, Version, Date);") # use your column names here

with open('agave_processed.csv','rb') as fin: # `with` statement available in 2.5+
    # csv.DictReader uses first line in file for column headings by default
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['IP'], i['ChartId'], i['Action'], i['Version'], i['Date']) for i in dr]

cur.executemany("INSERT INTO Interactions VALUES (?, ?, ?, ?, ?);", to_db)
"""

cur.execute("DROP TABLE IF EXISTS Ips")
cur.execute('CREATE TABLE Ips (\
				Id INTEGER PRIMARY KEY AUTOINCREMENT,\
				Ip char(100));')

cur.execute('SELECT DISTINCT IP FROM Interactions;')
Ips = cur.fetchall()
cur.executemany('INSERT INTO Ips (Ip) VALUES (?)', Ips)

cur.execute("DROP TABLE IF EXISTS ChartIds")
cur.execute('CREATE TABLE ChartIds (\
				Id INTEGER PRIMARY KEY AUTOINCREMENT,\
				ChartId char(100));')

cur.execute('SELECT DISTINCT ChartId FROM Interactions;')
ChartIds = cur.fetchall()
cur.executemany('INSERT INTO ChartIds (ChartId) VALUES (?)', ChartIds)

cur.execute("DROP TABLE IF EXISTS Events")
cur.execute('CREATE TABLE Events (\
				Id INTEGER PRIMARY KEY AUTOINCREMENT,\
				EventName char(100));')

cur.execute('SELECT DISTINCT Action FROM Interactions;')
ChartIds = cur.fetchall()
cur.executemany('INSERT INTO Events (EventName) VALUES (?)', ChartIds)

### change IDs
cur.execute('UPDATE Interactions I, Actions A SET I.Action = A.Id WHERE I.Action = A.EventName;')



con.commit()
con.close()
