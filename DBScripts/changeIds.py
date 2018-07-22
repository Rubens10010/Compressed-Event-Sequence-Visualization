import csv
import collections
import itertools

ofile = open("agavue_final.csv","w")

Events = collections.defaultdict(itertools.count().next)
ChartIds = collections.defaultdict(itertools.count().next)
Ips = collections.defaultdict(itertools.count().next)

first = True

with open('agave_processed.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=",")
	ofile.write("IP,ChartId,Action,Version,Date\n")
	for row in reader:
		if first:
			first = False
			continue
		ofile.write(str(Ips[row[0]]) + "," + str(ChartIds[row[1]]) + "," + str(Events[row[2]]) + "," + row[3] + "," + row[4] + "\n" )

ofile.close()
print "Saved Agavue"
ofile = open("Eventos.csv","w")
ofile.write("Id, EventName\n")
for k,v in Events.iteritems():
	ofile.write(str(v) + "," + str(k) + "\n")
	
ofile.close()
print "Saved Events"
ofile = open("ChartIds.csv","w")
ofile.write("Id, ChartId\n")
for k,v in ChartIds.iteritems():
	ofile.write(str(v) + "," + str(k)+ "\n")

ofile.close()
print "Saved ChartIds"
ofile = open("Ips.csv","w")
ofile.write("Id, IP\n")
for k,v in Ips.iteritems():
	ofile.write(str(v) + "," + str(k)+ "\n")
ofile.close()

print "Saved Ips"
