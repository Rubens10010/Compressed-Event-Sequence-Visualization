import csv

ofile = open("agave_processed.csv","w")

with open('agave_full.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter="\t")
	for row in reader:
		ofile.write(row[0] + "," + row[1] + "," + row[2] + "," + row[3] + "," + row[4] + "\n" )

ofile.close()
