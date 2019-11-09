import csv

with open('Python.csv') as file:
    readCSV=csv.reader(file)
    for row in readCSV:
        print(row)
        print('')
