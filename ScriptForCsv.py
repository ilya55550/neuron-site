import csv

res = []

with open('ListingSecurityList.csv') as f:
    reader = csv.reader(f)
    count_id = 0
    for row in reader:
        if row[0] == 'Акции':
            res.append([count_id] + [str(row[i]) for i in range(1, 3)])
            count_id += 1

with open('res.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(res)

    """заходим в psql"""
    """\COPY таблица FROM /home/ilya/pycharmprojects/neuron_project/res.csv DELIMITER ‘,’ CSV HEADER;"""
