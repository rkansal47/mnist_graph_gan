import numpy as np
from os import listdir
import csv

dir = '/graphganvol/mnist_superpixels/'

csv_columns = ['Model', "Minimum FID", "Epoch"]

dict_data = []

for f in listdir(dir):
    full_path = dir + 'losses/' + f
    print(f)
    try:
        fid = np.loadtxt(full_path + '/fid.txt')
        row = {'Model': f, 'Minimum FID': np.min(fid), "Epoch": np.argmin(fid)}
        dict_data.append(row)
    except:
        continue

with open("min_fids.csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)
