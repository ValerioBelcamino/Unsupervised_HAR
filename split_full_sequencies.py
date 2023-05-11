import os
import numpy as np
from operator import itemgetter
import sys 
from scipy import interpolate

def load_set_indexes(path):
    with open(path, 'r') as f:
        content = f.readlines()
        new_content = []
        for line in content:
            new_content.append(line.split('\n')[0])
    return new_content


def resample(sequence, tstamp, trgtstamp):
    tstamp = tstamp.squeeze()
    trgtstamp = trgtstamp.squeeze()
    sequence = np.array(sequence)
    newAxes = []
    tries = -1

    # FIX SEQUENCE LENGTH IF THE NEW STAMPS ARE OUTSIDE THE INTERPOLATION INTERVAL
    while tstamp[-1] < trgtstamp[-1]:
        trgtstamp = trgtstamp[:-1]
        tries -=1
    for ax in range(sequence.shape[1]):
        seq = sequence[:,ax]
        f = interpolate.interp1d(tstamp, seq)
        newAxes.append(f(trgtstamp))
    return np.column_stack(newAxes)




def split_sequencies(sequencies, length):
    new_sequencies = []
    for seq in sequencies:
        n_sub = int(seq.shape[0]/length)
        for i in range(n_sub):
            new_sequencies.append(seq[i*length:(i+1)*length, :])
    return new_sequencies


base_saving_path = '/home/index1/Desktop/Unsupervised_HAR'
base_path = '/home/index1/Desktop/TER_HAR_Dataset'
full_data_path = os.path.join(base_path, 'full_sequences')
test_folders = load_set_indexes(os.path.join(base_path, 'test_set.txt'))
skiplist = ['s24_nh', 's11', 's44', 's16', 's42', 
            's18_nh', 's26_nh', 's26', 's14_nh', 
            's15', 's33_nh', 's55_nh', 's02_nh',
            's15_nh', 's23_nh', 's35', 's41_nh', 
            's45_nh', 's52_nh', 's55', 's57', 's47_nh',
            's57_nh', 's13_nh', 's11_nh', 's53', 's22']

full_sequencies = []


for seq_folder in os.listdir(full_data_path):
    if seq_folder in test_folders:
        if seq_folder in skiplist:
            continue    
        current_folder = os.path.join(full_data_path, seq_folder)

        print(current_folder)
        sensor_files = []
        for file in sorted(os.listdir(current_folder)):

            if 'action' in file:
                continue
            else:
                sensor_files.append(np.loadtxt(open(os.path.join(current_folder, file), "rb"), delimiter=",", skiprows=1,usecols=(0,5,6,7,8,9,10)))

        '''FIND COMMON START AND END'''
        new_min_stamp = max(sensor_files[0][0,0], sensor_files[1][0,0], sensor_files[2][0,0], sensor_files[3][0,0])
        new_max_stamp = min(sensor_files[0][-1,0], sensor_files[1][-1,0], sensor_files[2][-1,0], sensor_files[3][-1,0])


        min_len = sys.maxsize
        for i in range(len(sensor_files)):
            sensor_files[i] = sensor_files[i][np.where(sensor_files[i][:,0] >= new_min_stamp)]
            sensor_files[i] = sensor_files[i][np.where(sensor_files[i][:,0] <= new_max_stamp)]
            if len(sensor_files[i]) < min_len:
                min_len = len(sensor_files[i])
        
        for i in range(len(sensor_files)):
            sensor_files[i] = resample(sensor_files[i], 
                                       np.array([s for s in range(len(sensor_files[i]))]), 
                                       np.array([s for s in range(min_len)]))
            sensor_files[i] = sensor_files[i][:,1:]

        full_sequencies.append(np.column_stack(sensor_files))

print(f'{len(full_sequencies)=}')

full_sequencies = split_sequencies(full_sequencies, 500)
print(f'{len(full_sequencies)=}')
print(f'{full_sequencies[0].shape=}')

np_full_sequencies = np.array(full_sequencies)
print(f'{np_full_sequencies.shape=}')
np.save(os.path.join(base_saving_path, 'full_sequencies.npy'), np_full_sequencies)