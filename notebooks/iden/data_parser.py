import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from math import pi
import scipy.signal
from scipy.signal import butter, filtfilt
from scipy.stats.stats import pearsonr   

name = "Temporary_Patient_2019_11_01_14_54_52.txt"
results = {}

#Column names for different tables
cols = {9:['Timestamp', 'Eye hor', 'Eye ver', 'Head W', 'Head X', 'Head Y', 'Head Z', 'Eye abs X', 'Eye abs Y'],
	2:['Timestamp', 'Torsion']}

experiment = None #UID of current experiment
data = [] #Data for current experiment

with open(name) as file:
	for line in file:
		r = re.search('<TestUID>(?P<uid>\d+)</TestUID>', line)
		
		if r is not None: #This is a header string
			#First save previous results
			if experiment is not None:
				df = pd.DataFrame(data, columns=cols[len(data[0])])
				
				#If experiment already exists then we join to it else we create it
				if experiment in results:
					results[experiment] = results[experiment].join(df.set_index('Timestamp'), on='Timestamp')
				else:
					results[experiment] = df
			
			#Then clear the data
			experiment = int(r.group('uid'))
			data = []
			continue
		
		l = line.strip().replace(',', '.').split(';')
		data.append(l)

#Doing it once again for the last experiment
df = pd.DataFrame(data, columns=cols[len(data[0])])
if experiment in results:
	results[experiment] = results[experiment].join(df.set_index('Timestamp'), on='Timestamp')

#Adding relative time column, removing Timestamp
for key in results.keys():
	results[key] = results[key].astype(dtype = 'float')
	start_val = results[key]['Timestamp'].min()
	results[key]['Time'] = (results[key]['Timestamp'] - start_val) / 10000
	results[key] = results[key].drop(labels='Timestamp', axis=1).set_index('Time')
	
	#Converting head values to a single quaternion and getting yaw
	q = results[key].loc[:, ['Head W', 'Head X', 'Head Y', 'Head Z']].values
	results[key]['roll'] = np.arctan2(2.0*(q[:,2]*q[:,3] + q[:,0]*q[:,1]), 1 - 2 * (q[:,1]**2 + q[:,2]**2)) 		/ pi * 180
	results[key]['yaw'] = np.arctan2(2.0*(q[:, 1]*q[:, 2] + q[:, 0]*q[:, 3]), -1 + 2 * (q[:, 0]**2 + q[:, 1]**2))	/ pi * 180
	results[key]['pitch'] = np.arcsin(2.0*(q[:, 0]*q[:, 2] - q[:, 1]*q[:, 3]))										/ pi * 180
	
	#results[key]['yaw'][results[key]['yaw'] < 0] += 360
	
for key, value in results.items():
	#value.loc[:, ['Eye hor','Eye ver', 'yaw']].plot(secondary_y='yaw')
	
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax2.set_ylim(bottom=-10, top=10)
	
	eh = ax1.plot(value.index, value['Eye hor'], label = 'Eye hor')
	ev = ax1.plot(value.index, value['Eye ver'], color = 'tab:orange', label = 'Eye ver')
	et = ax1.plot(value.index, value['Torsion'], color = '#FF0000', label = 'Torsion')
	#hy = ax2.plot(value.index, value['yaw'], color = 'tab:red', label = 'Head yaw')
	#hp = ax2.plot(value.index, value['pitch'], color = 'tab:blue', label = 'Head pitch')
	#hr = ax2.plot(value.index, value['roll'], color = 'tab:green', label = 'Head roll')
	
	#lns = hy+hp+hr
	#lns = eh+ev+hy+hp+hr+et
	lns = eh+ev+et
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs, loc='best')
	
	plt.xlabel = 'Time, miliseconds'
	plt.suptitle('Experiment ' + str(key))
	
plt.show()

#Calculating velocities and accelerations
#cur_data = results[81].loc[:, ['yaw', 'Eye hor', 'Torsion', 'roll', 'pitch']].reset_index(col_fill='Time')
#print(cur_data.loc[:, ['roll', 'Torsion']])
#Different types of smoothing:
#Getting each nth row
#cur_data = cur_data[cur_data.index % 15 == 0].reset_index(drop=True)
#Median smoothing
#cur_data['yaw'] = scipy.signal.medfilt(cur_data['yaw'].values, kernel_size = 3)
#Butterworth filter
#b, a = butter(5, 0.05)
#cur_data['yaw'] = filtfilt(b, a, cur_data['yaw'])

#cur_data['Time'] /= 1000 #Converting to seconds
#vals = cur_data['yaw'].values.astype('float')
#time = cur_data['Time'].values
#size = time.shape[0] - 1 #Max index

#Creating new columns
#cur_data = cur_data.assign(forw1_vel=np.nan, cent2_vel = np.nan, forw2_vel=np.nan, cent4_vel = np.nan)
#cur_data = cur_data.assign(forw1_acc=np.nan, cent2_acc = np.nan, forw2_acc=np.nan, cent4_acc = np.nan)
#cur_data = cur_data.assign(cent4_vel=np.nan, cent4_acc = np.nan)

#Max index is size
#cur_data.loc[0:size-1,'forw1_vel'] = (vals[1:] - vals[0:-1]) / (time[1:] - time[0:-1])
#cur_data.loc[1:size-1,'cent2_vel'] = (vals[2:] - vals[0:-2]) / (time[2:] - time[0:-2])
#cur_data.loc[0:size-2,'forw2_vel'] = (- vals[2:] + 4 * vals[1:-1] - vals[0:-2] * 3) / (time[2:] - time[0:-2])
#cur_data.loc[2:size-2,'cent4_vel'] = (- vals[4:] / 3 + vals[3:-1] * 8/3 - vals[1:-3] * 8/3 + vals[0:-4] / 3) / (time[4:] - time[0:-4])
#cur_data.loc[0:size-2,'forw1_acc'] = (4 * vals[0:-2] - 8 * vals[1:-1] + 4 * vals[2:]) / (time[2:] - time[0:-2])**2
#cur_data.loc[0:size-3,'forw2_acc'] = (18 * vals[0:-3] - 45 * vals[1:-2] + 36 * vals[2:-1] - 9 * vals[3:]) / (time[3:] - time[0:-3])**2
#cur_data.loc[1:size-1,'cent2_acc'] = (4 * vals[0:-2] - 8 * vals[1:-1] + 4 * vals[2:0]) / (time[2:] - time[0:-2])**2
#cur_data.loc[2:size-2,'cent4_acc'] = (- vals[0:-4] * 4/3 + vals[1:-3] * 64/3 - vals[2:-2] * 40 + vals[3:-1] * 64/3 - vals[4:] * 4/3) / (time[4:] - time[0:-4])**2

#cur_data.set_index('Time').plot()
#plt.show()