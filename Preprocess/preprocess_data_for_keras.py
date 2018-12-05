import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy import signal
from math import factorial

lemmatizer = WordNetLemmatizer()
hm_lines = 6000000								#prosoxh otan yperbei to 100000

'''
sequences twn 100 opws sumbainoun sth pragmatikotita xwris na ta xwrisw se slots
'''

def read_lap_data(lap_txt):
	lap_data = []
	with open(lap_txt,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			results = list(map(float, all_words))		#string list -> float list
			lap_data += list(results)

	count = 1
	latitude = []
	longitude = []
	elevation = []
	speed = []
	iac = []

	flag = True														#gia na shmeiwnw otan lat or long = 0, kai na mh ta lambanw ypopsi
	for index in lap_data:											#dioti tote trwei kolhmata to gps
		if ((count == 9) and (index!=0)):							#xwrizw se listes tis eisodous
			latitude += list([index])								#index != 0 giati exw kapoia sfalmata logo thlemetrias
		elif ((count==9) and (index == 0)):							#index = 0 prosperase ola ta stoixeia
			flag = False
		elif((count == 10) and flag):
			longitude += list([index])
		elif((count==11) and flag):
			elevation += list([index])	
		elif((count==12) and flag):
			speed += list([index])
		elif((count==21) and flag):
			iac += list([index])
			count = 0
		elif((count==21) and(flag==False)):							#paw sthn epomenh tetrada dedomenwn
			flag = True
			count = 0
		count+=1
		
	return(elevation, speed, iac)

def make_sequences_of_n(v,iac,e,seq_len=300,test_size = 0.95):

	Velocity = []
	I = []
	Elevation = []
	sequence_counter = 0

	v_row = []
	e_row = []
	i_row = 0

	for i in range(len(v)):		#tha fitaksw minimum_len dekades

		if (sequence_counter==seq_len):
			
			Velocity.append(v_row)
			Elevation.append(e_row)
			I.append([i_row])

			v_row = []
			e_row = []
			i_row = 0
			sequence_counter = 0
							
		v_row.append(v[i]/100)		#~1	append to stoixeio 		
		i_row+=  iac[i]/10000		#~1
		e_row.append(e[i]/100)		#to ypsometro
		
		sequence_counter+=1

	testing_size = int(test_size*len(I))
	
	train_X = []
	train_Y = []

	test_X = []
	test_Y = []

	for i in range(len(I)-1):
		if (i<testing_size):
			train_X.append([Velocity[i],Elevation[i]])
			train_Y.append(I[i])
		else:
			test_X.append([Velocity[i],Elevation[i]])
			test_Y.append(I[i])
	
	return train_X,train_Y,test_X,test_Y	

if __name__ == '__main__':

	print("Reading...")
	
	with open('fake_elevation_acceleration', 'rb') as fp:
		fake_elevation_acceleration = pickle.load(fp)
	with open('fake_iac_acceleration', 'rb') as fp:
		fake_iac_acceleration = pickle.load(fp)
	with open('fake_speed_acceleration', 'rb') as fp:
		fake_speed_acceleration = pickle.load(fp)
	
	
	elevation=list(fake_elevation_acceleration)
	speed=list(fake_speed_acceleration)
	iac=list(fake_iac_acceleration)
	
	temp_elevation, temp_speed, temp_iac = read_lap_data('/IECC/Data Analysis/SEM 2016/Telemetry Data/3rd_test_30_6_2016.txt')
	elevation+=list(temp_elevation)
	speed+=list(temp_speed)
	iac+=list(temp_iac)
	
	temp_elevation, temp_speed, temp_iac = read_lap_data('/IECC/Data Analysis/SEM 2016/Telemetry Data/Blown tire.txt')
	elevation+=list(temp_elevation)
	speed+=list(temp_speed)
	iac+=list(temp_iac)
	
	
	temp_elevation, temp_speed, temp_iac = read_lap_data('/IECC/Data Analysis/SEM 2017/24_5_2017.txt')
	elevation+=list(temp_elevation)
	speed+=list(temp_speed)
	iac+=list(temp_iac)
	
	with open('fake_elevation_constant_speed', 'rb') as fp:
		fake_elevation_constant_speed = pickle.load(fp)
	with open('fake_iac_constant_speed', 'rb') as fp:
		fake_iac_constant_speed = pickle.load(fp)
	with open('fake_speed_constant_speed', 'rb') as fp:
		fake_speed_constant_speed = pickle.load(fp)

	elevation+=list(fake_elevation_constant_speed)
	speed+=list(fake_speed_constant_speed)
	iac+=list(fake_iac_constant_speed)
	
	temp_elevation, temp_speed, temp_iac = read_lap_data('/IECC/Data Analysis/SEM 2017/27_5_2017.txt')
	elevation+=list(temp_elevation)
	speed+=list(temp_speed)
	iac+=list(temp_iac)
	
	temp_elevation, temp_speed, temp_iac = read_lap_data('/IECC/Data Analysis/SEM 2016/Telemetry Data/2nd_test_30_6_2016.txt')
	elevation+=list(temp_elevation)
	speed+=list(temp_speed)
	iac+=list(temp_iac)
	print(len(temp_iac))
	
	
	filtered_elevation = signal.savgol_filter(elevation, 1001, 3,mode='nearest') # window size 1001, polynomial order 3

	print('Elevation-Iac-Speed')

	plt.subplot(3, 1, 1)
	plt.plot(filtered_elevation)
	plt.title('Elevation')
	plt.subplot(3, 1, 2)
	plt.plot(iac)
	plt.title('Iac')
	plt.subplot(3,1,3)
	plt.plot(speed)
	plt.title('Speed')
	plt.show()	

	print(len(filtered_elevation),len(speed),len(iac))

	print("Making sequences of 300...")

	train_X,train_Y,test_X,test_Y = make_sequences_of_n(speed,iac,filtered_elevation)
	
	with open('Train_X', 'wb') as fp:
		pickle.dump(train_X, fp)
	with open('Train_Y', 'wb') as fp:
		pickle.dump(train_Y, fp)

	with open('Test_X', 'wb') as fp:
		pickle.dump(test_X, fp)
	with open('Test_Y', 'wb') as fp:
		pickle.dump(test_Y, fp)

	with open('Filtered_elevation', 'wb') as fp:
		pickle.dump(filtered_elevation, fp)
	
	print("Train data len:	",len(train_X))
	print("Test data len:	",len(test_X))
	print("Sequences made and saved!")
