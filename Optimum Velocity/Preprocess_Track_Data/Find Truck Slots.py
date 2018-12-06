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
from numpy import linspace

lemmatizer = WordNetLemmatizer()
hm_lines = 42800

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
		
	return(elevation, speed, iac, latitude, longitude)

def make_slots(Elevation):
	
	'''
	slope = 1 --> meiwsh Elevation
	slope = 0 --> auksish elevationn
	'''

	slope = 0
	previous_p = 0
	margin_of_error = 5		#allazontas auta ta dyo meiwneis kai megalwneis ton arithmo twn slots
	acceptable_error = margin_of_error
	gradient= [0] 			#orizw me 0 ean to prwto slot einai anifora kai me 1 ean einai katifora, thewrw to prwto slot anifora

	begin_of_slot = []

	count = 0

	for j in Elevation:

		i = np.round(j,2)
		
		if ((slope ==0) and (i>previous_p) and (margin_of_error<acceptable_error)):
			margin_of_error+=1

		if ((slope ==0) and (i < previous_p) and (margin_of_error!=0)):
			margin_of_error-=1

		if ((slope ==0) and (i<previous_p) and (margin_of_error==0)):
			begin_of_slot.append(count)
			margin_of_error = acceptable_error
			gradient.append(1)
			slope = 1

		if ((slope ==1) and (i<previous_p) and (margin_of_error<acceptable_error)):
			margin_of_error+=1

		if ((slope ==1) and (i > previous_p) and (margin_of_error!=0)):
			margin_of_error-=1

		if ((slope ==1) and (i > previous_p) and (margin_of_error==0)):
			begin_of_slot.append(count)
			margin_of_error = acceptable_error
			slope = 0
			gradient.append(0)

		count+=1
		previous_p = i

	final_slots = []
	previous_begin_of_slot = 0
	for i in begin_of_slot:	#Ean to mhkos tou slot einai mikro diegrapse to
		if (abs(i - previous_begin_of_slot) > 6):
			final_slots.append(i)		
		previous_begin_of_slot = i

	l = len(final_slots)
	return final_slots,gradient

def make_sequences_of_n(v,iac,e,seq_len=300,test_size = 1):

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

	elevation, speed ,iac, latitude, longitude = read_lap_data('/home/andreas/Desktop/IECC/Data Analysis/SEM 2017/27_5_2017.txt')

	#delete first 3000 elements, they are error
	elevation = elevation[28000:]
	latitude = latitude[28000:]
	longitude = longitude[28000:]
	speed = speed[28000:]
	iac = iac[28000:]

	filtered_elevation = signal.savgol_filter(elevation, 1001, 3,mode='nearest') # window size 1001, polynomial order 3

	begin_of_slot, gradient = make_slots(filtered_elevation)

	print('Gradient per slot: ', gradient)
	'''
	count = 0
	for i in begin_of_slot:
		temp = []
		previous = count
		e_count = 0
		for j in filtered_elevation:
			if ((count<i) and (e_count>=previous)):
				temp.append(j)
				count+=1
			e_count += 1
		plt.plot(temp)
		plt.show()

	#for the last slot
	
	temp = []
	previous = count
	e_count = 0
	for j in filtered_elevation:
		if ((count<len(elevation)) and (e_count>=previous)):
			temp.append(j)
			count+=1
		e_count += 1
	plt.plot(temp)
	plt.show()
	'''

	'''
	print('Plotting Map')
	plt.plot(longitude,latitude,'g')
	plt.xlabel('X: Longitude')
	plt.ylabel('Y: Latitude')
	plt.show()

	'''
	print("saving the data...")

	end_of_slots = begin_of_slot
	end_of_slots.append(len(elevation))

	table=[]
	count=0
	for i in range(len(elevation)):
		if (i==end_of_slots[count]):
			table.append([0])
			count+=1
		else:
			table.append([-10])

	#plt.subplot(2, 1, 1)
	#plt.plot(elevation)
	#plt.subplot(2, 1, 2)
	plt.figure(1)
	plt.plot(filtered_elevation,'g')
	plt.plot(table,'ro')
	plt.ylim(0,50)
	plt.title('Elevation')
	plt.show()

	print('number of slots:', len(end_of_slots))
	print('end of each slot:', end_of_slots)
	print('length of the track data:', len(elevation))

	train_X,train_Y,test_X,test_Y = make_sequences_of_n(speed,iac,filtered_elevation)
	train_X = train_X[1:]

	with open('End_Of_Each_Slot', 'wb') as fp:
		pickle.dump(end_of_slots, fp)
	with open('Track_Elevation', 'wb') as fp:
		pickle.dump(filtered_elevation, fp)
	with open('Track_Velocities', 'wb') as fp:
		pickle.dump(speed, fp)
	with open('Track_Iac', 'wb') as fp:
		pickle.dump(iac, fp)
	with open('Lap_Data', 'wb') as fp:
		pickle.dump(train_X, fp)
