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
hm_lines = 22550	

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

	flag = True	#The flag indicates if the GPS data are corrupted										
	for index in lap_data:											
		if ((count == 9) and (index!=0)):	#index!=0 to avoid temetry corrupted data
			latitude += list([index])	
		elif ((count==9) and (index == 0)):	#index=0 pass the whole line
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
		elif((count==21) and(flag==False)):	#go the next usefull data
			flag = True
			count = 0
		count+=1
		
	return(elevation, latitude, longitude)

def make_slots(Elevation):
	
	'''
	slope = 1 --> decrease Elevation
	slope = 0 --> increase Elevation
	'''

	slope = 0
	previous_p = 0
	margin_of_error = 2		#these variables changes the total number of slots
	acceptable_error = margin_of_error
	gradient= [0] 			# 0 if the first slot is an uphill - 1 if it is downhill

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
	print('number of slots:', l)
	return final_slots,gradient

def make_velocities_Iac_acceleration_deceleration(begin_of_slot, gradient,max_speed=45,min_speed=10):

	iac = []
	speed = []
	previous_slot = 0
	gradient_count = -1
	current_speed = 25
	hz_count = 49	#to gps pairnei digamta me rythmo 50 hz
	slot_counter=0

	for slot_stop in begin_of_slot:

		gradient_count+=1
		slot_counter +=1

		while (previous_slot<slot_stop):
			
			if (gradient[gradient_count]==0):	#in case of uphill
				if (hz_count==0):				
					if (current_speed<max_speed):
						current_speed+=1
					speed.append(current_speed)
					if (slot_counter==3):		#the third slot has a huge slope
						iac.append(200)
					else:
						iac.append(100)
					hz_count=49
				else:
					speed.append(current_speed)
					if (slot_counter==3):
						iac.append(200)
					else:
						iac.append(100)
					hz_count-=1
			else:							#in case of downhill
				if (hz_count==1):				
					if (current_speed>min_speed):
						current_speed-=1
					speed.append(current_speed)
					iac.append(40)
					hz_count=49
				else:
					speed.append(current_speed)
					iac.append(40)
					hz_count-=1	

			previous_slot+=1

	return(speed,iac)

def make_velocities_Iac_acceleration_deceleration_in_favor_of_downhill(begin_of_slot, gradient,max_speed=45,min_speed=10):

	iac = []
	speed = []
	previous_slot = 0
	gradient_count = -1
	current_speed = 25
	hz_count = 49	#to gps pairnei digamta me rythmo 50 hz
	slot_counter=0

	for slot_stop in begin_of_slot:

		gradient_count+=1
		slot_counter +=1

		while (previous_slot<slot_stop):
			
			if (gradient[gradient_count]==0):	
				if (hz_count==0):				
					if (current_speed>min_speed):
						current_speed-=1
					speed.append(current_speed)
					iac.append(2)
					hz_count=49
				else:
					speed.append(current_speed)
					iac.append(2)
					hz_count-=1
			else:								
				if (hz_count==1):				
					if (current_speed<max_speed):
						current_speed+=1
					speed.append(current_speed)
					iac.append(0)
					hz_count=49
				else:
					speed.append(current_speed)
					iac.append(0)
					hz_count-=1	

			previous_slot+=1

	return(speed,iac)

def make_velocities_Iac_costant_speed(begin_of_slot, gradient,current_speed=45):

	iac = []
	speed = []
	previous_slot = 0
	gradient_count = -1
	hz_count = 49	#to gps pairnei digamta me rythmo 50 hz
	slot_counter=0

	for slot_stop in begin_of_slot:

		gradient_count+=1
		slot_counter +=1

		while (previous_slot<slot_stop):
			
			if (gradient[gradient_count]==0):	
				if (hz_count==0):				
					speed.append(current_speed)
					if (slot_counter==3):		
						iac.append(100)
					else:
						iac.append(40)
					hz_count=49
				else:
					speed.append(current_speed)
					if (slot_counter==3):
						iac.append(100)
					else:
						iac.append(40)
					hz_count-=1
			else:								
				if (hz_count==1):				
					speed.append(current_speed)
					iac.append(0)
					hz_count=49
				else:
					speed.append(current_speed)
					iac.append(0)
					hz_count-=1	

			previous_slot+=1

	return(speed,iac)

def make_velocities_Iac_costant_speed_only_in_steep_slope(begin_of_slot, gradient,track_elevation,current_speed=45):

	iac = []
	speed = []
	elevation = []
	previous_slot = 0
	element_counter = 0
	gradient_count = -1
	hz_count = 49	#to gps pairnei digamta me rythmo 50 hz
	slot_counter=0

	for slot_stop in begin_of_slot:

		gradient_count+=1
		slot_counter +=1

		while (previous_slot<slot_stop):
			
			if (gradient[gradient_count]==0):	
				if (hz_count==0):				
					if (slot_counter==3):		
						iac.append(200)
						speed.append(current_speed)
						elevation.append(track_elevation[element_counter])
					hz_count=49
				else:
					if (slot_counter==3):
						speed.append(current_speed)
						iac.append(200)
						elevation.append(track_elevation[element_counter])
					hz_count-=1
			else:								
				if (hz_count==1):				
					hz_count=49
				else:
					hz_count-=1	

			previous_slot+=1
			element_counter+=1

	return(speed,iac,elevation)

def make_velocities_Iac_small_costant_speed_only_in_steep_slope(begin_of_slot, gradient,track_elevation,current_speed=10):

	iac = []
	speed = []
	elevation = []
	previous_slot = 0
	element_counter = 0
	gradient_count = -1
	hz_count = 49	#to gps pairnei digamta me rythmo 50 hz
	slot_counter=0

	for slot_stop in begin_of_slot:

		gradient_count+=1
		slot_counter +=1

		while (previous_slot<slot_stop):
			
			if (gradient[gradient_count]==0):	
				if (hz_count==0):				
					if (slot_counter==3):		
						iac.append(5)
						speed.append(current_speed)
						elevation.append(track_elevation[element_counter])
					hz_count=49
				else:
					if (slot_counter==3):
						speed.append(current_speed)
						iac.append(5)
						elevation.append(track_elevation[element_counter])
					hz_count-=1
			else:								
				if (hz_count==1):				
					hz_count=49
				else:
					hz_count-=1	

			previous_slot+=1
			element_counter+=1

	return(speed,iac,elevation)	

if __name__ == '__main__':

	print("Reading...")

	elevation, latitude, longitude = read_lap_data('/home/andreas/Desktop/IECC/Data Analysis/SEM 2016/Telemetry Data/Blown tire.txt')

	#delete first 3000 elements, they are error
	elevation = elevation[3000:]
	latitude = latitude[3000:]
	longitude = longitude[3000:]

	filtered_elevation = signal.savgol_filter(elevation, 1001, 3,mode='nearest') # window size 1001, polynomial order 3

	begin_of_slot, gradient = make_slots(filtered_elevation)
	print('Gradient per slot: ', gradient)
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
		#plt.plot(temp)
		#plt.show()

	speed_acceleration, iac_acceleration = make_velocities_Iac_acceleration_deceleration(begin_of_slot,gradient)
	temp_speed_acceleration,temp_iac_acceleration = make_velocities_Iac_acceleration_deceleration_in_favor_of_downhill(begin_of_slot,gradient)
	
	last_useless_elements=len(filtered_elevation)-len(iac_acceleration)
	
	speed_acceleration+=list(temp_speed_acceleration)
	iac_acceleration+=list(temp_iac_acceleration)


	elevation_acceleration = list(elevation[:-last_useless_elements])
	elevation_acceleration += list(elevation_acceleration)
	filtered_elevation_acceleration = list(filtered_elevation[:-last_useless_elements])
	temp_filtered_elevation_acceleration = filtered_elevation_acceleration
	filtered_elevation_acceleration+=list(temp_filtered_elevation_acceleration)

	print('Elevation-Iac-Speed-Acceleration')

	plt.subplot(3, 1, 1)
	plt.plot(filtered_elevation_acceleration)
	plt.title('Elevation')
	plt.subplot(3, 1, 2)
	plt.plot(iac_acceleration)
	plt.title('Iac')
	plt.subplot(3,1,3)
	plt.plot(speed_acceleration)
	plt.title('Speed')
	plt.show()	

	speed_constant_speed, iac_constant_speed, elevation_constant_speed = make_velocities_Iac_costant_speed_only_in_steep_slope(begin_of_slot,gradient,elevation)
	
	temp_speed_constant_speed,temp_iac_constant_speed,temp_elevation_constant_speed = make_velocities_Iac_small_costant_speed_only_in_steep_slope(begin_of_slot,gradient,elevation)
	speed_constant_speed+=list(temp_speed_constant_speed)
	iac_constant_speed+=list(temp_iac_constant_speed)
	elevation_constant_speed+=list(temp_elevation_constant_speed)	

	temp_speed_constant_speed,temp_iac_constant_speed,temp_elevation_constant_speed = make_velocities_Iac_costant_speed_only_in_steep_slope(begin_of_slot,gradient,elevation,25)
	speed_constant_speed+=list(temp_speed_constant_speed)
	iac_constant_speed+=list(temp_iac_constant_speed)
	elevation_constant_speed+=list(temp_elevation_constant_speed)

	temp_speed_constant_speed,temp_iac_constant_speed,temp_elevation_constant_speed = make_velocities_Iac_small_costant_speed_only_in_steep_slope(begin_of_slot,gradient,elevation)
	speed_constant_speed+=list(temp_speed_constant_speed)
	iac_constant_speed+=list(temp_iac_constant_speed)
	elevation_constant_speed+=list(temp_elevation_constant_speed)	


	print('Constent len',len(speed_constant_speed),len(iac_constant_speed),len(elevation_constant_speed))
	print('Elevation-Iac-Speed-Constant_Speed')

	plt.subplot(3, 1, 1)
	plt.plot(elevation_constant_speed)
	plt.title('Elevation')
	plt.subplot(3, 1, 2)
	plt.plot(iac_constant_speed)
	plt.title('Iac')
	plt.subplot(3,1,3)
	plt.plot(speed_constant_speed)
	plt.title('Speed')
	plt.show()	

	print('length of latitude data',len(latitude))

	print('Plotting Map')
	plt.plot(longitude,latitude)
	plt.title('MAP')
	plt.show()


	plt.subplot(2, 1, 1)
	plt.plot(elevation)
	plt.subplot(2, 1, 2)
	plt.plot(filtered_elevation)
	plt.title('Elevation')
	plt.show()


	print(len(elevation_acceleration),len(iac_acceleration),len(speed_acceleration))
	print("saving the data...")

	with open('fake_elevation_acceleration', 'wb') as fp:
		pickle.dump(elevation_acceleration, fp)
	with open('fake_iac_acceleration', 'wb') as fp:
		pickle.dump(iac_acceleration, fp)
	with open('fake_speed_acceleration', 'wb') as fp:
		pickle.dump(speed_acceleration, fp)


	with open('fake_elevation_constant_speed', 'wb') as fp:
		pickle.dump(elevation_constant_speed, fp)
	with open('fake_iac_constant_speed', 'wb') as fp:
		pickle.dump(iac_constant_speed, fp)
	with open('fake_speed_constant_speed', 'wb') as fp:
		pickle.dump(speed_constant_speed, fp)
