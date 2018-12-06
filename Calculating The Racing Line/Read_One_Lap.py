import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
#hm_lines = 22550	#track 2016
hm_lines = 43200	#track 2017

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
		
	return(latitude, longitude)

def kepp_1_every_n_elements(lst,elements): # I cut some elements in order to simpliy the process of lerarnig by reducing the amount of states
								  # besase every mesurment is a state
	counter = 0
	final_list = []

	for el in lst:
		counter+=1
		if (counter==elements):
			final_list.append(el)
			counter = 0
		
	return(final_list)	



if __name__ == '__main__':

	print("Reading...")

	#track 2016
	#latitude, longitude = read_lap_data('/home/andreas/Desktop/IECC/Data Analysis/SEM 2016/Telemetry Data/Blown tire.txt')

	#track 2017
	latitude, longitude = read_lap_data('/home/andreas/Desktop/IECC/Data Analysis/SEM 2017/27_5_2017.txt')

	#track 2016
	#delete first 3000 elements, they are error
	#latitude = latitude[3000:]
	#longitude = longitude[3000:]
	
	#track 2017
	#delete first 25000 elements, they are error
	latitude = latitude[28000:]
	longitude = longitude[28000:]

	print('Amount of mesurments before cut = ',len(latitude))

	latitude = list(kepp_1_every_n_elements(latitude,140))
	longitude = list(kepp_1_every_n_elements(longitude,140))

	print('Amount of mesurments = ',len(latitude))
	plt.figure(1)
	print('Plotting Map')
	plt.plot(longitude,latitude,'g')
	plt.title('MAP')
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.show()

	with open('Latitude', 'wb') as fp:
		pickle.dump(latitude, fp)
	with open('Longitude', 'wb') as fp:
		pickle.dump(longitude, fp)