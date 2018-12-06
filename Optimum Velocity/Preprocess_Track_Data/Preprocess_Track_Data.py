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
hm_lines = 6000000								


def make_sequences_of_n(v,iac,e,seq_len=300):

	Velocity = []
	I = []
	Elevation = []
	sequence_counter = 0

	v_row = []
	e_row = []
	i_row = 0

	for i in range(len(v)):		

		if (sequence_counter==seq_len):
			
			Velocity.append(v_row)
			Elevation.append(e_row)
			I.append([i_row])

			v_row = []
			e_row = []
			i_row = 0
			sequence_counter = 0
							
		v_row.append(v[i]/100)		#~1	 		
		i_row+=  iac[i]/10000		#~1
		e_row.append(e[i]/100)		#~1
		
		sequence_counter+=1

	test_X = []
	test_Y = []

	for i in range(len(I)-1):
		test_X.append([Velocity[i],Elevation[i]])
		test_Y.append(I[i])
	
	return test_X,test_Y	

if __name__ == '__main__':

	print("Reading...")

	with open('Track_Elevation', 'rb') as fp:
		elevation = pickle.load(fp)
	with open('Track_Velocities', 'rb') as fp:
		speed = pickle.load(fp)
	with open('Track_Iac', 'rb') as fp:
		iac = pickle.load(fp)

	print("Making sequences of 300...")

	test_X,test_Y = make_sequences_of_n(speed,iac,elevation)

	with open('Lap_Data', 'wb') as fp:
		pickle.dump(test_X, fp)

	print("Data len: ", len(test_X))

	print("Sequences made and saved!")