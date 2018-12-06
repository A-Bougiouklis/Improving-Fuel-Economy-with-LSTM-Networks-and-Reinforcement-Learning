from __future__ import print_function
import os
import warnings
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from operator import *
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras.models import model_from_json
from scipy import signal

#based on coison.py Desktop/keras Toutorial
# Activate virtualenv (prompt will change)	->	. ./.py35/bin/activate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

learnig_rate = 0.7
epochs = 100000
discount_factor = 0	#Einai to idio na parw reward eite twra eite se epomeno state
disarable_average_speed = 31
speed_margin = 	1		# avg_speed-speed_margin <= final_avg_speed <= avg_speed+speed_margin
max_velocity = 45
min_velocity = 10			
early_stop_patience = 2000	#pollaplasio tou 10!

with open('End_Of_Each_Slot', 'rb') as fp:
	end_of_slot = pickle.load(fp)
with open('Lap_Data', 'rb') as fp:
	test_X = pickle.load(fp)

track_slots = len(end_of_slot)

len_of_each_slot = []
previous_end = 0
for end in end_of_slot:			#I calculate these in order to calculate the correct average speed
	len_of_each_slot.append(end-previous_end)
	previous_end = end
	
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")


def Create_Q_Table():

	velocity_range = max_velocity-min_velocity+1

	# Q(State,Action)
	Q_table = [[0 for x in range(velocity_range)] for y in range(track_slots)] 

	return(Q_table)	

def Modify_Lap_Data(lap_data,local_policy):

	data_counter = 0
	element_counter = 0
	new_lap_data = list(lap_data)
	state = 0

	#print(local_policy)

	plt_elevation = []
	plt_vel = []

	for net_input in lap_data:	#net_input = [[velocity],[elevation]]

		velocity = list(net_input[0])
		el = list(net_input[1])

		for sequence_element in range(len(velocity)):

			velocity[sequence_element]= local_policy[state]/100	#eisagw th taxhthta tou policy kai frontizw na einai ~1
			
			plt_vel += list([local_policy[state]])
			plt_elevation+= list([el[sequence_element]*100])

			element_counter+=1
			if(element_counter==end_of_slot[state]):			#the elements are in a row and thats how I know in which state they belong
				state+=1

		new_lap_data[data_counter][0] = list(velocity)		#restore the correct values
		data_counter+=1
	
	filtered_vel = signal.savgol_filter(plt_vel, 1001, 3,mode='nearest') # window size 1001, polynomial order 3
	
	plt.plot(filtered_vel)
	plt.plot(plt_elevation)
	#plt.title('Different Velocity Profiles')
	return(new_lap_data)

def Calculate_Average_Speed(local_policy):

	total_sum =  0
	state_counter = 0


	for v in local_policy:	#weighted average
		total_sum += v*len_of_each_slot[state_counter]
		state_counter+=1

	avg = total_sum/(end_of_slot[state_counter-1])	#total sum / mesurments
	return(avg)

def Find_the_largest_q_value(state):

	maximum_q_value = -100000000000
	counter = 0

	for q_value in state:
		if (q_value>maximum_q_value):	#tha exei th tash na epilegei mikroteres taxhthtes
			maximum_q_value = q_value
			position_max_q_value = counter
		counter+=1

	return(position_max_q_value)

def Select_Action(state,n):

	#n = number of iterations, the age of the agent
	#E = exploitation parameter, 0<E<1

	E = 0.0001
	e_greedy_probability =  np.exp(-E*n)

	#a = list, len = 1, if a = 0 then choose the bigger Q_Value else choose a random one
	a = np.random.choice(2, 1, p=[1-e_greedy_probability,e_greedy_probability])
	a = a[0] #first element of list
	
	#action = velocity
	if (a==0):
		action = Find_the_largest_q_value(state)
	else:
		velocity = random.randint(min_velocity,max_velocity)
		action = velocity-min_velocity 				#h thesh ths taxythtas sto q_table

	return(action)

def Calculate_Immediate_Reward(lap_data,policy_consumption):

	new_policy_consumption = model.predict(lap_data, batch_size=2)

	prediction = 0
	for i in new_policy_consumption:
		for j in i:
			prediction += j

	immediate_reward = 0.02*(policy_consumption - prediction)
	#print(immediate_reward)

	return(immediate_reward,prediction)

def Update_Q_table(q_table,immediate_reward,state,action,time_reward):

	present_q_value = q_table[state][action]
	'''
	if (state<(track_slots-1)):
		next_max_q_value = Find_the_largest_q_value(q_table[state+1])
	else:
		next_max_q_value = Find_the_largest_q_value(q_table[0])
	
	new_experience = learnig_rate*(immediate_reward + time_reward + discount_factor*next_max_q_value - present_q_value)
	new_experience = learnig_rate*(immediate_reward + time_reward - present_q_value)
	'''
	new_experience = learnig_rate*(immediate_reward + time_reward - present_q_value)
	new_q_value = present_q_value + new_experience
	q_table[state][action] = new_q_value

	return(q_table)

def Calculate_time_Reward(local_policy):

	avg_speed = Calculate_Average_Speed(local_policy)

	if ((avg_speed > (disarable_average_speed+speed_margin)) or (avg_speed < (disarable_average_speed-speed_margin))):	#ektos oriwn
		#print('-')
		return (-0.5)
	else:
		#print('+')
		return (0.5)

def Q_Algorithm(epochs,q_table):

	policy = [disarable_average_speed for x in range(track_slots)]
	
	lap_data = Modify_Lap_Data(test_X,policy)
	
	network_prediction =  model.predict(lap_data, batch_size=2)
	#predicted_output = []
	policy_consumption = 0
	for i in network_prediction:
		for j in i:
			policy_consumption += j
			#predicted_output.append(j)
	'''
	plt.plot(predicted_output)
	plt.title('Prediction')
	plt.show()
	'''
	print("Consumpion with contant speed = avg_speed -> ", policy_consumption)

	print("training...")

	no_change_in_consumption_after_epochs = 0
	change_in_policy =  False

	for iteration in range(epochs):
		
		if (mod(iteration,10)==0):

			print("after ",iteration," epochs the predicted consumption is = ", policy_consumption )
			print("policy is ->",policy)

			if (no_change_in_consumption_after_epochs>=early_stop_patience):

				print()
				print("EARLY STOP")
				print()

				zeros = 0
				for i in q_table:
					for j in i:
						if (j==0):
							zeros+=1

				print('zeros = ', zeros)
				print('episkefthike = ', ((648-zeros)/648)*100, '%')
				print()

				return(policy)

		for state in range(track_slots):
			#print("state=",state)
			#action = h sthlh pou ekprosopei th taxytha
			action = Select_Action(q_table[state],iteration)

			max_Q_values = []
			for po in q_table:
				max_Q_values.append(Find_the_largest_q_value(po)+min_velocity)

			max_Q_values[state]= action+min_velocity 				#update temp_policy, action+min_velocity = velocity
			
			new_lap_data = list(Modify_Lap_Data(test_X,max_Q_values))
			immediate_reward,new_policy_consumption = Calculate_Immediate_Reward(new_lap_data,policy_consumption)
			time_reward = Calculate_time_Reward(max_Q_values)
			q_table = list(Update_Q_table(q_table,immediate_reward,state,action,time_reward))

			if (new_policy_consumption <= policy_consumption):# and (time_reward > 0)): #with <= I let it to explore more
				policy = list(max_Q_values)
				policy_consumption = new_policy_consumption
				change_in_policy = True

		if (change_in_policy):	#Early Stop Check
			change_in_policy = False
			no_change_in_consumption_after_epochs = 0
		else:
			no_change_in_consumption_after_epochs += 1

	print()
	print('MAX Q_VALUES')	
	max_Q_values = []
	for andreas in q_table:
		max_Q_values.append(Find_the_largest_q_value(andreas)+min_velocity)

	print(max_Q_values)

	print('Avg Speed of MAX Q values: ', Calculate_Average_Speed(max_Q_values))

	print()
	print('BEST POLICY')

	print('Average Speed = ', Calculate_Average_Speed(policy))

	print()

	zeros = 0
	for i in q_table:
		for j in i:
			if (j==0):
				zeros+=1

	print('zeros =', zeros)
	print('episkefthike = ', ((648-zeros)/648)*100, '%')
	print()


	return(policy)


if __name__ == '__main__':

	q_table = Create_Q_Table()

	plt.figure(1)
	
	plt.title('Velocity Profile')
	plt.xlabel('Time 50 x sec')
	plt.ylabel('Velocity km/h')
	plt.show()
	#print(Calculate_Average_Speed(list([24,31,11,12,22,17,27,31,28,29,30,34,29,21])))
	
	policy_final = Q_Algorithm(epochs,q_table)
	print("'Optimum_Velocities --->	",policy_final)

	print()

	print(Calculate_Average_Speed(policy_final))

	print()

	print('Predicting...')


	new_lap_data = Modify_Lap_Data(test_X,policy_final)
	predicted_output =  model.predict(new_lap_data, batch_size=2)

	prediction = 0
	for i in predicted_output:
		for j in i:
			prediction +=j

	print("Consumption after training = ", prediction)

	with open('Optimum_Velocities', 'wb') as fp:
		pickle.dump(policy_final, fp)

	del model 	#delete the model