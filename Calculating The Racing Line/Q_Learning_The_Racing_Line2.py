import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import random
from operator import *
from collections import Counter
from nltk.stem import WordNetLemmatizer
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy import signal

action_step = 1
number_of_actions = 7	#-3,-2,-1,0,1,2,3
epochs = 1000000
learnig_rate = 0.001
discount_factor = 0
early_stop_patience = 10000

def Create_Q_Table(num_of_states,num_of_actions):


	# Q(State,Action)
	Q_table = [[0 for x in range(num_of_actions)] for y in range(num_of_states)] 

	return(Q_table)	

def find_radius_of_circle_from_3_elements(lon1,lat1,lon2,lat2,lon3,lat3):

	x1 = (lon2+lon1)/2	#madiator 1	x
	y1 = (lat2+lat1)/2	#madiator 1 y
	
	l1 = -1/((lat2-lat1)/(lon2-lon1))	# l_e1 * l_lat1_lon_2 = -1

	x2 = (lon3+lon2)/2	#madiator 2	x
	y2 = (lat3+lat2)/2	#madiator 2 y

	l2 = -1/((lat3-lat2)/(lon3-lon2))	# l_e2 * l_lat2_lon_3 = -1

	if (l1!=l2):	

		common_x = (x2*l2 - y2 - l1*x1 + y1)/(l2-l1)
		common_y = l1*(common_x-x1) + y1
		r = np.sqrt(np.square(common_y-lat1)+(np.square(common_x-lon1)))

	else:	#e1||e2

		r = 1

	return(r)

def find_new_coordinates(lon1,lat1,lon2,lat2,action):

	l = -1/((lat2-lat1)/(lon2-lon1))	# l * l_lat1_lon_2 = -1
	#move the point on the line by action

	abs_action = np.absolute(action)
	
	b = 2*l*l*lon1+2*lon1
	delta = b*b - 4*(l*l+1)*(lon1*lon1+lon1*lon1*l*l-abs_action)
	#print(delta)

	if (action==0):
		new_x = lon1
		new_y = lat1
	elif (action>0):
		new_x = (b+np.sqrt(delta))/(2*(l*l+1))
		new_y = l*(new_x-lon1)+lat1
	else:
		new_x = (b-np.sqrt(delta))/(2*(l*l+1))
		new_y = l*(new_x-lon1)+lat1

	return(new_x,new_y)

def find_the_largest_q_value(state):

	maximum_q_value = -100000000000
	counter = 0

	for q_value in state:
		if (q_value>maximum_q_value):	#tends to chose smaller velocities
			maximum_q_value = q_value
			position_max_q_value = counter
		counter+=1

	return(position_max_q_value)

def select_action(Q_table,state,n):

	#n = number of iterations, the age of the agent
	#E = exploitation parameter, 0<E<1

	E = 0.001
	e_greedy_probability =  np.exp(-E*n)

	#a = list, len = 1, if a = 0 then choose the bigger Q_Value else choose a random one
	a = np.random.choice(2, 1, p=[1-e_greedy_probability,e_greedy_probability])
	a = a[0] #first element of list
	
	#action = how ofset from the middle
	if (a==0):
		largest_q_value_position = find_the_largest_q_value(Q_table[state])
		action = largest_q_value_position-3	# 0->-3, 1->-2, 2->-1, 3->0, 4->1, 5->2, 6->3
	else:
		action = random.randint(-(number_of_actions-1)/2,(number_of_actions-1)/2)

	return(action)

def Update_Q_table(q_table,radius_reward,state,action,number_of_states,imediate_reward = 0):

	
	present_q_value = q_table[state][action]
	
	#there is no reason for the discount factor because as a next state there is inly one option
	new_experience = learnig_rate*(radius_reward+imediate_reward-present_q_value)# + discount_factor*next_max_q_value - present_q_value)
	new_q_value = present_q_value + new_experience
	q_table[state][action] = new_q_value

	return(q_table)

def Q_Learnig(Q_table,latitude,longitude):

	change_in_policy = False
	no_change_in_consumption_after_epochs = 0
	policy = [0 for x in range(len(latitude)-2)] #the last 2 are used only for the radious calculation
	policy_radius = [0 for x in range(len(latitude)-2)]

	for element in range(len(policy)):

		x_1 = longitude[element] 
		y_1 = latitude[element]
		x_2 = longitude[element+1]
		y_2 = latitude[element+1]
		x_3 = longitude[element+2]
		y_3 = latitude[element+2]
		policy_radius[element] = find_radius_of_circle_from_3_elements(x_1,y_1,x_2,y_2,x_3,y_3)

	for element in range(len(policy)):

		if (element>1):
			x_1 = longitude[element] 
			y_1 = latitude[element]
			x_2 = longitude[element-1]
			y_2 = latitude[element-1]
			x_3 = longitude[element-2]
			y_3 = latitude[element-2]
			temp_policy_radius = find_radius_of_circle_from_3_elements(x_1,y_1,x_2,y_2,x_3,y_3)
			if (temp_policy_radius>policy_radius[element]):
				policy_radius[element] = temp_policy_radius

	for iterations in range(epochs):
		
		if (mod(iterations,1000)==0):
			print('after epchos = ', iterations)

			if (no_change_in_consumption_after_epochs>=early_stop_patience):

				print()
				print("EARLY STOP")
				print()

				zeros = 0
				for i in Q_table:
					for j in i:
						if (j==0):
							zeros+=1

				print('zeros = ', zeros)
				print('episkefthike = ', ((648-zeros)/648)*100, '%')
				print()

				policy = []
				for po in Q_table:
					policy.append((find_the_largest_q_value(po)-3)*0.000000001)

				return(policy))
		
		for state in range(len(latitude)-2):

			action = 0.000000001*select_action(Q_table,state,iterations)

			new_x,new_y = find_new_coordinates(longitude[state],latitude[state],longitude[state+1],latitude[state+1],action)
			
			if (state<(len(latitude)-4)):	#the last 2 elemnts do not move and they do not hane a policy
				#i calculate the forward radius
				policy_1 = (find_the_largest_q_value(Q_table[state+1])-3)*0.000000001
				policy_next_1_x,policy_next_1_y = find_new_coordinates(longitude[state+1],latitude[state+1],longitude[state+2],latitude[state+2],policy_1)
				policy_1 = (find_the_largest_q_value(Q_table[state+2])-3)*0.000000001
				policy_next_2_x,policy_next_2_y = find_new_coordinates(longitude[state+2],latitude[state+2],longitude[state+1],latitude[state+1],policy_1)
				
			else:
				
				policy_next_1_x,policy_next_1_y = find_new_coordinates(longitude[state+1],latitude[state+1],longitude[state+2],latitude[state+2],0)
				policy_next_2_x,policy_next_2_y = find_new_coordinates(longitude[state+2],latitude[state+2],longitude[state+1],latitude[state+1],0)
				
			next_new_r = find_radius_of_circle_from_3_elements(new_x,new_y,policy_next_1_x,policy_next_1_y,policy_next_2_x,policy_next_2_y)

			policy_1 = (find_the_largest_q_value(Q_table[state])-3)*0.000000001 #in order to culculate the policy radius
			policy_x,policy_y =  find_new_coordinates(longitude[state],latitude[state],longitude[state+1],latitude[state+1],policy_1)
			policy_radius_forward = find_radius_of_circle_from_3_elements(policy_x,policy_y,policy_next_1_x,policy_next_1_y,policy_next_2_x,policy_next_2_y)
			
			if (state>1):	#the first 2 elemnts do not move and they do not hane a policy
				#i calculate the backward radius
				policy_1 = (find_the_largest_q_value(Q_table[state-1])-3)*0.000000001
				policy_previous_1_x,policy_previous_1_y = find_new_coordinates(longitude[state-1],latitude[state-1],longitude[state-2],latitude[state-2],policy_1)
				policy_1 = (find_the_largest_q_value(Q_table[state-2])-3)*0.000000001
				policy_previous_2_x,policy_previous_2_y = find_new_coordinates(longitude[state-2],latitude[state-2],longitude[state-1],latitude[state-1],policy_1)
				
				previous_new_r = find_radius_of_circle_from_3_elements(new_x,new_y,policy_previous_1_x,policy_previous_1_y,policy_previous_2_x,policy_previous_2_y)
				policy_radius_backward = find_radius_of_circle_from_3_elements(policy_x,policy_y,policy_previous_1_x,policy_previous_1_y,policy_previous_2_x,policy_previous_2_y)

				policy_radius_medium = find_radius_of_circle_from_3_elements(policy_x,policy_y,policy_previous_1_x,policy_previous_1_y,policy_next_1_x,policy_next_1_y)
				medium_new_r = find_radius_of_circle_from_3_elements(new_x,new_y,policy_previous_1_x,policy_previous_1_y,policy_next_1_x,policy_next_1_y)

			else:
				previous_new_r = 0 
				policy_radius_backward = 0
				medium_new_r = 0
				policy_radius_medium = 0
			
			#radius_reward = next_new_r - policy_radius_forward + previous_new_r - policy_radius_backward + medium_new_r - policy_radius_medium
			radius_reward = (next_new_r - policy_radius_forward)*1.2 + (medium_new_r - policy_radius_medium)*0.6 + (previous_new_r - policy_radius_backward)*0

			Q_table = list(Update_Q_table(Q_table,radius_reward,state,int((action/0.000000001)+3),len(latitude)-2))
			
			if (next_new_r>medium_new_r):
				new_r = next_new_r
			else:
				new_r = medium_new_r

			if (new_r<previous_new_r):
				new_r = previous_new_r

			if (policy_radius_forward>policy_radius_medium):
				previous_r = policy_radius_forward
			else:
				previous_r = policy_radius_medium

			if (previous_r<policy_radius_backward):
				previous_r = policy_radius_backward
			

			if (new_r>previous_r):

				print('update = ', iterations)
				change_in_policy = True

		if (change_in_policy):	#Early Stop Check
			change_in_policy = False
			no_change_in_consumption_after_epochs = 0
		else:
			no_change_in_consumption_after_epochs += 1

	policy = []
	for po in Q_table:
		policy.append((find_the_largest_q_value(po)-3)*0.000000001)
			

	return(policy)

if __name__ == '__main__':

	with open('Latitude', 'rb') as fp:
		latitude = pickle.load(fp)
	with open('Longitude', 'rb') as fp:
		longitude = pickle.load(fp)

	Q_table = list(Create_Q_Table(len(latitude),number_of_actions))

	policy = list(Q_Learnig(Q_table,latitude,longitude))

	final_latitude = []
	final_longitude = []

	for i in range(len(latitude)-2):
		final_latitude.append(0)
		final_longitude.append(0)

	for element in range(len(latitude)-2):

		final_longitude[element],final_latitude[element] = find_new_coordinates(longitude[element],latitude[element],longitude[element+1],latitude[element+1],policy[element])

	final_longitude = final_longitude[:-1]
	final_latitude = final_latitude[:-1]

	print('Plotting Map')

	plt.figure(1)
	plt.plot(longitude,latitude,'ro',color='g')
	plt.plot(final_longitude,final_latitude,'ro',color='r')
	plt.title('MAP Green: Beginning Line & Red: Racing Line')

	plt.figure(2)
	plt.plot(longitude,latitude,color='g')
	plt.plot(final_longitude,final_latitude,color='r')
	plt.title('MAP Green: Beginning Line & Red: Racing Line')
	plt.show()