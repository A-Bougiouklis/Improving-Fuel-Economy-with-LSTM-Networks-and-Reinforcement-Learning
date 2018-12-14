# Improving-Fuel-Economy-with-LSTM-Networks-and-Reinforcement-Learning

This project presents a system for calculating the optimum
velocities and trajectories of an electric vehicle for a specific route. The
objective is to minimize the consumption over a trip without impacting
the overall trip time. The system uses a particular segmentation of the
route and involves a two-step procedure. In the first step, a neural net-
work is trained on telemetry data to model the consumption of the vehicle
based on its velocity and the surface gradient. In the second step, two Q-
learning algorithms compute the optimum velocities and the racing line
in order to minimize the consumption. This system was installed on a light
electric vehicle (LEV) and by adopting the suggested driving strategy
we reduced its consumption by 24.03% with respect to the classic 
constant-speed control technique.

The LSTM has been made by using the Keras Python3 package with TensorFlow
as a backend.

The correct way of studing the project is the following.
1) Preprocess
2) LSTM Neural Network
3) Optimum Velocity
4) Calculating The Racing Line

I have put into every folder a short documentation of the idea and the code.
