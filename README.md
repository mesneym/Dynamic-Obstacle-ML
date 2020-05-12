# Drone Navigation with Reinforcement Learning 

## Overview:
  In this project we train a drone to pass rectangular holes using Deep Qlearning 

## Dependencies
ROS  
Openai-gym  
Gazebo  
Keras/Tensorflow  
Openai-ros

## Run Instructions
To run the code, refer to the instructions below
```
git clone git@github.com:mesneym/Dynamic-Obstacle-ML.git
cd Dynamic-Obstacle-ML/dep_ws
source devel/setup.bash

cd Dynamic-Obstacle/parrot_ws
source devel/setup.bash

rosrun drone_construct start_simulation_localy.sh
```

In a new terminal enter the following 
```
cd Dynamic-Obstacle/training_ws
source devel/setup.bash

roslaunch drone_training main.launch
```

## Plot graph
To plot graph, open a new terminal and enter the following
```
rosrun rqt_multiplot rqt_multiplot
```

Add a curve and select the topic <b>\openai\_reward</b>

## Sample outputs:
Refer to the directory for the videos
Dynamic-Obstacle-ML/training\_ws/src/drone\_training/drone\_training/training\_results/reinforcement\_video


