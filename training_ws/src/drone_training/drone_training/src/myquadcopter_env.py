#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np
import tf
import time
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
# from hector_uav_msgs.msg import Altimeter
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection

#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='myquadcopter_env:QuadCopterEnv',
    max_episode_steps=5000,
    # timestep_limit=100,
    )


class QuadCopterEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.takeoff_pub = rospy.Publisher('/drone/takeoff', EmptyTopicMsg, queue_size=0)
        
        # gets training parameters from param server
        self.speed_value = rospy.get_param("/speed_value")
        desired_pose1 = Pose()
        desired_pose2 = Pose()
        desired_pose3 = Pose()
        desired_pose4 = Pose()

        desired_pose1.position.z = rospy.get_param("/desired_poses/z1")
        desired_pose1.position.x = rospy.get_param("/desired_poses/x1")
        desired_pose1.position.y = rospy.get_param("/desired_poses/y1")
        desired_pose2.position.z = rospy.get_param("/desired_poses/z2")
        desired_pose2.position.x = rospy.get_param("/desired_poses/x2")
        desired_pose2.position.y = rospy.get_param("/desired_poses/y2")
        desired_pose3.position.z = rospy.get_param("/desired_poses/z3")
        desired_pose3.position.x = rospy.get_param("/desired_poses/x3")
        desired_pose3.position.y = rospy.get_param("/desired_poses/y3")
        desired_pose4.position.z = rospy.get_param("/desired_poses/z4")
        desired_pose4.position.x = rospy.get_param("/desired_poses/x4")
        desired_pose4.position.y = rospy.get_param("/desired_poses/y4")

        self.desired_poses = [desired_pose1,desired_pose2,desired_pose3,desired_pose4]
        self.reward_checked = [False, False, False, False]

        self.forward_reward = rospy.get_param("/forward_reward")
        self.turn_reward = rospy.get_param("/turn_reward")
        self.up_reward = rospy.get_param("/up_reward")
        self.down_reward = rospy.get_param("/down_reward")
        self.goal_reward = rospy.get_param("/goal_reward")

        self.running_step = rospy.get_param("/running_step")
        self.max_incl = rospy.get_param("/max_incl")
        self.max_altitude = rospy.get_param("/max_altitude")
        self.min_x = rospy.get_param("/min_x")
        self.min_y = rospy.get_param("/min_y")
        self.max_x = rospy.get_param("/max_x")
        self.max_y = rospy.get_param("/max_y")
       
        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        
        self.action_space = spaces.Discrete(5) #Forward,Left,Right,Up,Down
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def _reset(self):
        
        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection()
        self.init_desired_pose()
        self.takeoff_sequence()

        # 4th: takes an observation of the initial condition of the robot
        data_pose, data_imu = self.take_observation()
        observation = [data_pose.position.x,data_pose.position.y,data_pose.position.z, \
                       data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z]
        
        # 5th: pauses simulation
        self.gazebo.pauseSim()
        self.reward_checked = [False, False, False, False]

        return observation

    def _step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        
        # 1st, we decide which velocity command corresponds
        vel_cmd = Twist()
        if action == 0: #FORWARD
            vel_cmd.linear.x = self.speed_value
            vel_cmd.angular.z = 0.0
        elif action == 1: #LEFT
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = self.speed_value
        elif action == 2: #RIGHT
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -self.speed_value
        elif action == 3: #Up
            vel_cmd.linear.z = self.speed_value
            vel_cmd.angular.z = 0.0
        elif action == 4: #Down
            vel_cmd.linear.z = -self.speed_value
            vel_cmd.angular.z = 0.0

        # Then we send the command to the robot and let it go
        # for running_step seconds
        self.gazebo.unpauseSim()
        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)
        data_pose, data_imu = self.take_observation()
        self.gazebo.pauseSim()

        # finally we get an evaluation based on what happened in the sim
        reward,done = self.process_data(data_pose, data_imu)

        # Promote going forwards instead if turning
        if action == 0:
            reward -= self.forward_reward 
        elif action == 1 or action == 2:
            reward -= self.turn_reward 
        elif action == 3:
            reward -= self.up_reward
        else:
            reward -= self.down_reward

        state = [data_pose.position.x,data_pose.position.y,data_pose.position.z, \
                 data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z]

        return state, reward, done, {}


    def take_observation (self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = rospy.wait_for_message('/drone/gt_pose', Pose, timeout=5)
            except:
                rospy.loginfo("Current drone pose not ready yet, retrying for getting robot pose")

        data_imu = None
        while data_imu is None:
            try:
                data_imu = rospy.wait_for_message('/drone/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Current drone imu not ready yet, retrying for getting robot imu")
        
        return data_pose, data_imu



    def init_desired_pose(self):
        current_init_pose, imu = self.take_observation()
        self.best_dist = self.calculate_dist(current_init_pose.position, self.desired_poses)
    

    def check_topic_publishers_connection(self):
        rate = rospy.Rate(10) # 10hz
        while(self.takeoff_pub.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Takeoff yet so we wait and try again")
            rate.sleep();
        rospy.loginfo("Takeoff Publisher Connected")

        while(self.vel_pub.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Cmd_vel yet so we wait and try again")
            rate.sleep();
        rospy.loginfo("Cmd_vel Publisher Connected")
        

    def reset_cmd_vel_commands(self):
        # We send an empty null Twist
        vel_cmd = Twist()
        vel_cmd.linear.z = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)


    def takeoff_sequence(self, seconds_taking_off=1):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts
        self.reset_cmd_vel_commands()
        
        takeoff_msg = EmptyTopicMsg()
        rospy.loginfo( "Taking-Off Start")
        self.takeoff_pub.publish(takeoff_msg)
        time.sleep(seconds_taking_off)
        rospy.loginfo( "Taking-Off sequence completed")
        
    
    def distance(self,p_init,p_end):
        a = np.array((p_init.x, p_init.y, p_init.z))
        b = np.array((p_end.position.x,p_end.position.y,p_end.position.z))
        dist = np.linalg.norm(a-b)
        return dist 

    def calculate_dist(self,p_init,p_ends):
        a = np.array((p_init.x, p_init.y, p_init.z))
        mindist = np.inf
        for p_end in p_ends:
            b = np.array((p_end.position.x,p_end.position.y,p_end.position.z))
            dist = np.linalg.norm(a-b)
            if(dist<mindist):
                mindist = dist 
        return mindist 

    def improved_distance_reward(self, current_pose):
        current_dist = self.calculate_dist(current_pose.position, self.desired_poses)
        #rospy.loginfo("Calculated Distance = "+str(current_dist))
        threshold = 0.75

        for i in range(4):
             if(self.distance(current_pose.position,self.desired_poses[i]) <= threshold and not self.reward_checked[i]):
                 reward = self.goal_reward
                 self.reward_checked[i] = True

                 for j in range(4):
                     if(j != i):
                         self.reward_checked[j] = False
                 break
        else:
            reward = -1
            #print "Made Distance bigger= "+str(self.best_dist)
        return reward
       

    def process_data(self, data_position, data_imu):
        done = False
        
        euler = tf.transformations.euler_from_quaternion([data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z,data_imu.orientation.w])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad = data_position.position.z > self.max_altitude
        position_bad = data_position.position.x < self.min_x or data_position.position.y < self.min_y  or \
                       data_position.position.x > self.max_x or data_position.position.y > self.max_y
                 
        if altitude_bad or pitch_bad or roll_bad or position_bad:
            rospy.loginfo ("(Drone flight status is wrong) >>> ("+str(altitude_bad)+","+str(pitch_bad)+","+str(roll_bad)+"," + str(position_bad) +")")
            done = True
            reward = -100
        else:
            reward = self.improved_distance_reward(data_position)

        return reward,done

