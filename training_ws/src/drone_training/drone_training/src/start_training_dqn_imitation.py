#!/usr/bin/env python

import numpy                                        #import useful libraries
import random
import time
import gym

import dqn                                          #import reinforcement learning code
# import qlearn

import myquadcopter_env                             #import training environment

import rospy                                        #Required ros packages
import rospkg
from drone_training.msg import RewardInfo


def publish_reward_topic(pub, reward, episode_number):
        rewardInfo = RewardInfo()
        rewardInfo.episode = episode_number
        rewardInfo.reward =  reward
        pub.publish(rewardInfo)

def print_status(start_time,alpha,gamma,reward,epsilon,episode_number):
     m, s = divmod(int(time.time() - start_time), 60)
     h, m = divmod(m, 60)
     rospy.loginfo ( ("EP: "+str(episode_number+1)+" - [alpha: "+str(round(alpha,2))+" - gamma: "+str(round(gamma,2))+ " - epsilon: "+ 
                       str(round(epsilon,2))+"] - Reward: "+str(reward)+ "     Time: %d:%02d:%02d" % (h, m, s))) 


def takeInput(x,state):
    action = 0
    if(x<10):
       action = agent.choose_action(state,agent.get_epsilon(x))          #agent takes an action
    else:
        #take input from user
        pass
    return action


if __name__=='__main__':
    rospy.init_node('drone_gym', anonymous=True)                          #Initialize Node
    pub = rospy.Publisher('/openai/reward', RewardInfo,queue_size = 10)   #setting reward


    env = gym.make('QuadcopterLiveShow-v0')                               #Create the Gym environment
    rospy.loginfo ( "Gym environment done")
      

    Alpha = rospy.get_param("/alpha")                                     #load learning parameters
    Epsilon = rospy.get_param("/epsilon")
    Gamma = rospy.get_param("/gamma")
    Epsilon_Discount = rospy.get_param("/epsilon_discount")

    nepisodes = rospy.get_param("/nepisodes")                             #load training parameters
    nsteps = rospy.get_param("/nsteps")


    last_time_steps = numpy.ndarray(0)                                    #Keeping traack of reward, and time
    highest_reward = 0
    start_time = time.time()
    

    state_dim = 6                                                         #position and orientation of robot
    action_dim = 5                                                        #up,down,forward,left,right

    agent = dqn.DQNAgent(action_dim,
                         state_dim,
                         Gamma,
                         Alpha,
                         Epsilon,
                         Epsilon_Discount
                        ) 

    for x in range(nepisodes):
         rospy.loginfo("STARTING Episode #"+str(x))
         
         observation = env.reset()                                         #Reset robot to original state
         state = agent.preprocess_state(observation) 

         done = False
         cumulate_reward = 0

         # i = 1
         # while not done:
         for i in range(nsteps):                 
             # action = agent.choose_action(state,agent.get_epsilon(x))          #agent takes an action
             action = takeInput(x,state)

             observation,reward,done,info = env.step(action)                   #make observation and update reward and retrieve
             next_state = agent.preprocess_state(observation) 

             cumulate_reward += reward                                     
             if highest_reward < cumulate_reward:
                 highest_reward = cumulate_reward

             if not(done):
                 agent.memorize(state,action,reward,next_state,done)           #save (s,a,r,s')
                 state = next_state                                            #update state
             else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break  
             
             # i += 1
         agent.replay(batch_size=100)
         
         print_status(start_time,agent.alpha,agent.gamma,cumulate_reward,agent.epsilon,x+1)
         publish_reward_topic(pub,cumulate_reward,x+1)


    l = last_time_steps.tolist()                                           #Output overall score
    l.sort()

    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
  
             


    
     

    
    



