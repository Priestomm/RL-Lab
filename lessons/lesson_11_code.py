import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ):
# Initialize the neural network
    model = Sequential()
    #
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))

    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation="relu"))

    model.add(Dense(nOutputs, activation=last_activation))
    #
    return model

def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ):

    critic_optimizer = tf.optimizers.Adam()
    actor_optimizer = tf.optimizers.Adam()

    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    for ep in range(episodes):

        # reset the environment and obtain the initial state
        state = env.reset()[0]
        state = np.reshape(state, (-1, 2))
        ep_reward = 0
        while True:

            # select the action to perform
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(3, p=distribution)


            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.reshape(next_state, (-1, 2))
            memory_buffer.append((state, action, next_state, reward, terminated))
            ep_reward += reward

            # exit condition for the episode
            if ( terminated or truncated ) == True: 
                break

            # update the current state
            state = next_state

            # Perform the actual training every 'frequency' episodes

        if ep % frequency == 0:
            updateRule( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer )
            memory_buffer = []

        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list

def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """

    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle( memory_buffer )
        memory_buffer = np.asarray(memory_buffer)
        states = np.vstack(memory_buffer[:,0])
        next_states = np.vstack(memory_buffer[:,2])
        rewards = np.vstack(memory_buffer[:,3])
        dones = np.vstack(memory_buffer[:,4])

        # Tape for the critic
        with tf.GradientTape() as critic_tape:
            #TODO: Compute the target and the MSE between the current prediction
            # and the expected advantage    
            target = rewards + (1 - dones.astype(int)) * gamma * critic_net(next_states) 
            prediction = critic_net(states)
            mse = tf.math.square(prediction - target)

            #TODO: Perform the actual gradient-descent process
            grad = critic_tape.gradient(mse, critic_net.trainable_variables)   
            critic_optimizer.apply_gradients( zip(grad, critic_net.trainable_variables) ) 


    # Tape for the actor
    with tf.GradientTape() as actor_tape:
    #TODO: compute the log-prob of the current trajectory and 
    # the objective function, notice that:
    # the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
    # multiplied by advantage
        objectives = []
		
        for step in memory_buffer:
            probability = actor_net(step[0])[0][step[1]]
            log_prob = tf.math.log(probability)
            adv_a = step[3] + gamma * critic_net(step[2]).numpy().reshape(-1)
            adv_b = critic_net(step[0]).numpy().reshape(-1)
            objectives.append(log_prob * (adv_a - adv_b))

        # Implement the update rule, notice that the REINFORCE objective 
        # is the sum of the logprob (i.e., the probability of the trajectory)
        # multiplied by the sum of the reward
        objective = -tf.math.reduce_mean(objectives)
        grads = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))

class OverrideReward( gymnasium.wrappers.NormalizeReward ):
	"""
	Gymansium wrapper useful to update the reward function of the environment

	"""

	def step(self, action):
		previous_observation = np.array(self.env.state, dtype=np.float32)
		observation, reward, terminated, truncated, info = self.env.step( action )
		
		#TODO: extract the information from the observations
		position, velocity = observation
		previous_position, previous_velocity = previous_observation
		#TODO: override the reward function before the return
		if (position - previous_position > 0 and action == 2) or (position - previous_position < 0 and action == 0):
			reward = 1
                    
		if position == 0.6:
			reward = 100
            

		return observation, reward, terminated, truncated, info
	

def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	_training_steps = 2000

	# Crete the environment and add the wrapper for the custom reward function
	gymnasium.envs.register(
		id='MountainCarMyVersion-v0',
		entry_point='gymnasium.envs.classic_control:MountainCarEnv',
		max_episode_steps=1000
	)
	env = gymnasium.make( "MountainCarMyVersion-v0" )
	env = OverrideReward(env)
		
	# Create the networks and perform the actual training
	actor_net = createDNN(2, 3, nLayer=2, nNodes=32, last_activation="softmax" )
	critic_net = createDNN(2, 1, nLayer=2, nNodes=32, last_activation="linear" )
	rewards_training, ep_lengths = training_loop( env, actor_net, critic_net, A2C, frequency=10, episodes=_training_steps  )

	# Save the trained neural network
	actor_net.save( "MountainCarActor.h5" )

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, ep_lengths, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "length", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	