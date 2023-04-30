import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


def createDNN( nInputs, nOutputs, nLayer, nNodes ): 
	# Initialize the neural network
	model = Sequential()
	#
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))

	for _ in range(nLayer - 1):
		model.add(Dense(nNodes, activation="relu"))

	model.add(Dense(nOutputs, activation="softmax"))
	#
	return model


def training_loop( env, neural_net, updateRule, frequency=10, episodes=100 ):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	# initialize the optimizer 
	optimizer = tf.optimizers.Adam()
	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	memory_buffer = [[]]
	for ep in range(episodes):

		# reset the environment and obtain the initial state
		state = env.reset()[0]
		state = np.reshape(state, (-1, 4))
		ep_reward = 0
		while True:

			# select the action to perform
			distribution = neural_net(state).numpy()[0]
			action = np.random.choice(2, p=distribution)


			# Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, _ = env.step(action)
			next_state = np.reshape(next_state, (-1, 4))
			memory_buffer[ep % 10].append((state, action, next_state, reward, terminated))
			ep_reward += reward

			# exit condition for the episode
			if ( terminated or truncated ) == True: 
				break

			# update the current state
			state = next_state

		# Perform the actual training every 'frequency' episodes
		
		if ep % frequency == 0:
			updateRule( neural_net, memory_buffer, optimizer )
			memory_buffer = [[] for _ in range(10)]

		# Update the reward list to return
		reward_queue.append( ep_reward )
		rewards_list.append( np.mean(reward_queue) )
		print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list



def REINFORCE_naive( neural_net, memory_buffer, optimizer ):
	"""
	Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.

	"""

	# Setup the tape
	with tf.GradientTape() as tape:
		# Initialize the array for the objectives, one for each episode considered
		objectives = []
		# Iterate over all the trajectories considered
		for traj in memory_buffer:
			states = []
			actions = []
			rewards = []
			probabilities = []
			# Extract the information from the buffer (for the considered episode)
			for step in traj:
				states.append(step[0])
				actions.append(step[1])
				rewards.append(step[3])

			# Compute the log-prob of the current trajectory
	
			for step in traj:
				probability = neural_net(step[0])[0][step[1]]
				probabilities.append(probability)

			#log_probs = tf.math.log(probabilities)
			log_prob_sum = tf.math.reduce_sum(tf.math.log(probabilities))
			objectives.append(log_prob_sum * sum(rewards))

			# Implement the update rule, notice that the REINFORCE objective 
			# is the sum of the logprob (i.e., the probability of the trajectory)
			# multiplied by the sum of the reward
		objective = -tf.math.reduce_mean(objectives)
		grads = tape.gradient(objective, neural_net.trainable_variables)
		optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))

		# Compute the final final objective to optimize




def REINFORCE_rw2go( neural_net, memory_buffer, optimizer ):
	"""
	Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,

	"""
	with tf.GradientTape() as tape:
		# Initialize the array for the objectives, one for each episode considered
		objectives = []
		
		# Iterate over all the trajectories considered
		for traj in memory_buffer:
			probabilities = []
			states = []
			actions = []
			rewards = []
			reward_per_episode = np.zeros(len(traj))

			# Extract the information from the buffer (for the considered episode)
			for i, step in enumerate(traj):
				states.append(step[0])
				actions.append(step[1])
				rewards.append(step[3])

			# Compute the log-prob of the current trajectory
			reward_per_episode = np.flip(np.cumsum(rewards))

			for i, step in enumerate(traj):
				probability = neural_net(step[0])[0][step[1]]
				probabilities.append(tf.math.log(probability))
				probabilities[-1] *= reward_per_episode[i]

			log_prob_sum = tf.math.reduce_sum(probabilities)
			objectives.append(log_prob_sum)

			# Implement the update rule, notice that the REINFORCE objective 
			# is the sum of the logprob (i.e., the probability of the trajectory)
			# multiplied by the sum of the reward
		objective = -tf.math.reduce_mean(objectives)
		grads = tape.gradient(objective, neural_net.trainable_variables)
		optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))



def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the ninth lesson of the RL-Lab!   *" )
	print( "*                 (REINFORCE)                   *" )
	print( "*************************************************\n" )

	_training_steps = 1500
	env = gymnasium.make( "CartPole-v1" )

	# Training A)
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards_naive = training_loop( env, neural_net, REINFORCE_naive, episodes=_training_steps  )
	print()

	# Training B)
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards_rw2go = training_loop( env, neural_net, REINFORCE_rw2go, episodes=_training_steps  )

	# Plot
	t = np.arange(0, _training_steps)
	plt.plot(t, rewards_naive, label="naive", linewidth=3)
	plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	
