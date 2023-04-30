import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections

# Setting the seeds
SEED = 15
np.random.seed(SEED); tf.random.set_seed(SEED)


def createDNN( nInputs, nOutputs, nLayer, nNodes ):
	"""
	Function that generates a neural network with the given requirements.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	
	# Initialize the neural network
	model = Sequential()
	#
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))

	for _ in range(nLayer - 1):
		model.add(Dense(nNodes, activation="relu"))

	model.add(Dense(nOutputs, activation="linear"))
	#
	return model


def mse( network, dataset_input, target ):
	"""
	Compute the MSE loss function

	"""
	
	# Compute the predicted value, over time this value should
	# looks more like to the expected output (i.e., target)
	predicted_value = network( dataset_input )
	
	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse


def training_loop( env, neural_net, updateRule, eps=1, episodes=100, updates=1 ):
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

	optimizer = tf.keras.optimizers.Adam(lr=0.001)
	rewards_list, memory_buffer = [], collections.deque( maxlen=1000 )
	averaged_rewards = []
	for ep in range(episodes):

		# reset the environment and obtain the initial state
		state = env.reset()[0]
		state = np.reshape(state, (-1,4))
		ep_reward = 0
		while True:

			# select the action to perform
			action = env.action_space.sample() 

			# Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, _ = env.step(action)
			next_state = np.reshape(next_state, (-1, 4))
			memory_buffer.append((state, action, next_state, reward, terminated))
			ep_reward += reward

			# Perform the actual training
			updateRule( neural_net, memory_buffer, optimizer )

			# exit condition for the episode
			if ( terminated or truncated ) == True: 
				break

			# update the current state
			state = next_state

		# Update the reward list to return
		rewards_list.append( ep_reward )
		averaged_rewards.append( np.mean(rewards_list) )
		print( f"episode {ep:2d}: rw: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f}" )

	# Close the enviornment and return the rewards list
	env.close()
	return averaged_rewards


def DQNUpdate( neural_net, memory_buffer, optimizer, batch_size=32, gamma=0.99 ):

	"""
	Main update rule for the DQN process. Extract data from the memory buffer and update 
	the newtwork computing the gradient.

	"""

	if len(memory_buffer) < batch_size: return

	indices = np.random.randint( len(memory_buffer), size=batch_size)
	for idx in indices: 

		# extract data from the buffer 
		state, action, next_state, reward, done = memory_buffer[idx]
		
		# compute the target for the training

		target = neural_net(state).numpy()[0]

		if done == True:
			target[action] = reward
		else:
			max_q = max(neural_net(next_state)).numpy()[0]
			target[action] = reward + (max_q * gamma)

		# compute the gradient and perform the backpropagation step
		with tf.GradientTape() as tape:
			objective = mse( neural_net, state, target )
			grad = tape.gradient(objective, neural_net.trainable_variables)
			optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the eighth lesson of the RL-Lab!   *" )
	print( "*               (Deep Q-Network)                 *" )
	print( "**************************************************\n" )

	_training_steps = 100

	env = gymnasium.make( "CartPole-v1" )#, render_mode="human" )
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards = training_loop( env, neural_net, DQNUpdate, episodes=_training_steps  )

	t = np.arange(0, _training_steps)
	plt.plot(t, rewards, label="eps: 0", linewidth=3)
	plt.xlabel( "episodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.show()


if __name__ == "__main__":
	main()	