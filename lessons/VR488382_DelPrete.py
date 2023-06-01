import warnings; warnings.filterwarnings("ignore")
import sys, os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from point_discrete import PointNavigationDiscrete
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

def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99, observation_number=None ):

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
        rewards = memory_buffer[:,3]
        dones = memory_buffer[:,4]

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
        actions = np.array(list(memory_buffer[:, 1]), dtype=int)


        probabilities = actor_net(states)
        indices = tf.transpose(tf.stack([tf.range(probabilities.shape[0]),actions]))
        probs = tf.gather_nd(
            indices=indices,
            params=probabilities
        )
        log_probs = tf.math.log(probs)
        adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
        adv_b = critic_net(states).numpy().reshape(-1)
        objective = log_probs * (adv_a - adv_b)

        # Implement the update rule, notice that the REINFORCE objective 
        # is the sum of the logprob (i.e., the probability of the trajectory)
        # multiplied by the sum of the reward
        grads = actor_tape.gradient(-tf.math.reduce_sum(objective), actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))

def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	"""

	#TODO: initialize the optimizer 
	optimizer = tf.optimizers.Adam() 
	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	success_list, success_queue = [], collections.deque( maxlen=100 )
	memory_buffer = []
	for ep in range(episodes):

		#TODO: reset the environment and obtain the initial state
		state = env.reset()[0]
		state = np.reshape(state, (-1, 9)) 
		ep_reward = 0
		while True:

			#TODO: select the action to perform
			distribution = actor_net(state).numpy()[0]
			action = np.random.choice(3, p=distribution) 

			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, info = env.step(action)
			next_state = np.reshape(next_state, (-1, 9)) # CHANGE HERE
			memory_buffer.append((state, action, next_state, reward, terminated))
			ep_reward += reward

			#TODO: exit condition for the episode
			if ( terminated or truncated ) == True: 
				break

			#TODO: update the current state
			state = next_state

		#TODO: Perform the actual training every 'frequency' episodes
		if ep % frequency == 0:
			updateRule( actor_net, critic_net, memory_buffer, optimizer, optimizer )
			memory_buffer = []

		if ep == 900:
			env = PointNavigationDiscrete(render_mode = "human" ) #optional: render_mode="human"
			env = OverrideReward(env)

		# Update the reward list to return
		reward_queue.append( ep_reward )
		rewards_list.append( np.mean(reward_queue) )
		success_queue.append( info["goal_reached"] )
		success_list.append( np.mean(success_queue) )
		print( f"episode {ep:4d}: reward: {ep_reward:5.2f} (averaged: {np.mean(reward_queue):5.2f}), success rate ({int(np.mean(success_queue)*100):3d}/100)" )

	# Close the enviornment and return the rewards list
	env.close()
	return success_list



class OverrideReward( gymnasium.wrappers.NormalizeReward ):


	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step( action )

		# Extract the information from the observations
		old_heading, old_distance, old_lidars = self.previous_observation[0], self.previous_observation[1], self.previous_observation[2:]
		heading, distance, lidars = observation[0], observation[1], observation[2:]

		# Exploting useful flags
		goal_reached = bool(info["goal_reached"])
		collision = bool(info["collision"])
		
		# Override the reward function
		if (heading == 0 and action == 0) or (heading < 0 and action == 2) or (heading > 0 and action == 1):
			reward += 1

		if ((old_distance > 0.5) and (distance < 0.5)):
			reward += 1

		if collision:
			reward -= 100
		elif goal_reached:
			reward += 100
		else:
			reward -= 1

		# here!

		return observation, reward, terminated, truncated, info
	

def main(): 
	print( "\n*****************************************************" )
	print( "*    Welcome to the final activity of the RL-Lab    *" )
	print( "*                                                   *" )
	print( "*****************************************************\n" )

	_training_steps = 1000
	
	# Load the environment and override the reward function
	env = PointNavigationDiscrete() #optional: render_mode="human"
	env = OverrideReward(env)

	# Create the networks and perform the actual training
	actor_net = createDNN( 9, 3, nLayer=2, nNodes=32, last_activation="softmax" )
	critic_net = createDNN( 9, 1, nLayer=2, nNodes=32, last_activation="linear" )
	success_training = training_loop( env, actor_net, critic_net, A2C, frequency=5, episodes=_training_steps  )

	# Save the trained neural network
	actor_net.save( "VR488382_DelPrete.h5" )

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, success_training, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "success", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	