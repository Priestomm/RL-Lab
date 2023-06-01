import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
# Notice that the value function has only one output with a linear activation
# function in the last layer
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
        state = np.reshape(state, (-1, 4))
        ep_reward = 0
        while True:

            # select the action to perform
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(2, p=distribution)


            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.reshape(next_state, (-1, 4))
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

    #TODO: implement the update rule for the critic (value function)
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


    #TODO: implement the update rule for the actor (policy function)
    #TODO: extract the information from the buffer for the policy update
    # Tape for the actor
    objectives = []
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
#TODO: compute the final objective to optimize, is the average between all the considered trajectories



def main(): 
    print( "\n*************************************************" )
    print( "*  Welcome to the tenth lesson of the RL-Lab!   *" )
    print( "*                    (A2C)                      *" )
    print( "*************************************************\n" )

    _training_steps = 2500

    env = gymnasium.make( "CartPole-v1" )
    actor_net = createDNN( 4, 2, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN( 4, 1, nLayer=2, nNodes=32, last_activation="linear")
    rewards_naive = training_loop( env, actor_net, critic_net, A2C, episodes=_training_steps  )

    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_naive, label="A2C", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()	

