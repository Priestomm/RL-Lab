import os, sys, random
import numpy as np 
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def random_dangerous_grid_world( environment ):

	# Args: environment: OpenAI Gym environment
	# Returns: trajectory: an array containing the sequence of states visited by the agent

    goal_state = environment.goal_state
    trajectory = [] 
    state= environment.start_state

    for _ in range(10):
        random_move = random.randint(0,3)
        state = environment.sample(random_move,state)
        trajectory.append(environment.state_to_pos(state))
        if (environment.is_terminal(state)): 
            print("Terminal state reached")
            break 

    return trajectory


class RecyclingRobot():

    def __init__( self ):

        # Loading the default parameters
        self.alfa = 0.7
        self.beta = 0.7
        self.r_search = 0.5 # probabilita reward 
        self.r_wait = 0.2 # probabilita di fare wait 
        self.r_rescue = -3 # probabilita di fare wait 


        # Defining the environment variables
        self.observation_space = 2  
        self.action_space = 2 
        self.actions = {0:"W",1:"S",2:"R"} #{0:"W",1:"S",2:"R"}
        self.states = [0,1] 

        # It would be foolish too recharge if the robot is already high in energy  
        # self.high_actions = {0:"W",1:"S"} 

    def reset(self):
        self.state = 1 # recharge the robot 
        return self.state


    def step( self, action ):
        reward = 0
        # print("Step Action", action,self.state,"\n") 

        if (action == 0): # WAIT 
            reward = self.r_wait

        elif (action==1): # SEARCH  
            if self.state:
                next_state = random.choices([0,1],[self.alfa,(1-self.alfa)])
                self.state = next_state[0]
                reward = self.r_search 
            else:
                rescue = random.choices([0,1],[self.beta,(1-self.beta)])
                reward = self.r_search if rescue[0]== 0 else self.r_rescue
                self.state = rescue[0]


        elif (action == 2): # RECHARGE
            self.state = 1  

        else: # ERROR HANDLING 
            print("\nERROR\n")
             
        return self.state, reward, False, None


    def render( self ):
        # Idea copiata da Farinelli perchè non so come farlo senza mettermi a fare ascii art:
        #outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        #outfile.write(np.array_str(self.grid.reshape(self.rows, self.cols)) + "\n")
        return True


def main():
    print( "\n************************************************" )
    print( "*  Welcome to the first lesson of the RL-Lab!  *" )
    print( "*             (MDP and Environments)           *" )
    print( "************************************************" )
    print( "\nA) Random Policy on Dangerous Grid World:" )
    env = GridWorld()
    env.render()
    random_trajectory = random_dangerous_grid_world( env )
    print( "\nRandom trajectory generated:", random_trajectory )

    print( "\nB) Custom Environment: Recycling Robot" )
    env = RecyclingRobot()
    state = env.reset()
    ep_reward = 0
    for step in range(10):
        a = random.randint( 0, env.action_space )
        # print("Random Action", a) 
        new_state, r, _, _ = env.step( a )
        ep_reward += r
        print( f"\tFrom state '{env.states[state]}' selected action '{env.actions[a]}': \t total reward: {ep_reward:1.1f}" )
        state = new_state

    print("Final Reward:",ep_reward)


if __name__ == "__main__":
    main()