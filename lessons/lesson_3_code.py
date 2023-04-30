import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def on_policy_mc( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
    """
    Performs the on policy version of the every-visit MC control

    Args:
    environment: OpenAI Gym environment
    maxiters: timeout for the iterations
    eps: random value for the eps-greedy policy (probability of random action)
    gamma: gamma value, the discount factor for the Bellman equation
		
    Returns:
    policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """
    p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]   # policy, manca epsilon soft initialization, scegli un azioene a caso per ogni stato e li metti il valore di (1-epsilon)/(epsilon- #_azioni)
    Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    returns = [[[] for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    
    for _ in range(maxiters):

        episode = environment.sample_episode(p)[::-1]
        G = 0 
        for val in episode:
            state,action,reward = val[0],val[1],val[2]
            G = gamma*G + reward 
            returns[state][action].append(G)    
            Q[state][action] = sum(returns[state][action]) / len(returns[state][action])

            actions = []
            for action in range(environment.action_space): 
                actions.append([Q[state][action], action]) 

            best_action = max(actions)[1]
            
            
            for action in range(environment.action_space):
                p[state][action] = (1- eps + (eps/environment.action_space)) if (action == best_action) else (eps/environment.action_space)

    deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]	
    return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!   *" )
	print( "*       (Temporal Difference Methods)           *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy" )
	mc_policy = on_policy_mc( env )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()