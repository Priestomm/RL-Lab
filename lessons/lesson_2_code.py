import os, sys
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld
import random as rd
import numpy as np


def value_iteration(environment, maxiters=300, discount=0.9, max_error=1e-3):
	"""
	Performs the value iteration algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		discount: gamma value, the discount factor for the Bellman equation
		max_error: the maximum error allowd in the utility of any state
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	
	U_1 = [0 for _ in range(environment.observation_space)] # vector of utilities for states S
	delta = 0 # maximum change in the utility o any state in an iteration
	U = U_1.copy()

	for i in range(maxiters):
		U = U_1.copy()
		delta = 0

		for s in range(environment.observation_space):
			actionRews = [0 for _ in range(environment.action_space)]
			for a in range(environment.action_space):
				for s_1 in range(environment.observation_space):
					actionRews[a] += environment.transition_prob(s, a, s_1) * U[s_1]
			U_1[s] = environment.R[s] + (discount * max(actionRews))

			if (U_1[s] - U[s]) > delta:
				delta = (U_1[s] - U[s])

		if delta < max_error * (1 - discount) / discount:
			break

	return environment.values_to_policy( U )

	

def policy_iteration(environment, maxiters=300, discount=0.9, maxviter=10):
	"""
	Performs the policy iteration algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		discount: gamma value, the discount factor for the Bellman equation
		maxviter: number of epsiodes for the policy evaluation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	   
	U = [0 for _ in range(environment.observation_space)] #utility array
	p = [0 for _ in range(environment.observation_space)]
	for i in range(maxiters):
		# 1) Policy Evaluation
		for s in range(environment.observation_space):
			summation = 0
			for s_1 in range(environment.observation_space):
				summation += environment.transition_prob(s, p[s], s_1) * U[s_1]
			U[s] = environment.R[s] + (discount * summation)

		unchanged = True

	# 2) Policy Improvement
		for s in range(environment.observation_space):
			actionRews = [0 for _ in range(environment.action_space)]
			policyRew = 0
			for a in range(environment.action_space):
				for s_1 in range(environment.observation_space):
					actionRews[a] += environment.transition_prob(s, a, s_1) * U[s_1]
			for s_1 in range(environment.observation_space):
				policyRew += environment.transition_prob(s, p[a], s_1) * U[s_1]
			if max(actionRews) > policyRew:
				p[s] = np.argmax(actionRews)
				unchanged = False
		if unchanged:
			break  
	
	return p



def main():
	print( "\n************************************************" )
	print( "*  Welcome to the second lesson of the RL-Lab! *" )
	print( "*    (Policy Iteration and Value Iteration)    *" )
	print( "************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n1) Value Iteration:" )
	vi_policy = value_iteration( env )
	env.render_policy( vi_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(vi_policy) )

	print( "\n1) Policy Iteration:" )
	pi_policy = policy_iteration( env )
	env.render_policy( pi_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(pi_policy) )


if __name__ == "__main__":
	main()