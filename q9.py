# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:42:20 2018

@author: Hitesh_Bahar
"""


import sys

import math

import time

import copy

import random

import operator

import numpy as np



class Logger(object):

    def __init__(self):

        self.terminal = sys.stdout

        self.log = open("terminal_session.log", "w")



    def write(self, message):

        self.terminal.write(message)

        self.log.write(message)  



    def flush(self):

        #this flush method is needed for python 3 compatibility.

        #this handles the flush command by doing nothing.

        #you might want to specify some extra behavior here.

        pass





def egreedy_policy(world, start_pos, goal_pos, beta, alpha, epsilon):

	return EpsiGreedyPolicy(world, start_pos, goal_pos,beta, alpha, epsilon)




def boltzmann_policy(world, start_pos, goal_pos, beta, alpha, temp):

	return BoltzmannExploration(world, start_pos, goal_pos,beta, alpha, temp)

							   
def mat_diff(mat_one, mat_two):

	diff = 0

	for i, row in enumerate(mat_one):



		index_one, value_one = max(enumerate(mat_one[i]),

								   key=operator.itemgetter(1))

		index_two, value_two = max(enumerate(mat_two[i]),

								   key=operator.itemgetter(1))

		if index_one != index_two: diff += 1

	

	return diff							   

							   
def world():



	cols = 10

	rows = 10

	grid = [['0' for col in range(cols)]

			 for row in range(rows)]



	for i in range(1, 5): grid[2][i] = gridworld.wall

	for i in range(6, 9): grid[2][i] = gridworld.wall

	for i in range(3, 8): grid[i][4] = gridworld.wall


	grid[5][5] = '1'





	grid[4][5] = '-1'

	grid[4][6] = '-1'

	grid[5][6] = '-1'

	grid[5][8] = '-1'

	grid[6][8] = '-1'

	grid[7][3] = '-1'

	grid[7][5] = '-1'

	grid[7][6] = '-1'

	

	return gridworld(grid)

	
	

class Policy:



	def __init__(self, world, start, goal, beta, alpha):

		act = ['up', 'right', 'down', 'left']

		num_s = world.num_row * world.num_col

		num_act = len(act)



		self.world = world

		self.start = copy.deepcopy(start)

		self.goal = copy.deepcopy(goal)

		self.pos = copy.deepcopy(start)

		self.act = act

		self.Q = np.zeros((num_s, num_act))

		self.R = np.zeros((world.num_row, world.num_col))

		self.beta = beta

		self.alpha = alpha



	def reset(self):

		self.pos = copy.deepcopy(self.start)


	def next(self):	

		raise NotImplementedError



	def action_to_pos(self, act):

		pos = copy.deepcopy(self.pos)

		if act == self.act[0]:

			if pos[0] != 0: pos[0] -= 1

		elif act == self.act[1]:

			if pos[1] != (self.world.num_col - 1): pos[1] += 1

		elif act == self.act[2]:

			if pos[0] != (self.world.num_row - 1): pos[0] += 1

		elif act == self.act[3]:

			if pos[1] != 0: pos[1] -= 1

		else:

			raise RuntimeError

		if self.world.get_cell(pos) == gridworld.wall:

			pos = self.pos

		return pos

	def best_action(self, pos):

		location = pos[0] * self.world.num_col + pos[1]

		moves = self.Q[location]


		action_dictionary = {}

		for move_i, move_q in enumerate(moves):

			action_dictionary[move_i] = move_q

		action_items = list(action_dictionary.items())

		random.shuffle(action_items)

		highest_move = -1

		highest_move_index = -1

		for move_i, move_q in action_items:

			if move_q > highest_move:

				highest_move = move_q

				highest_move_index = move_i



		return highest_move_index, highest_move


	def Q_matrix_pos(self, pos):

		return pos[0] * self.world.num_col + pos[1]

	def reward(self, pos):

		c = float(self.world.grid[pos[0]][pos[1]])

		self.R[pos[0]][pos[1]] = c



		return c

		
		

	def print_policy(self):

		traverse = ['u', 'R', 'D', 'L']



		title = "Path\n"

		pad_len = 3

		this_str = ""

		sep = "*" * ((len(self.world.grid) + pad_len + 1) * 2) + "\n"

	

		for i, row in enumerate(self.world.grid):

			for j, cell in enumerate(row):



				new_elem = ""





				if cell == gridworld.wall:

					new_elem = " "

				elif [i, j] == self.goal:



					new_elem += gridworld.goal

				elif self.world.get_cell([i, j]) == "-1":



					new_elem += gridworld.bad

				else:









					a_i, a_q = self.best_action([i, j])

					new_elem += traverse[a_i]





				new_elem += " " * (pad_len - len(new_elem))

				this_str += new_elem

			this_str += "\n"

		this_str = title + sep + this_str + sep

			

		return this_str



class EpsiGreedyPolicy(Policy):

	

	def __init__(self, world, start_pos, goal_pos, beta,

				 alpha, epsilon):

		Policy.__init__(self, world, start_pos, goal_pos, beta,

						alpha)

		self.epsilon = epsilon

		

	def next(self):

		move = -1

		location = self.Q_matrix_pos(self.pos)

		moves = self.Q[location]

		



		best_move_index, best_move_q = self.best_action(self.pos)

		prob = random.uniform(0,1)

		if prob < self.epsilon: 

			move = random.randint(0,3)

		else:

			move = best_move_index


		a = self.act[move] 

		

		self.pos = self.action_to_pos(a) 

		

		

		q = self.Q[location][move] 

		reward = self.reward(self.pos) 

		_, q_hat = self.best_action(self.pos) 

		



		self.Q[location][move] = q + self.alpha * (reward + self.beta * (q_hat) - q)

		



		if self.pos == self.goal:

			return False

		else:

			return True


class gridworld:


	wall = "_"

	pos = "*"

	goal = "+"

	bad = "-"



	def __init__(self, grid):

		self.grid = grid

		self.num_row = len(grid)

		self.num_col = len(grid[0])

		

	def get_cell(self, pos):

		return self.grid[pos[0]][pos[1]]

	



	def print_world(self):



		max_width = 0

		for row in self.grid:

			for cell in row:

				if len(cell) > max_width:

					max_width = len(cell)


		title = "10 X 10 Q learning Grid World \n"

		this_str = ""

		sep = "*" * ((len(self.grid) + 4) * max_width) + "\n"

		for i, row in enumerate(self.grid):

			for j, cell in enumerate(row):

				pad = (max_width - len(cell) + 1) * " "

				this_str += cell

				this_str += pad

			this_str += "\n"

		this_str = title + sep + this_str + sep
	
		return this_str
		

class BoltzmannExploration(Policy):



	def __init__(self, world, start_pos, goal_pos, beta, alpha, temp):

		Policy.__init__(self, world, start_pos, goal_pos, beta, alpha)

		self.temp = temp

	



	def reset(self):


		if self.temp >= 1: self.temp -= 1

		Policy.reset(self)

		

	def next(self):

		move = -1

		location = self.Q_matrix_pos(self.pos)

		moves = self.Q[location]

		

		if self.temp > 0:

			num_probs = []

			denom_probs = []

			

			for m in moves:

				num = math.exp(m/self.temp)

				a_hat = self.act[int(m)]

				pos_new = self.action_to_pos(a_hat)

				pos_new_index = self.Q_matrix_pos(pos_new)

				moves_hat = self.Q[pos_new_index]

				denom = 0

				for mh in moves_hat:

					denom += math.exp(mh/self.temp)

				denom_probs.append(denom)

				num_probs.append(num)

				

			action_probs = [num_probs[i]/denom_probs[i] for i in range(0, len(moves))]

			



			rand_prob = random.uniform(0,1)

			prob_sum = 0

			for i, prob in enumerate(action_probs):

				prob_sum += prob 

				if rand_prob <= prob_sum:

					move = i

					break

			


			if move == -1:

				move, _ = self.best_action(self.pos)

		else:



			move, _ = self.best_action(self.pos)



		a = self.act[move] 
		

		self.pos = self.action_to_pos(a) 

		q = self.Q[location][move] 

		reward = self.reward(self.pos) 

		_, q_hat = self.best_action(self.pos) 



		self.Q[location][move] = q + self.alpha * (reward + self.beta * (q_hat) - q)

		



		if self.pos == self.goal:

			return False

		else:

			return True

	

if __name__ == "__main__":



	sys.stdout = Logger()

	world = world()



	beta = 0.9 

	alpha = 0.01 

	start = [0, 0] 

	goal = [5, 5] 

	

	pol = []

	converge_thresh = 0 

	converge_count = 100 

	

	policy_fun = [egreedy_policy,boltzmann_policy]

	policy_par = [("epsilon", [0.1, 0.2, 0.3]),("temp", [1000, 100, 10, 1])]

	total_time = 0

	print(world.print_world(),file = f)

	for i, policy_fun in enumerate(policy_fun):

		par_name, par_list = policy_par[i]

		for param in par_list:

			policy = policy_fun(world, start, goal, beta, alpha, param)

			print("Running......... \n" )
			print( str(policy.__class__.__name__) + " with " + par_name + " = " + str(param))

			num_iter = 0

			this_conv_count = converge_count

			ts = time.clock()

			while True:

				num_iter += 1

				last_q_matrix = copy.deepcopy(policy.Q)


				while policy.next() == True: pass

				policy.reset()


				if mat_diff(last_q_matrix, policy.Q) <= converge_thresh:

					this_conv_count -= 1

					if this_conv_count == 0: 

						if isinstance(policy, BoltzmannExploration) == True:

							policy.temp = param

						break



				else: this_conv_count = converge_count

			te = time.clock()

			total_time += te - ts

			print("Time completed in  {0:.2f} sec and {1} iterations".format(te-ts, num_iter - converge_count))


			pol.append(policy)



	

	print("Finished ")

	

	print()




	for p in pol:

		par_name = ''

		param = 0

		if isinstance(p, BoltzmannExploration):

			par_name = "temp"

			param = p.temp

		else:

			par_name = "epsilon"

			param = p.epsilon

			

		p_str = str(p.__class__.__name__) + ' at ' + par_name + ' = ' + str(param)

		

		

		print(p.print_policy())
        
        # Writes policy to csv file and prints the policy

	for p in pol:

		name = ''

		param = 0

		if isinstance(p, BoltzmannExploration):

			name = "temperature"

			param = p.temperature

		else:

			name = "epsilon"

			param = p.epsilon

			

		p_str = str(p.__class__.__name__) + ' at ' + name + ' = ' + str(param)

		print("Writing  to csv file.")

		p.save_q_matrix(p_str)

		print(p.print_policy())
