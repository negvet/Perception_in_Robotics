import numpy as np
from scipy import array, linalg, dot
import itertools
from scipy.linalg import cholesky

from tools.task import get_motion_noise_covariance, get_gaussian_statistics_xy
from tools.task import get_prediction, wrap_angle, get_observation, get_observation_b, get_inverse_observation
from tools.jacobian import observation_jacobian



class SAM():
	def __init__(self, beta, alphas, initial_position):
		# Noise params
		self._alphas = alphas # motion noise
		self._Q = np.diag([ beta[0]**2, np.deg2rad(beta[1])**2 ]) # observation noise
		# print('initial_position.mu', initial_position.mu.reshape(1,3))
		# self.initial_position = initial_position.mu[:,0]
		self.initial_position = initial_position.mu.squeeze()
		self.robot_positions = []#reshape(1,3)
		# print((self.robot_positions))

		self.motions = []
		self.observations = []

		self.landmarks_id_list = []
		self.landmarks = []

		self.lm_positions = []
		# self.lm0_positions_mu = np.array([0,0])
		# self.lm0_positions_sigma = np.array([ [0,0],[0,0] ])

		self.lm_correspondences = []

	def predict(self, u): # Include the zero robot_postion into the graph]
		if  len(self.robot_positions) == 0:
			new_position = get_prediction(self.initial_position, u)
		else:
			new_position = get_prediction(self.robot_positions[-1], u)
		self.robot_positions.append(new_position)
		self.motions.append(u) # add new motion to self.motions



	def update(self, z):


		z = z.copy()
		# print('\n\ngot observation',z)
		# Check the new observation
		for idx,single_lm in enumerate(z):
			lm_id = single_lm[2]
			if lm_id not in self.landmarks_id_list: # if not observed before
				# print('add new landmark', lm_id)
				self.landmarks_id_list.append(lm_id)
				self.landmarks.append(get_inverse_observation(self.robot_positions[-1], single_lm[:2]))
				self.lm_positions.append([])
				self.lm_correspondences.append([int(z[idx, 2]), self.landmarks_id_list.index(z[idx, 2])])
			# single_lm[2] = self.landmarks_id_list.index(single_lm[2])
			# print('lm_id before the update',z[idx, 2])
			z[idx, 2] = self.landmarks_id_list.index(z[idx, 2]) # change ids of lm to be in the appearance order
			# print('lm_id after the update',z[idx, 2])

			# if z[idx, 2] == 0:
			self.lm_positions[int(z[idx, 2])].append(get_inverse_observation(self.robot_positions[-1], single_lm[:2]))


		# print(self.lm_correspondences.append([z[idx, 2], self.landmarks_id_list.index(z[idx, 2])]))


		# print('\n\nself.lm_positions', self.lm_positions)
		# print(np.array(self.lm_positions).shape)






		# # add new observation to self.observations
		# print('append bservation ',z)
		self.observations.append(z)



		# DEFINE a state vector (from positions and observations) ####

		# Nodes (columns) of the A matrix
		self.number_of_robot_pos_nodes = (1+len(self.robot_positions)) * 3 # number_of_movemtns * number_of_DOF
		self.number_of_landmark_nodes = len(self.landmarks) * 2
		self.number_of_nodes = self.number_of_robot_pos_nodes + self.number_of_landmark_nodes

		# Factors (rows) of the A matrix
		number_of_motion_factors = (1+len(self.motions))*3
		number_of_observation_factors = len(self.observations)*2*2 # self.observations.shape[0] * self.observations.shape[1] * 2
		self.number_of_factors = number_of_motion_factors + number_of_observation_factors

		# create state vector from scratch every time we solve SAM
		self.state = np.zeros(self.number_of_nodes)
		self.state[:3] = self.initial_position
		# state parameter for robot position
		for i in range(len(self.motions)):
			self.state[3 * i + 3: 3 * i + 6] = self.robot_positions[i]
		# state parameters for landmarks
		for i in range(len(self.landmarks)):
			self.state[self.number_of_robot_pos_nodes + 2 * i:self.number_of_robot_pos_nodes + 2 * i + 2] = self.landmarks[i]




		###   ###
		self.motions = np.array(self.motions)
		self.observations = np.array(self.observations)


		# print('self.robot_positions',self.robot_positions)
		# print('self.motions',self.motions)
		# print('self.landmarks',self.landmarks)
		# print('\nself.state', self.state)
		# print('self.observations',self.observations)

		# Optimize
		self.solve_sam()

		self.robot_positions = list(self.get_path())
		self.landmarks = list(self.get_lm())

		# print('\n\nself.landmarks',np.array(self.landmarks))
		# self.lm_positions[int(z[idx, 2])].append(get_inverse_observation(self.robot_positions[-1], single_lm[:2]))


		# print('\n\n',self.landmarks)
   
		###   ###
		self.motions = list(self.motions)
		self.observations = list(self.observations)

		# UPDATE self.initial_position as well ###
		self.initial_position = self.state[:3]


		# self.trajectory = list(self.get_path())
		# self.landmarks = list(self.get_lm())

		# print('\n\nself.robot_positions',self.robot_positions)



		return self.robot_positions.copy(), self.landmarks.copy()









	def adjacency_matrix(self):
		# Empty adjacency matrix
		A = np.zeros((self.number_of_factors, self.number_of_nodes))
		# print('\nA',A)
		# Fill diagonal with non zero values,to get psd matrix, to solve cholesky
		# A[:self.number_of_nodes, :self.number_of_nodes] =  np.eye(self.number_of_nodes) # 
		A[:3, :3] = - 1e0 * np.eye(3)

		# print('\nA',A)

		# Update A, odometry factors rows, for states columns (not for map columns)
		for i in range(self.motions.shape[0]):
			W = self.get_pre_mult_matrix_motion(self.state[3 * i: 3 * i + 3], self.motions[i])
			
			idx = 3*i+3
			A [idx : idx + 3, idx : idx + 3] = W @ (-np.eye(3))
			V = self.get_g_prime_wrt_state(self.state[idx - 3: idx], self.motions[i])
			A [idx : idx + 3, idx - 3 : idx] = W @ V

		# print('\nA odom factors',A)

		# Update A, observations factor rows.
		for (i, j) in itertools.product(range(self.observations.shape[0]), range(self.observations.shape[1])):
			x = self.state[3*i+3:3*i+6]
			lm_idx = self.number_of_robot_pos_nodes + 2 * int(self.observations[i, j, 2])
			m = self.state[lm_idx:lm_idx + 2]
			W = self.get_pre_mult_matrix_observation()
			# print(W)
			# print('\nbefore J',x,m)
			J_x, J_m = observation_jacobian(x, m)
			factor_idx = self.number_of_robot_pos_nodes + 2 * (i * self.observations.shape[1] + j)
			A[factor_idx:factor_idx + 2, 3*i+3:3*i+6] = W @ J_x
			A[factor_idx:factor_idx + 2, lm_idx:lm_idx + 2] = W @ J_m
		
		# print('\nA.shaep',A.shape)
		# print('\nA',A)

		return A

	def b_matrix(self):
		state = self.state.copy()

		b = np.zeros(self.number_of_factors)
		
		# b[:3] = self.initial_state - self.state[:3]

		for i in range(self.motions.shape[0]):
			odom_id = 3*i+3
			W = self.get_pre_mult_matrix_motion(state[3 * i: 3 * i + 3], self.motions[i])
			new_robot_state = get_prediction(state[odom_id - 3:odom_id], self.motions[i])
			diff = new_robot_state - state[odom_id:odom_id + 3]
			diff[2] = wrap_angle(diff[2])
			b[odom_id:odom_id + 3] = W @ diff
			
		# print(b)

		for (i, j) in itertools.product(range(self.observations.shape[0]), range(self.observations.shape[1])):
			odom_id = 3 * i + 3
			x = state[odom_id:odom_id + 3]
			lm_idx = self.number_of_robot_pos_nodes + 2 * int(self.observations[i, j, 2])
			m = state[lm_idx:lm_idx + 2]
			W = self.get_pre_mult_matrix_observation()
			new_observation = get_observation_b(x, m)
			diff = new_observation - self.observations[i, j, :2]
			diff[1] = wrap_angle(diff[1])
			factor_idx = self.number_of_robot_pos_nodes + 2 * (i * self.observations.shape[1] + j)
			b[factor_idx:factor_idx + 2] = W @ diff
		
		# print(b)

		return b

	def solve_linear(self,R, y, right=True):
		x = np.zeros(y.shape[0])
		if right:
			for i in range(y.shape[0] - 1, -1, -1):
				x[i] = (y[i] - np.sum(R[i] * x)) / R[i, i]
		else:
			for i in range(y.shape[0]):
				x[i] = (y[i] - np.sum(R[i] * x)) / R[i, i]
		return x

	def solve_sam(self):
		A = self.adjacency_matrix()
		b = self.b_matrix()

		# print('\n\nA',A)
		# print('\n\nb',b)

		R = linalg.cholesky(A.T @ A)
		# R = linalg.cholesky(A.T @ A, lower=True)
		y = self.solve_linear(R.T, A.T @ b, False)
		delta = self.solve_linear(R, y)
		self.state = self.state - delta
		
	def get_lm(self):
		landmarks = np.zeros((len(self.landmarks), 2))
		for i in range(len(self.landmarks)):
			# n_state_factors = 3 * (1 + self.motions.shape[0])
			landmarks[i] = self.state[self.number_of_robot_pos_nodes + 2 * i: self.number_of_robot_pos_nodes + 2 * i + 2]
		return landmarks
	
	def get_path(self):
		path = np.zeros((self.motions.shape[0], 3))
		for i in range(self.motions.shape[0]):
			path[i] = self.state[3 + i * 3:3 * i + 6]
		return path

	def get_pre_mult_matrix_motion(self, state, motion):
		Vx = self.get_g_prime_wrt_motion(state, motion)
		Mx = get_motion_noise_covariance(motion, self._alphas)
		R = Vx @ Mx @ Vx.T + 1e-8 * np.eye(3) #???
		# L = linalg.cholesky(R, lower=True)
		# pre_mult_matrix = np.linalg.inv(L.T) # ^0.5 ???
		E = np.linalg.inv(cholesky(R, True).T)
		return E

	def get_pre_mult_matrix_observation(self):
		L = linalg.cholesky(self._Q, lower=True)
		pre_mult_matrix = np.linalg.inv(L.T) # ^0.5 ???
		return pre_mult_matrix

	def get_g_prime_wrt_motion(self, state, motion):
		"""
		:param state: The current state mean of the robot (format: np.array([x, y, theta])).
		:param motion: The motion command at the current time step (format: np.array([drot1, dtran, drot2])).
		:return: Jacobian of the state transition matrix w.r.t. the motion command.
		"""

		drot1, dtran, drot2 = motion

		return np.array([[-dtran * np.sin(state[2] + drot1), np.cos(state[2] + drot1), 0],
						 [dtran * np.cos(state[2] + drot1), np.sin(state[2] + drot1), 0],
						 [1, 0, 1]])

	def get_g_prime_wrt_state(self, state, motion):
		"""
		:param state: The current state mean of the robot (format: np.array([x, y, theta])).
		:param motion: The motion command at the current time step (format: np.array([drot1, dtran, drot2])).
		:return: Jacobian of the state transition matrix w.r.t. the state.
		"""

		drot1, dtran, drot2 = motion

		return np.array([[1, 0, -dtran * np.sin(state[2] + drot1)],
						 [0, 1, dtran * np.cos(state[2] + drot1)],
						 [0, 0, 1]])