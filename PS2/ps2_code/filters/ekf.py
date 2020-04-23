"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle

from tools.task import get_jacobian_G
from tools.task import get_jacobian_V
from tools.task import get_jacobian_H
from tools.objects import Gaussian

class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = self.mu[np.newaxis].T
        self._state_bar.Sigma = self.Sigma

        # print('\nCovariance before\n',self.Sigma)
        # print('State before\n',self.mu)
        # print('action:\n',u)

        # INPUTS (previous belief):
        # previous state X_{t-1}:   x, y, theta //our best estimate so far, mean
        # previous covariance Sigma_{t-1}: 3*3 matrix:
								# 				       [sigma_x_x     sigma_x_y     sigma_x_theta
								# 				        sigma_y_x     sigma_y_y     sigma_y_theta
								# 				        sigma_theta_x sigma_theta_y sigma_theta_theta ]
        # Currect actions/motions u:         drot1, dtran, drot2
        # TASK: Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.

        # g(u_t,x{t-1}) = A*x_{t-1} + B*u + R # probablu u with noise, R is not present in this project/tast
        # x_t = g(u_t,x{t-1}) + N(0,R) (decomposed into two parts: transition function without noise and mapped (from control to the state space) noise)
        
        # Jacobian for linear transformation g(u_t,x{t-1}) = g(u_t, mu_{t-1}) + G * (x_{t-1} - m_{t-1})
        G_t = get_jacobian_G(self.mu, u) # derivative of transition function wrt to X_{t-1}  evaluated at u_t and mu_{t-1}

        # Propagate control trough state pose_mean = g(pose_mean, u)
        # Predicted belief (belief_bar) without incorporation the measurements z
        # the mean is updated using the deterministic version of the state transition function
        mu_bar = get_prediction(self.mu, u)

        # Covariance for noise in action space
        M_t = get_motion_noise_covariance(u, self._alphas) # has to be mapped into the state space
        # Propagate (transform) total covariance
        # The transformation from control space to the state space is performed by another linear approximation
        # jacobian V_t  is needed for this operation - the derivative of the transformation function g wrt motion parameters u, evaluated at u_t and mu_{t-1}
        V_t = get_jacobian_V(self.mu, u)
        Sigma_bar = G_t.dot(self.Sigma.dot(G_t.T)) + V_t.dot(M_t.dot(V_t.T)) # final covariance for the propagation step

        self._state_bar = Gaussian(mu_bar, Sigma_bar)

        # for now we have bel_bar

        # print('Covariance after\n',self.Sigma)
        # print('State after\n',self.mu)


    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma


        # observation model:
        # Z = [ atan(m_y-mu_y ; m_x-mu_x) - mu_theta ; signature(id) ] # I did not include the signatere for now



        # Take the state and covariance from the prediction stage (with bar)
        mu_bar = self._state_bar.mu
        Sigma_bar = self._state_bar.Sigma

        # print('\n\nState before\n',mu_bar)

        # FOR EACH OBSERVATION Z_i IN LANDMARK SET DO (recursive approach (superposition also can be aplied, when we just sum the observation contributions at once)):
        # we also need to linearize the observation model
        # jacobian of observation function wrt to state, evaluated at mu_t (obtained from the prediction step)
        H_t = get_jacobian_H(np.squeeze(mu_bar), z)
        S_t = H_t.dot(Sigma_bar.dot(H_t.T)) + self._Q
        # Kalman gain - specifies the degree with which the measurement is incorporated into the new state estimate
        K_t = Sigma_bar.dot(H_t.T).dot(np.linalg.inv(S_t))
        z_expected = get_expected_observation(np.squeeze(mu_bar), z[1]) # expected from observation model based on the estimated state
        
        # Innovation - is the difference betw actual measurement and the expected measurment (expected is estimated via observation model)
        innovation = np.array([[z[0] - z_expected[0]]])
        # innovation = wrap_angle(innovation)
        # innovation = (z[0] - z_expected[0] + np.pi) % (2 * np.pi) - np.pi

        mu_bar = (mu_bar + K_t.dot( innovation ))
        Sigma_bar = (np.eye(3) - K_t.dot(H_t)).dot(Sigma_bar)
        # END for loop
        self._state = Gaussian(mu_bar, Sigma_bar)

        # for now we have bel