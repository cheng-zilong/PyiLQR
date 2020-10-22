import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import cvxpy as cp
from numba import njit, jitclass, jit
import numba

class dynamic_model_wrapper(object):
    """ This is a wrapper class for the dynamic model
    """
    def __init__(self, dynamic_model_function, x_u_var, initial_states, additional_var = None):
        """ Initialization
            
            Parameters
            ---------------
            dynamic_model_function : sympy.array with symbols

            x_u_var : tuple with sympy.symbols 
                State and input variables in the model
            additional_var : tuple with sympy.symbols 
                For the use of designing new algorithms
            initial_states : array(n_int, 1)
                The initial state vector of the system
        """
        self.initial_states = initial_states
        self.n_int = int(initial_states.shape[0])
        self.m_int = int(len(x_u_var) - self.n_int)
        if additional_var is None:
            additional_var = sp.symbols("no_use")
        self.dynamic_model_lamdify = njit(sp.lambdify([x_u_var, additional_var], dynamic_model_function, "math"))
        gradient_dynamic_model_array = sp.transpose(sp.derive_by_array(dynamic_model_function, x_u_var))
        self.gradient_dynamic_model_lamdify = njit(sp.lambdify([x_u_var, additional_var], gradient_dynamic_model_array, "math"))

    def evaluate_trajectory(self, input_vector_all, additional_variables_all):
        """Evaluate the system trajectory by given initial states and input vector

            Parameters
            -----------------
            input_vector_all : array(T_int, n_int, 1)
            additional_variables_all : array(T_int, p_int)
                For the purpose of new method design

            Return
            ---------------
            trajectory : array(T_int, m_int+n_int, 1)
                The whole trajectory
        """
        return self._evaluate_trajectory_static(self.dynamic_model_lamdify, self.initial_states, self.m_int, self.n_int, input_vector_all, additional_variables_all)

    def update_trajectory(self, old_trajectory_list, K_matrix_all, k_vector_all, alpha, additional_variables_all): 
        """Update the trajectory by using iLQR

            Parameters
            -----------------
            old_trajectory_list : array(T_int, m_int+n_int, 1)
                The trajectory in the last iteration
            K_matrix_all : array(T_int, m_int, n_int)
            k_vector_all : array(T_int, m_int, 1)
            alpha : double
                Step size in this iteration
            additional_variables_all : array(T_int, p_int)
                For the purpose of new method design

            Return
            ---------------
            new_trajectory_list : array(T_int, m_int+n_int, 1) 
                The updated trajectory
        """
        return self._update_trajectory_static(self.dynamic_model_lamdify, self.m_int, self.n_int, old_trajectory_list, K_matrix_all, k_vector_all, alpha, additional_variables_all)

    def evaluate_gradient_dynamic_model_function(self, trajectory_list, additional_variables_all):
        """Return the matrix of the gradient of the dynamic_model

            Parameters
            -----------------
            trajectory_list : array(T_int, m_int+n_int, 1) 
            additional_variables_all : array(T_int, p_int)
                For the purpose of new method design

            Return
            ---------------
            grad : array(T_int, m_int, n_int)
                The gradient of the dynamic_model
        """
        return self._evaluate_gradient_dynamic_model_function_static(self.gradient_dynamic_model_lamdify, trajectory_list, additional_variables_all)
    
    @staticmethod
    @njit
    def _evaluate_trajectory_static(dynamic_model_lamdify, initial_states, m_int, n_int, input_vector_all, additional_variables_all):

        T_int = int(input_vector_all.shape[0])
        trajectory_list = np.zeros((T_int, m_int+n_int, 1))
        trajectory_list[0] = np.vstack((initial_states, input_vector_all[0]))
        for tau in range(T_int-1):
            trajectory_list[tau+1, :n_int, 0] = np.asarray(dynamic_model_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]),dtype=np.float64)
            trajectory_list[tau+1, n_int:] = input_vector_all[tau+1]
        return trajectory_list

    @staticmethod
    @njit
    def _update_trajectory_static(dynamic_model_lamdify, m_int, n_int, old_trajectory_list, K_matrix_all, k_vector_all, alpha, additional_variables_all):
        T_int = int(K_matrix_all.shape[0])
        new_trajectory_list = np.zeros((T_int, m_int+n_int, 1))
        new_trajectory_list[0] = old_trajectory_list[0] # initial states are the same
        for tau in range(T_int-1):
            # The amount of change of state x
            delta_x = new_trajectory_list[tau, 0:n_int] - old_trajectory_list[tau, 0:n_int]
            # The amount of change of input u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real input of next iteration
            input_u = old_trajectory_list[tau, n_int:n_int+m_int] + delta_u
            new_trajectory_list[tau,n_int:] = input_u
            new_trajectory_list[tau+1,0:n_int] = np.asarray(dynamic_model_lamdify(new_trajectory_list[tau,:,0], additional_variables_all[tau]),dtype=np.float64).reshape(-1,1)
            # dont care the input at the last time stamp, because it is always zero
        return new_trajectory_list

    @staticmethod
    @njit
    def _evaluate_gradient_dynamic_model_function_static(gradient_dynamic_model_lamdify, trajectory_list, additional_variables_all):
        T_int = int(trajectory_list.shape[0])
        F_matrix_initial =  gradient_dynamic_model_lamdify(trajectory_list[0,:,0], additional_variables_all[0])
        F_matrix_list = np.zeros((T_int, len(F_matrix_initial), len(F_matrix_initial[0])))
        F_matrix_list[0] = np.asarray(F_matrix_initial, dtype = np.float64)
        for tau in range(1, T_int):
            F_matrix_list[tau] = np.asarray(gradient_dynamic_model_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64)
        return F_matrix_list