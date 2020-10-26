#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import cvxpy as cp
from numba import njit, jitclass, jit
import numba

class iLQRWrapper(object):
    """This is a wrapper class for the iLQR iteraton
    """
    def __init__(self,  dynamic_model_function, 
                        objective_function, 
                        additional_variables_for_dynamic_model=None, 
                        additional_variables_for_objective_function=None):
        """ Initialization the iLQR solver class

            Parameter
            -----------
            dynamic_model_function : DynamicModelWrapper
                The dynamic model of the system
            objective_function : ObjectiveFunctionWrapper
                The objective function of the iLQR
            additional_variables_for_dynamic_model : array(T_int, q_int)
            additional_variables_for_objective_function : array(T_int, p_int)
                The additional arguments for the lamdify function
                q_int, p_int are the number of additional parameters
        """
        # Initialize the functions
        self.dynamic_model_function = dynamic_model_function
        self.objective_function = objective_function
        
        # Parameters for the model
        self.n_int=self.dynamic_model_function.n_int
        self.m_int=self.dynamic_model_function.m_int
        self.T_int=self.dynamic_model_function.T_int
        
        # Additional parameters for iLQR
        self.additional_variables_for_dynamic_model = additional_variables_for_dynamic_model
        self.additional_variables_for_objective_function = additional_variables_for_objective_function

        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self.trajectory_list = self.dynamic_model_function.evaluate_trajectory(additional_variables_all = self.additional_variables_for_dynamic_model)
        self.F_matrix_list = self.dynamic_model_function.evaluate_gradient_dynamic_model_function(self.trajectory_list, self.additional_variables_for_dynamic_model)
        self.objective_function_value_last = self.objective_function.evaluate_objective_function(self.trajectory_list, self.additional_variables_for_objective_function)
        self.c_vector_list = self.objective_function.evaluate_gradient_objective_function(self.trajectory_list, self.additional_variables_for_objective_function)
        self.C_matrix_list = self.objective_function.evaluate_hessian_objective_function(self.trajectory_list, self.additional_variables_for_objective_function)
        
    def vanilla_line_search(self,  gamma):
        """To ensure the value of the objective function is reduced monotonically

            Parameters
            ----------
            gamma : double 
                Gamma is the parameter for the line search : alpha=gamma*alpha

            Return
            ----------
            current_trajectory_list : array(T_int, m_int+n_int, 1)
                The current_iteration_trajectory after line search.
            current_objective_function_value : double
                The value of the objective function after the line search
        """
        # alpha: Step size
        alpha = 1.
        current_trajectory_list = np.zeros((self.T_int, self.n_int+self.m_int, 1))
        while(True): # Line Search if the z value is greater than zero
            current_trajectory_list = self.dynamic_model_function.update_trajectory(self.trajectory_list, self.K_matrix_list, self.k_vector_list, alpha, self.additional_variables_for_dynamic_model)
            current_objective_function_value = self.objective_function.evaluate_objective_function(current_trajectory_list, self.additional_variables_for_objective_function)
            delta_objective_function_value = current_objective_function_value-self.objective_function_value_last
            alpha = alpha * gamma
            if delta_objective_function_value<0:
                break
        return current_trajectory_list, current_objective_function_value
        
    def feasibility_line_search(self, gamma):
        """To ensure the value of the objective function is reduced monotonically, and ensure the trajectory for the next iteration is feasible.

            Parameters
            ----------
            gamma : double 
                Gamma is the parameter for the line search : alpha=gamma*alpha

            Return
            ----------
            current_trajectory_list : float64[T_int,m_int+n_int,1]
                The current_iteration_trajectory after line search.
            current_objective_function_value : float64
                The value of the objective function after the line search
        """
        alpha = 1.
        current_trajectory_list = np.zeros((self.T_int, self.n_int+self.m_int, 1))
        while(True): # Line Search if the z value is greater than zero
            current_trajectory_list = self.dynamic_model_function.update_trajectory(self.trajectory_list, self.K_matrix_list, self.k_vector_list, alpha, self.additional_variables_for_dynamic_model)
            current_objective_function_value = self.objective_function.evaluate_objective_function(current_trajectory_list, self.additional_variables_for_objective_function)
            delta_objective_function_value = current_objective_function_value-self.objective_function_value_last
            alpha = alpha * gamma
            if delta_objective_function_value<0 and (not np.isnan(delta_objective_function_value)):
                break
        return current_trajectory_list, current_objective_function_value

    def vanilla_stopping_criterion(self,    delta_objective_function_value, stopping_criterion):
        """Check the amount of change of the objective function. If the amount of change is less than the specific value, the stopping criterion is satisfied.

            Parameters
            ----------
            delta_objective_function_value : double
                The delta_objective_function_value in the current iteration.

            stopping_criterion : double 
                The number of input variables

            Return
            ----------
            isStop: Boolean
                Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        """
        delta_objective_function_value = delta_objective_function_value - self.objective_function_value_last
        if (abs(delta_objective_function_value) < stopping_criterion):
            return True
        return False
        
    def forward_pass(self, gamma = 0.5, stopping_criterion = 1e-6, line_search_method = "vanilla", stopping_criterion_method = "vanilla"):
        """Forward_pass in the iLQR algorithm with simple line search
        
            Parameters
            ----------
            gamma : double 
                Gamma is the parameter for the line search: alpha=gamma*alpha
            stopping_criterion : double 
                The number of input variables
            line_search_method : string
                Line search method
            stopping_criterion_method : string
                stopping criterion

            Return
            ----------
            objective_function_value: double
                The value of the objective function after the line search
            is_stop: Boolean
                Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        """
        # Do line search
        if line_search_method == "vanilla":
            self.trajectory_list, objective_function_value = self.vanilla_line_search(gamma)
        elif line_search_method == "feasibility":
            self.trajectory_list, objective_function_value = self.feasibility_line_search(gamma)

        # Do forward pass
        self.c_vector_list = self.objective_function.evaluate_gradient_objective_function(self.trajectory_list, self.additional_variables_for_objective_function)
        self.C_matrix_list = self.objective_function.evaluate_hessian_objective_function(self.trajectory_list, self.additional_variables_for_objective_function)
        self.F_matrix_list = self.dynamic_model_function.evaluate_gradient_dynamic_model_function(self.trajectory_list, self.additional_variables_for_dynamic_model)

        # Check the stopping criterion
        if stopping_criterion_method == "vanilla":
            is_stop = self.vanilla_stopping_criterion(objective_function_value, stopping_criterion)

        # Finally update the objective_function_value_last
        self.objective_function_value_last = objective_function_value
        return objective_function_value, is_stop

    def backward_pass(self):
        """Backward_pass in the iLQR algorithm
        """
        self.K_matrix_list, self.k_vector_list = self.backward_pass_static(self.m_int, self.n_int, self.T_int, self.C_matrix_list, self.c_vector_list, self.F_matrix_list)
    
    @staticmethod
    @njit
    def backward_pass_static(m_int, n_int, T_int, C_matrix_list, c_vector_list, F_matrix_list):
        V_matrix = np.zeros((n_int,n_int))
        v_vector = np.zeros((n_int,1))
        K_matrix_list = np.zeros((T_int, m_int, n_int))
        k_vector_list = np.zeros((T_int, m_int, 1))
        for i in range(T_int-1,-1,-1):
            Q_matrix = C_matrix_list[i] + F_matrix_list[i].T@V_matrix@F_matrix_list[i]
            q_vector = c_vector_list[i] + F_matrix_list[i].T@v_vector
            Q_uu = Q_matrix[n_int:n_int+m_int,n_int:n_int+m_int].copy()
            Q_ux = Q_matrix[n_int:n_int+m_int,0:n_int].copy()
            q_u = q_vector[n_int:n_int+m_int].copy()

            K_matrix_list[i] = -np.linalg.solve(Q_uu,Q_ux)
            k_vector_list[i] = -np.linalg.solve(Q_uu,q_u)
            V_matrix = Q_matrix[0:n_int,0:n_int]+\
                            Q_ux.T@K_matrix_list[i]+\
                            K_matrix_list[i].T@Q_ux+\
                            K_matrix_list[i].T@Q_uu@K_matrix_list[i]
            v_vector = q_vector[0:n_int]+\
                            Q_ux.T@k_vector_list[i] +\
                            K_matrix_list[i].T@q_u +\
                            K_matrix_list[i].T@Q_uu@k_vector_list[i]
        return K_matrix_list, k_vector_list

#############################
######## Example ############
######## Log Barrier ########
#############################

class iLQR_log_barrier_class(iLQRWrapper):
    def clear_objective_function_value_last(self):
        self.objective_function_value_last = np.inf

    def update_additional_variables_for_objective_function(self, additional_variables_for_objective_function):
        self.additional_variables_for_objective_function = additional_variables_for_objective_function

#############################
######## Example ############
########## ADMM #############
#############################
###### Not done yet #########
#############################
class ADMM_iLQR_class(iLQRWrapper):
    def __init__(self, x_u, dynamic_model, objective_function, n_int, m_int, T_int, initial_states, initial_input, initial_t):
        """Initialization of the class 
        
            Parameters
            ----------
            x_u : sympy.symbols 
                Vector including system states and input. e.g. x_u = sp.symbols('x_u:6')
            dynamic_model : dynamic_model_wrapper 
                The dynamic model of the system
            objective_function : objective_function_wrapper
                The objective function (may include the log barrier term)
            n_int : int 
                The number of state variables
            m_int : int 
                The number of input variables
            T_int : int 
                The prediction horizon
            initial_states : array(n_int, 1) 
                The initial state vector
            initial_input : array(T_int, m_int, 1) 
                The initial input vector
            initial_t : array(1) 
                The initial parameter t for the log barrier method
        """
        self.x_u_sp_var = x_u
        (self.dynamic_model_lamdify, 
        self.gradient_dynamic_model_lamdify) = dynamic_model.return_dynamic_model_and_gradient(x_u)
        (self.objective_function_lamdify, 
        self.gradient_objective_function_lamdify, 
        self.hessian_objective_function_lamdify) = objective_function.return_objective_function_gradient_and_hessian(x_u)
        
        self.iLQR_iteration = Create_iLQR_iteration_class(  self.dynamic_model_lamdify, 
                                                            self.gradient_dynamic_model_lamdify,
                                                            self.objective_function_lamdify,
                                                            self.gradient_objective_function_lamdify,
                                                            self.hessian_objective_function_lamdify,
                                                            n_int, m_int, T_int, initial_states, initial_input,
                                                            additional_parameters_for_dynamic_model=None, 
                                                            additional_parameters_for_objective_function=[0.5])
    def forward_pass(self, additional_parameters_for_objective_function, gamma_float64 = 0.5, stopping_criterion_float64 = 1e-6):
        """ Forward_pass in the iLQR algorithm with simple line search
        
            Parameters
            ----------
            gamma_float64 : float64 
                Gamma is the parameter for the line search: alpha=gamma*alpha
            stopping_criterion : float64 
                The number of input variables

            Return
            ----------
            stopping_criterion_float64: float64
                The value of the objective function after the line search
            isStop: Boolean
                Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        """
        return self.iLQR_iteration.forward_pass_insider(gamma_float64, stopping_criterion_float64, None, additional_parameters_for_objective_function, "feasibility", "vanilla")
#%%iLQR