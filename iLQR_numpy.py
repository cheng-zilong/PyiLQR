#%%
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

class objective_function_wrapper(object):
    """This is a wrapper class for the objective function
    """
    def __init__(self, objective_function, x_u_var, additional_var = None):
        """ Initialization

            Parameters
            -----------------
            objective_function : function(x_u, (additional_variables))
                The function of the objetive function
                The methods in this class will only be related to the first argument, 
                The second optional argument is for the purpose of new methods design
        """
        if additional_var is None:
            additional_var = sp.symbols("no_use")

        self.objective_function_lamdify = njit(sp.lambdify([x_u_var,additional_var], objective_function, "numpy"))
        gradient_objective_function_array = sp.derive_by_array(objective_function, x_u_var)
        self.gradient_objective_function_lamdify = njit(sp.lambdify([x_u_var, additional_var], gradient_objective_function_array,"numpy"))       
        hessian_objective_function_array = sp.derive_by_array(gradient_objective_function_array, x_u_var)
        # A stupid method to ensure each element in the hessian matrix is in the type of float64
        self.hessian_objective_function_lamdify = njit(sp.lambdify([x_u_var, additional_var], np.asarray(hessian_objective_function_array)+1e-100*np.eye(hessian_objective_function_array.shape[0]),"numpy"))

    def evaluate_objective_function(self, trajectory_list, additional_variables_all):
        """Return the objective function value

            Parameters
            -----------------
            trajectory_list : array(T_int, m_int+n_int, 1) 
            additional_variables : tensor
                For the purpose of new method design

            Return
            ---------------
            obj : scalar
                The objective function value
        """
        return self._evaluate_objective_function_static(self.objective_function_lamdify, trajectory_list, additional_variables_all)

    def evaluate_gradient_objective_function(self, trajectory_list, additional_variables_all):
        """Return the objective function value

            Parameters
            -----------------
            trajectory_list : array(T_int, m_int+n_int, 1) 
            additional_variables : tensor
                For the purpose of new method design

            Return
            ---------------
            grad : array[T_int, m_int+n_int,1] 
                The objective function jacobian
        """
        return self._evaluate_gradient_objective_function_static(self.gradient_objective_function_lamdify, trajectory_list, additional_variables_all)

    def evaluate_hessian_objective_function(self, trajectory_list, additional_variables_all):
        """Return the objective function value

            Parameters
            -----------------
            trajectory_list : array(T_int, m_int+n_int, 1) 
            additional_variables : tensor
                For the purpose of new method design

            Return
            ---------------
            grad : array[T_int, m_int+n_int, m_int+n_int] 
                The objective function hessian
        """
        return self._evaluate_hessian_objective_function_static(self.hessian_objective_function_lamdify, trajectory_list, additional_variables_all)

    @staticmethod
    @njit
    def _evaluate_objective_function_static(objective_function_lamdify, trajectory_list, additional_variables_all):
        T_int = int(trajectory_list.shape[0])
        obj_value = 0
        for tau in range(T_int):
            obj_value = obj_value + np.asarray(objective_function_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64)
        return obj_value
        
    @staticmethod
    @njit
    def _evaluate_gradient_objective_function_static(gradient_objective_function_lamdify, trajectory_list, additional_variables_all):
        T_int = int(trajectory_list.shape[0])
        m_n_int = int(trajectory_list.shape[1])
        grad_all_tau = np.zeros((T_int, m_n_int, 1))
        for tau in range(T_int):
            grad_all_tau[tau] = np.asarray(gradient_objective_function_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64).reshape(-1,1)
        return grad_all_tau
        
    @staticmethod
    @njit
    def _evaluate_hessian_objective_function_static(hessian_objective_function_lamdify, trajectory_list, additional_variables_all):
        T_int = int(trajectory_list.shape[0])
        m_n_int = int(trajectory_list.shape[1])
        hessian_all_tau = np.zeros((T_int, m_n_int, m_n_int))
        for tau in range(T_int):
            hessian_all_tau[tau] = np.asarray(hessian_objective_function_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64)
        return hessian_all_tau

class iLQR_wrapper(object):
    """This is a wrapper class for the iLQR iteraton
    """
    def __init__(self,  dynamic_model_function, 
                        objective_function, T_int, 
                        initial_input_vector=None, 
                        additional_variables_for_dynamic_model=None, 
                        additional_variables_for_objective_function=None):

        # Initialize the functions
        self.dynamic_model_function = dynamic_model_function
        self.objective_function = objective_function
        
        # Parameters for the model
        self.n_int=self.dynamic_model_function.n_int
        self.m_int=self.dynamic_model_function.m_int
        self.T_int=T_int
        
        # Dynamic matrix taylor expansion at a point
        self.F_matrix_list = np.zeros((T_int, self.n_int, self.n_int+self.m_int))
        self.trajectory_list = np.zeros((T_int, self.n_int+self.m_int, 1))
        
        # parameters for iLQR
        self.K_matrix_list = np.zeros((T_int, self.m_int, self.n_int))
        self.k_vector_list = np.zeros((T_int, self.m_int, 1))

        self.C_matrix_list = np.zeros((T_int, self.m_int+self.n_int, self.m_int+self.n_int))
        self.c_vector_list = np.zeros((T_int, self.m_int+self.n_int, 1))
        
        if additional_variables_for_dynamic_model is None:
            self.additional_variables_for_dynamic_model = np.zeros(T_int)
        else:
            self.additional_variables_for_dynamic_model = additional_variables_for_dynamic_model
        if additional_variables_for_objective_function is None:
            self.additional_variables_for_objective_function = np.zeros(T_int)
        else:
            self.additional_variables_for_objective_function = additional_variables_for_objective_function
        if initial_input_vector is None:
            initial_input_vector = np.zeros((T_int,self.m_int,1))
        # Initialize the trajectory, F_matrix, C_matrix and c_vector
        self.trajectory_list = self.dynamic_model_function.evaluate_trajectory(initial_input_vector, self.additional_variables_for_dynamic_model)
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
            delta_objective_function_value : float64
                The delta_objective_function_value in the current iteration.

            stopping_criterion : float64 
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

class objective_function_log_barrier_class(objective_function_wrapper):
    def __init__(self, objective_function, x_u_var, inequality_constraints_list, variables_in_inequality_constraints_list = None):
        t_var = sp.symbols('t') # introduce the parameter for log barrier
        additional_variables_list = [] 
        additional_variables_list.append(t_var)
        # add the variables in the constraint to the additional_variables_list
        if variables_in_inequality_constraints_list != None:
            additional_variables_list = additional_variables_list + variables_in_inequality_constraints_list
        # construct the barrier objective function
        barrier_objective_function = objective_function
        # add the inequality constraints to the cost function
        for inequality_constraint in inequality_constraints_list:
            barrier_objective_function = barrier_objective_function + (-1/t_var)*sp.log(-inequality_constraint)
        super().__init__(barrier_objective_function, x_u_var, additional_variables_list)

class iLQR_log_barrier_class(iLQR_wrapper):
    def __init__(self,  dynamic_model_function, 
                        objective_function, T_int, 
                        initial_input_vector=None, 
                        additional_variables_for_dynamic_model=None, 
                        additional_variables_for_objective_function=None):
        """Initialization of the class 
        
            Parameters
            ----------
            dynamic_model_function : dynamic_model_wrapper 
                The dynamic model of the system
            objective_function : objective_function_wrapper
                The objective function (may include the log barrier term)
            initial_input_vector : array(T_int, m_int, 1) 
                The initial input vector
            additional_variables_for_dynamic_model : list(T_int, q_int)
            additional_variables_for_objective_function : list(T_int, q_int)
                The additional arguments for the lamdify function
                q_int is the number of additional parameters
        """
        super().__init__(dynamic_model_function, objective_function, T_int, initial_input_vector, additional_variables_for_dynamic_model, additional_variables_for_objective_function)

    def clear_objective_function_value_last(self):
        self.objective_function_value_last = np.inf

    def forward_pass(self, gamma = 0.5, stopping_criterion = 1e-6, line_search_method = "feasibility", stopping_criterion_method = "vanilla"):
        return super().forward_pass(gamma, stopping_criterion, line_search_method, stopping_criterion_method)

    def update_additional_variables_for_objective_function(self, additional_variables_for_objective_function):
        self.additional_variables_for_objective_function = additional_variables_for_objective_function

#############################
######## Example ############
########## ADMM #############
#############################

class ADMM_iLQR_class(iLQR_wrapper):
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