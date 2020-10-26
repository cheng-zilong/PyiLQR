import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import cvxpy as cp
from numba import njit, jitclass, jit
import numba

class ObjectiveFunctionWrapper(object):
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

    def evaluate_objective_function(self, trajectory_list, additional_variables_all = None):
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

    def evaluate_gradient_objective_function(self, trajectory_list, additional_variables_all = None):
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

    def evaluate_hessian_objective_function(self, trajectory_list, additional_variables_all = None):
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
        if additional_variables_all == None:
            additional_variables_all = np.zeros((T_int,1))
        obj_value = 0
        for tau in range(T_int):
            obj_value = obj_value + np.asarray(objective_function_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64)
        return obj_value
        
    @staticmethod
    @njit
    def _evaluate_gradient_objective_function_static(gradient_objective_function_lamdify, trajectory_list, additional_variables_all):
        T_int = int(trajectory_list.shape[0])
        m_n_int = int(trajectory_list.shape[1])
        if additional_variables_all == None:
            additional_variables_all = np.zeros((T_int,1))
        grad_all_tau = np.zeros((T_int, m_n_int, 1))
        for tau in range(T_int):
            grad_all_tau[tau] = np.asarray(gradient_objective_function_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64).reshape(-1,1)
        return grad_all_tau
        
    @staticmethod
    @njit
    def _evaluate_hessian_objective_function_static(hessian_objective_function_lamdify, trajectory_list, additional_variables_all):
        T_int = int(trajectory_list.shape[0])
        m_n_int = int(trajectory_list.shape[1])
        if additional_variables_all == None:
            additional_variables_all = np.zeros((T_int,1))
        hessian_all_tau = np.zeros((T_int, m_n_int, m_n_int))
        for tau in range(T_int):
            hessian_all_tau[tau] = np.asarray(hessian_objective_function_lamdify(trajectory_list[tau,:,0], additional_variables_all[tau]), dtype = np.float64)
        return hessian_all_tau

class objective_function_log_barrier_class(ObjectiveFunctionWrapper):
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