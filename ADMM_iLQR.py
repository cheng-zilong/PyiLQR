#%%
import control as ct
import control.matlab as mt
import numpy as np
import sympy as sp
import scipy as sci
# import casadi
import time as tm
from scipy import io
import cvxpy as cp

from numba import jit

@jit(nopython=True)  
def backward_pass(C_matrix_cost_and_iLQR, c_vector_iLQR, F_matrix):
    V_matrix = np.zeros((n_int,n_int))
    v_vector = np.zeros((n_int,1))
    K_matrix_list = np.zeros((T_int, m_int, n_int))
    k_vector_list = np.zeros((T_int, m_int, 1))

    for i in range(T_int-1,-1,-1):
        Q_matrix = C_matrix_cost_and_iLQR + F_matrix[i].T@V_matrix@F_matrix[i]
        q_vector = c_vector_iLQR[i] + F_matrix[i].T@v_vector
        K_matrix_list[i] = -np.linalg.inv(Q_matrix[n_int:n_int+m_int,n_int:n_int+m_int])@Q_matrix[n_int:n_int+m_int,0:n_int]
        k_vector_list[i] = -np.linalg.inv(Q_matrix[n_int:n_int+m_int,n_int:n_int+m_int])@q_vector[n_int:n_int+m_int,:]
        V_matrix = Q_matrix[0:n_int,0:n_int]+\
                        Q_matrix[0:n_int,n_int:n_int+m_int]@K_matrix_list[i,:,:]+\
                        K_matrix_list[i,:,:].T@Q_matrix[n_int:n_int+m_int,0:n_int]+\
                        K_matrix_list[i,:,:].T@Q_matrix[n_int:n_int+m_int,n_int:n_int+m_int]@K_matrix_list[i,:,:]

        v_vector = q_vector[0:n_int]+\
                        Q_matrix[0:n_int,n_int:n_int+m_int]@k_vector_list[i,:,:] +\
                        K_matrix_list[i,:,:].T@q_vector[n_int:n_int+m_int] +\
                        K_matrix_list[i,:,:].T@Q_matrix[n_int:n_int+m_int,n_int:n_int+m_int]@k_vector_list[i,:,:]
    return K_matrix_list, k_vector_list


# Derive the Jacobian matrix of the system
# self.r_x_symbol, self.r_y_symbol, self.phi_symbol, self.v_symbol, self.omega_symbol, self.a_symbol = sp.symbols('r_x r_y phi v omega a')
x_u = sp.symbols('x_u:6')
d_constant_int = 3
h_constant_int = 0.1
h_d_constant_int = h_constant_int/d_constant_int
b_function = d_constant_int + h_constant_int*x_u[3]*sp.cos(x_u[4])-sp.sqrt(d_constant_int**2 - (h_constant_int**2)*(x_u[3]**2)*(sp.sin(x_u[4])**2))
dynamic_model_function_array = sp.Array([  
                                    [x_u[0] + b_function*sp.cos(x_u[2])], 
                                    [x_u[1] + b_function*sp.sin(x_u[2])], 
                                    [x_u[2] + sp.asin(h_d_constant_int*x_u[3]*sp.sin(x_u[4]))], 
                                    [x_u[3]+h_constant_int*x_u[5]]])
dynamic_model_function_lamdify = jit(sp.lambdify([x_u],dynamic_model_function_array,"math"),nopython=True)
gradient_dynamic_model_function_array = sp.transpose(sp.derive_by_array(dynamic_model_function_array, x_u)[:,:,0])
gradient_dynamic_model_function_lamdify = jit(sp.lambdify([x_u], gradient_dynamic_model_function_array,"math"),nopython=True)
@jit(nopython=True)  
def evaluate_dynamic_gradient(x_u):
        '''
        x_u is a 2 dims vector *fghfghf*\\
        x_u[0] is position r_x\\
        x_u[1] is position r_y\\
        x_u[2] is angle phi\\
        x_u[3] is velocity v\\
        x_u[4] is angle omega\\
        x_u[5] is acceleration a
        '''
        temp = gradient_dynamic_model_function_lamdify(x_u[:,0])
        return np.asarray(temp)

@jit(nopython=True)  
def evaluate_dynamic_next_state(x_u):
    '''
    x_u is a 2 dims vector\\
    x_u[0] is position r_x\\
    x_u[1] is position r_y\\
    x_u[2] is angle phi\\
    x_u[3] is velocity v\\
    x_u[4] is angle omega\\
    x_u[5] is acceleration a
    '''
    temp = dynamic_model_function_lamdify(x_u[:,0])
    return np.asarray(temp)

@jit(nopython=True)  
def forward_pass(initial_system_states,
                    K_matrix_list, k_vector_list, 
                    trajectory_vector,
                    C_matrix_cost_and_iLQR, c_vector_iLQR, c_vector_cost,
                    F_matrix):
    state_x = initial_system_states
    cost = np.asarray([[0]],dtype=np.float64)
    for i in range(T_int):
        input_u = K_matrix_list[i]@(state_x - trajectory_vector[i,0:n_int])+\
            k_vector_list[i]+\
            trajectory_vector[i, n_int:n_int+m_int]
        trajectory_vector[i] = np.vstack((state_x,input_u))
        c_vector_iLQR[i] = C_matrix_cost_and_iLQR@trajectory_vector[i]+0.5*c_vector_cost[i]
        F_matrix[i] = evaluate_dynamic_gradient(trajectory_vector[i])
        state_x = evaluate_dynamic_next_state(trajectory_vector[i])
        # calculate the cost 
        cost = cost + trajectory_vector[i].T@C_matrix_cost_and_iLQR@trajectory_vector[i] + c_vector_cost[i].T@trajectory_vector[i]
    return cost, c_vector_iLQR, trajectory_vector, F_matrix

class vehicle_class:
    def __init__(self, n_int, m_int, T_int, p_int, sigma_int):
        # Parameters for the model
        self.n_int=n_int
        self.m_int=m_int
        self.T_int=T_int
        self.p_int = p_int
        # Parameters for optimizaiton
        self.sigma_int = sigma_int


        self.F_matrix = np.zeros([T_int, self.n_int, self.n_int+self.m_int])
        self.trajectory_vector = np.zeros([self.T_int, self.n_int+self.m_int, 1])

        # parameters for iLQR
        self.weighting_parameter_q1 = 1
        self.weighting_parameter_q2 = 1
        self.weighting_parameter_r1 = 10
        self.weighting_parameter_r2 = 1
        self.reference_velocity_v = 8

        self.c_vector_iLQR = np.zeros([T_int, self.n_int+self.m_int, 1])
        self.c_vector_cost = np.zeros([T_int, self.n_int+self.m_int, 1])
        ## C_matrix is the same both in the cost function and iLQR iteration
        self.C_matrix_cost_and_iLQR = np.asarray(np.diag([0,self.weighting_parameter_q1,0,self.weighting_parameter_q2,self.weighting_parameter_r1,self.weighting_parameter_r2]),dtype=np.float64 )   

        self.K_matrix_list = np.zeros([T_int, m_int, n_int])
        self.k_vector_list = np.zeros([T_int, m_int, 1])

        self.initial_system_states = np.asarray([0,0,0,4],dtype=np.float64).reshape(-1,1)

        self.hat_mathcal_A_matrix = np.asarray( [
                                                    [1,0,0,0,0,0],
                                                    [0,1,0,0,0,0],
                                                    [0,0,0,0,1,0],
                                                    [0,0,0,0,0,1]
                                                ])
        self.mathcal_A_matrix = np.kron(np.eye(T_int), self.hat_mathcal_A_matrix)

        self.input_upper_bound = np.asarray([[0.6],[3]])
        self.input_lower_bound = np.asarray([[-0.6],[-3]])

        self.current_cost_int = 1

        self.initialize_trajectory()

    def initialize_trajectory(self):
        current_state = self.initial_system_states.copy()
        # if (z_admm_variable is None):
        u_vector = np.zeros([2,1])
        # else:
        #     constant_beta = z_admm_variable - (1/self.sigma_int)*lambda_admm_variable
        for i in range(T_int):
            # if (z_admm_variable is not None):
            #     u_vector = constant_beta[self.p_int*i+2:self.p_int*i+self.p_int]
            x_u_vector = np.vstack([current_state, u_vector])
            self.trajectory_vector[i] = x_u_vector
            self.F_matrix[i] = evaluate_dynamic_gradient(x_u_vector)
            current_state = evaluate_dynamic_next_state(x_u_vector)
    


    
    def ADMM_first_step(self, z_admm_variable=None, lambda_admm_variable=None, max_iteration_iLQR = 100):
        '''
        return y_admm_variable
        '''
        # if this is the first step
        
        if (z_admm_variable is None):
            for i in range(self.T_int):
                self.c_vector_cost[i] = np.diag([0,0,0,-2*self.weighting_parameter_q2,0,0])@(np.asarray([0,0,0,self.reference_velocity_v,0,0]).reshape(-1,1))
                self.c_vector_iLQR[i] = self.C_matrix_cost_and_iLQR@self.trajectory_vector[i]+0.5*self.c_vector_cost[i]
        else:
            constant_beta = z_admm_variable - (1/self.sigma_int)*lambda_admm_variable
            self.C_matrix_cost_and_iLQR = np.diag([0,self.weighting_parameter_q1,0,self.weighting_parameter_q2,self.weighting_parameter_r1,self.weighting_parameter_r2]) \
                                +(self.sigma_int/2)*self.hat_mathcal_A_matrix.T@self.hat_mathcal_A_matrix
            for i in range(self.T_int):
                self.c_vector_cost[i] = -2*np.diag([0,0,0,self.weighting_parameter_q2,0,0])@(np.asarray([0,0,0,self.reference_velocity_v,0,0]).reshape(-1,1)) \
                    -self.sigma_int*self.hat_mathcal_A_matrix.T@constant_beta[self.p_int*i:self.p_int*i+self.p_int]
                self.c_vector_iLQR[i] = self.C_matrix_cost_and_iLQR@self.trajectory_vector[i]+0.5*self.c_vector_cost[i]
        
        for j in range(max_iteration_iLQR):
            time0 = tm.time()
            self.K_matrix_list, self.k_vector_list = backward_pass(self.C_matrix_cost_and_iLQR, self.c_vector_iLQR, self.F_matrix)
            last_cost_int = self.current_cost_int
            backward_time = tm.time()
            self.current_cost_int, self.c_vector_iLQR, self.trajectory_vector, self.F_matrix = forward_pass(self.initial_system_states,
                                                            self.K_matrix_list, self.k_vector_list, 
                                                            self.trajectory_vector,
                                                            self.C_matrix_cost_and_iLQR, self.c_vector_iLQR, self.c_vector_cost,
                                                            self.F_matrix)
            forward_time = tm.time()
            stopping_criterion = abs((self.current_cost_int-last_cost_int)/last_cost_int)
            print(  "###################################\n"+
                    "#########ADMM STEP ONE#############\n"+
                    "#########Iteration:%3d#############\n"%(j)+
                    "Backward Time:\t %.5f\n"%(backward_time - time0)+
                    "Forward Time:\t %.5f\n"%(forward_time - backward_time)+
                    "Stopping Criterion:\t%.2e\n"%(stopping_criterion)+
                    "Objective Value\t%.5f"%(self.current_cost_int))
            if stopping_criterion < 1e-4:
                print("ADMM Step One Complete!!!!!!\n")
                return self.trajectory_vector.reshape(-1,1), self.current_cost_int
        print("max iteration reached!!!\n")
        return self.trajectory_vector.reshape(-1,1), self.current_cost_int

    def ADMM_second_step(self, y_admm_variable, lambda_admm_variable):
        constant_rho = self.mathcal_A_matrix@y_admm_variable + (1/self.sigma_int)*lambda_admm_variable
        z_admm_variable = np.zeros([self.p_int*self.T_int,1])
        for i in range(self.T_int):
            # Collision Avoidance
            z_admm_variable[self.p_int*i:self.p_int*i+2] = self.projection_onto_ellipse(constant_rho[self.p_int*i:self.p_int*i+2])
            # System Input Constraints
            z_admm_variable[self.p_int*i+2:self.p_int*i+self.p_int] = np.maximum(
                    np.minimum(constant_rho[self.p_int*i+2:self.p_int*i+self.p_int], self.input_upper_bound), 
                    self.input_lower_bound)
        return z_admm_variable

    def projection_onto_box(self, position):
        # Define the box
        top = 1
        bottom = -1
        left = 10
        right = 15
        
        x = position[0,0]
        y = position[1,0]

        position__ = position.copy()
        # Judge inside
        if not((x<left) or (x>right) or (y>top) or (y<bottom)): # if inside
            # calulate the distance to top bottom left and right
            distance = np.asarray([abs(y-top), abs(y-bottom), abs(x-left), abs(x-right), ])
            index = np.argmin(distance)
            if index == 0: # projection to top
                position__[1] = top
            elif index == 1: # projection to bottom
                position__[1] = bottom
            elif index == 2: # projection to left
                position__[0] = left
            elif index == 3: # projection to right
                position__[0] = right
        return position__

    def projection_onto_ellipse(self, position):
        
        center_ellipse = np.asarray([[15],[-1]])
        a = 5
        b = 2.5
        position__ = position.copy() - center_ellipse

        # Judge inside
        if (position__[0,0]**2)/(a**2) + (position__[1,0]**2)/(b**2) <= 1: # if inside
            theta = np.arctan2(position__[1,0], position__[0,0])
            kappa = (a*b)/np.sqrt((b**2)*(np.cos(theta)**2)+(a**2)*(np.sin(theta)**2))
            position__[0,0] = kappa*np.cos(theta)
            position__[1,0] = kappa*np.sin(theta)
        return position__ + center_ellipse

#%%
if __name__=="__main__":
    n_int = 4
    m_int = 2
    T_int = 60
    p_int = 4
    sigma_int = 10
    vehicle = vehicle_class(n_int, m_int, T_int, p_int, sigma_int)
    Iteration_int = 20
    result = np.zeros([Iteration_int,T_int,n_int+m_int,1])
    admm_max_iteration = 20
    y_admm_variable = np.zeros([(n_int+m_int)*T_int,1])
    z_admm_variable = None
    lambda_admm_variable = np.zeros([p_int*T_int,1])

    result_solver_y = np.zeros([admm_max_iteration+1,(n_int+m_int)*T_int])
    result_solver_z = np.zeros([admm_max_iteration+1,p_int*T_int])
    result_cost = np.zeros([Iteration_int+1])

    y_admm_variable,current_cost_int = vehicle.ADMM_first_step(z_admm_variable,lambda_admm_variable)
    z_admm_variable = vehicle.ADMM_second_step(y_admm_variable,lambda_admm_variable)
    lambda_admm_variable = lambda_admm_variable + sigma_int*(vehicle.mathcal_A_matrix@y_admm_variable-z_admm_variable)
    result_solver_y[0,:] = y_admm_variable[:,0]
    result_solver_z[0,:] = z_admm_variable[:,0]
    result_cost[0] = current_cost_int
    time_start = tm.time()
    for j in range(admm_max_iteration):
        y_admm_variable,current_cost_int = vehicle.ADMM_first_step(z_admm_variable,lambda_admm_variable)
        z_admm_variable = vehicle.ADMM_second_step(y_admm_variable,lambda_admm_variable)
        lambda_admm_variable = lambda_admm_variable + sigma_int*(vehicle.mathcal_A_matrix@y_admm_variable-z_admm_variable)
        print(np.linalg.norm(vehicle.mathcal_A_matrix@y_admm_variable-z_admm_variable))
        result_solver_y[j+1,:] = y_admm_variable[:,0]
        result_solver_z[j+1,:] = z_admm_variable[:,0]
        result_cost[j+1] = current_cost_int
    all_time = tm.time()
    print("All time:%.5f"%(all_time - time_start))
    io.savemat("result_solver_vehicle_CiLQR.mat",{"result_solver_y": result_solver_y, "result_solver_z":result_solver_z, 'result_cost': result_cost})   
             
    


# %%
