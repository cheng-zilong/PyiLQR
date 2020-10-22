#%%
import numpy as np
import sympy as sp
import time as tm
from scipy import io
import importlib
import iLQR_numpy
importlib.reload(iLQR_numpy)

if __name__ == "__main__":
    #################################
    ##### Model of the vehicle ######
    #################################
    x_u = sp.symbols('x_u:6')
    d_constant_int = 3
    h_constant_int = 0.1
    h_d_constant_int = h_constant_int/d_constant_int
    b_function = d_constant_int \
                + h_constant_int*x_u[3]*sp.cos(x_u[4])\
                -sp.sqrt(d_constant_int**2 - (h_constant_int**2)*(x_u[3]**2)*(sp.sin(x_u[4])**2))
    initial_states = np.asarray([0,0,0,4],dtype=np.float64).reshape(-1,1)
    dynamic_model = iLQR_numpy.dynamic_model_wrapper  (    
                        sp.Array([  
                                    x_u[0] + b_function*sp.cos(x_u[2]), 
                                    x_u[1] + b_function*sp.sin(x_u[2]), 
                                    x_u[2] + sp.asin(h_d_constant_int*x_u[3]*sp.sin(x_u[4])), 
                                    x_u[3]+h_constant_int*x_u[5]
                                ]), x_u, initial_states)
    #################################
    ## Constraints of the vehicle ###
    #################################
    # box constraints
    inequality_constraint1 = x_u[5] - 8 # acceleration<=8
    inequality_constraint2 = -8 - x_u[5] # -3<=acceleration
    inequality_constraint3 = x_u[4] - 0.6 # omega<=0.6
    inequality_constraint4 = -0.6 - x_u[4] # -0.6<=omega
    # collision avoidance constraints
    obs1_x, obs1_y, obs2_x, obs2_y = sp.symbols('obs1_x, obs1_y, obs2_x, obs2_y')
    inequality_constraint5 = 1 - ((x_u[0] - obs1_x)**2)/25 - ((x_u[1] - obs1_y)**2)/4 
    inequality_constraint6 = 1 - ((x_u[0] - obs2_x)**2)/25 - ((x_u[1] - obs2_y)**2)/4 
    inequality_constraints_list = [ inequality_constraint1, 
                                    inequality_constraint2, 
                                    inequality_constraint3, 
                                    inequality_constraint4,
                                    inequality_constraint5,
                                    inequality_constraint6]

    #################################
    ###### Weighting Matrices #######
    #################################
    C_matrix = np.diag([0.,1.,0.,0.,1.,1.])
    r_vector = np.asarray([0.,4.,0.,0.,0.,0.])
    
    #################################
    ############ Parameters #########
    #################################
    # Parameters of the ego vehicle
    T_int = 60
    n_int = 4
    m_int = 2
    # Parameters of the obstacle
    obs1_x0 = 20
    obs1_y0 = 0
    obs1_velocity = 3
    obs2_x0 = 0
    obs2_y0 = 4
    obs2_velocity = 6

    # There are totally 5 additional variables
    # [t, obs1_x, obs1_y, obs2_x, obs2_y]
    objective_function = iLQR_numpy.objective_function_log_barrier_class   (
                                                                            (x_u - r_vector)@C_matrix@(x_u - r_vector),
                                                                            x_u,
                                                                            inequality_constraints_list,
                                                                            [obs1_x, obs1_y, obs2_x, obs2_y]
                                                                )
                                                                
#%%
if __name__ == "__main__":
    # Therefore, the additional parameters for the feedfoward pass show in the shape(T_int, 5)
    # The first one is the log barrier parameter t, the remainings are the position of the obstacle
    additional_obj_parameters_matrix = np.zeros((T_int, 5))
    for tau in range(T_int):
        additional_obj_parameters_matrix[tau] = np.asarray((0.5,    obs1_x0+h_constant_int*obs1_velocity*tau, obs1_y0, 
                                                                    obs2_x0+h_constant_int*obs2_velocity*tau, obs2_y0),
                                                                    dtype = np.float64)

    iLQR_log_barrier = iLQR_numpy.iLQR_log_barrier_class(   dynamic_model, 
                                                            objective_function,T_int,
                                                            additional_variables_for_objective_function = additional_obj_parameters_matrix)
    print(  "################################\n"+
            "#######Starting Iteration#######\n"+
            "################################\n"+
            "Initial Cost: %.5e"%(iLQR_log_barrier.objective_function_value_last))
    print("Iteration No.\t Backward Time \t Forward Time \t Objective Value\t")
    iLQR_log_barrier.clear_objective_function_value_last()
    iLQR_log_barrier.backward_pass()
    iLQR_log_barrier.forward_pass()
    time1 = tm.time()
    for j in [0.5, 1., 2., 5., 10., 20., 50., 100.]:
        for tau in range(T_int):
            additional_obj_parameters_matrix[tau,0] = j
            iLQR_log_barrier.update_additional_variables_for_objective_function(additional_obj_parameters_matrix)
        for i in range(100):
            time2 = tm.time()
            iLQR_log_barrier.backward_pass()
            time3 = tm.time()
            (obj, is_stop) = iLQR_log_barrier.forward_pass()
            time4 = tm.time()
            print("%5d\t\t %.5e\t %.5e\t %.5e"%(i,time3-time2,time4-time3,obj))
            if is_stop:
                iLQR_log_barrier.clear_objective_function_value_last()
                print("Complete One Inner Loop! The log barrier parameter t is %.5f"%(j) + " in this iteration!\n")
                print("Iteration No.\t Backward Time \t Forward Time \t Objective Value")
                break
    time5 = tm.time()
    print("Completed! All Time:%.5e"%(time5-time1))
    io.savemat("test.mat",{"result": iLQR_log_barrier.trajectory_list})  

#%%