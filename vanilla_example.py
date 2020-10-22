#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import importlib
from iLQRSolver import DynamicModel, ObjectiveFunction, iLQR

importlib.reload(iLQRSolver)


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
                -sp.sqrt(d_constant_int**2 
                    - (h_constant_int**2)*(x_u[3]**2)*(sp.sin(x_u[4])**2))
    initial_states = np.asarray([0,0,0,0],dtype=np.float64).reshape(-1,1)
    dynamic_model = DynamicModel.dynamic_model_wrapper  (   
                        sp.Array([  
                                    x_u[0] + b_function*sp.cos(x_u[2]), 
                                    x_u[1] + b_function*sp.sin(x_u[2]), 
                                    x_u[2] + sp.asin(h_d_constant_int*x_u[3]*sp.sin(x_u[4])), 
                                    x_u[3]+h_constant_int*x_u[5]
                                ]), x_u, initial_states)
    #################################
    ###### Weighting Matrices #######
    #################################
    C_matrix = np.diag([0.,1.,0.,1.,10.,10.])
    r_vector = np.asarray([0.,1.,0.,4.,0.,0.])
    objective_function = ObjectiveFunction.objective_function_wrapper((x_u - r_vector)@C_matrix@(x_u - r_vector), x_u)
    #################################
    ## Parameters of the vehicle ####
    #################################
    T_int = 100

    iLQR_vanilla = iLQR.iLQR_wrapper(dynamic_model, objective_function, T_int)
    
    print(  "################################\n"+
            "#######Starting Iteration#######\n"+
            "################################\n"+
            "Initial Cost: %.5e"%(iLQR_vanilla.objective_function_value_last))
    iLQR_vanilla.backward_pass()
    iLQR_vanilla.forward_pass()
    time1 = tm.time()
    for i in range(100):
        time2 = tm.time()
        iLQR_vanilla.backward_pass()
        time3 = tm.time()
        (obj, isStop) = iLQR_vanilla.forward_pass()
        time4 = tm.time()
        print("Iteration No.%3d\t Backward Time:%.3e\t Forward Time:%.3e\t Obj. Value:%.8e\t"%(i,time3-time2,time4-time3,obj))
        if isStop:
            break
    time5 = tm.time()
    print("Completed! All Time:%.5e"%(time5-time1))
    io.savemat("test.mat",{"result": iLQR_vanilla.trajectory_list})  

#%%