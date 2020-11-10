#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import torch
from iLQRSolver import DynamicModel, ObjectiveFunction, iLQR

if __name__ == "__main__":
    #################################
    #### Model of the cart-pole #####
    #################################
    T = 200
    cart_pole, x_u, n, m = DynamicModel.cart_pole_advanced()
    initial_states = np.asarray([0,0,0.1,-0.8,0],dtype=np.float64).reshape(-1,1)
    initial_inputs = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cart_pole, x_u, initial_states, initial_inputs, T)
    intial_trajectory = dynamic_model.eval_traj()
    
    #################################
    ###### Weighting Matrices #######
    #################################
    C_matrix_diag = sp.symbols("c:6")
    additional_obj_parameters_matrix = np.zeros((T, 6))
    for tau in range(T):
        if tau < T-1:
            additional_obj_parameters_matrix[tau] = np.asarray((10, 0.1, 10, 10, 1, 0.1), dtype = np.float64)
        else: 
            additional_obj_parameters_matrix[tau] = np.asarray((0, 0, 100, 100, 10000, 0), dtype = np.float64)
    r = np.asarray([0,0,0,1,0,0])
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u-r)@np.diag(np.asarray(C_matrix_diag))@(x_u-r), x_u_var = x_u, add_param_var=C_matrix_diag)

    iLQR_vanilla = iLQR.iLQRWrapper(dynamic_model, objective_function, add_param_objective=additional_obj_parameters_matrix)
    print(  "################################\n"+
            "#######Starting Iteration#######\n"+
            "################################\n"+
            "Initial Cost: %.5e"%(iLQR_vanilla.obj_fun_value_last))
    iLQR_vanilla.backward_pass()
    iLQR_vanilla.forward_pass()
    time1 = tm.time()
    for i in range(2000):
        time2 = tm.time()
        iLQR_vanilla.backward_pass()
        time3 = tm.time()
        (obj, isStop, _, _, _) = iLQR_vanilla.forward_pass()
        time4 = tm.time()
        print("Iter.No.%3d BWTime:%.3e FWTime:%.3e Obj.Val.:%.8e"%(i,time3-time2,time4-time3,obj))
        if isStop:
            break
    time5 = tm.time()
    print("Completed! All Time:%.5e"%(time5-time1))
    io.savemat("cart-pole.mat",{"result": iLQR_vanilla.trajectory})  

# %%