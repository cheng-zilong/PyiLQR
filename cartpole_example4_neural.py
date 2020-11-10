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
    initial_states = np.asarray([0,0,0.2,-1,0],dtype=np.float64).reshape(-1,1)
    initial_inputs = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cart_pole, x_u, initial_states, initial_inputs, T)
    intial_trajectory = dynamic_model.eval_traj()
    
    #################################
    ###### Weighting Matrices #######
    #################################
    C_matrix_diag = np.diag((10., 1., 0., 500., 15., 0.))
    r = np.asarray([0,0,0,1,0,0])
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u-r)@C_matrix_diag@(x_u-r), x_u_var = x_u)

    #################################
    ########## Training #############
    #################################
    x0_u_lower_bound = [-0, -0, -1, -1, -0, -30]
    x0_u_upper_bound = [ 0,  0,  1,  1,  0,  30]
    x0_u_bound = (x0_u_lower_bound, x0_u_upper_bound)
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=1000)
    good_data = io.loadmat("dataset.mat")["dataset"]
    for data in good_data:
        dataset_train.update_dataset(data)
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=10) 
    nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(DynamicModel.DummyNetwork(n+m, n),initial_states,initial_inputs,T)
    nn_dynamic_model.pre_train(dataset_train, dataset_vali, max_epoch=50000, stopping_criterion=1e-3, lr = 0.001, model_name = "CartPole.model")
    
    iLQR_real_system = iLQR.iLQRWrapper(dynamic_model, objective_function)
    trajectory = intial_trajectory
    print(  "###### Starting Iteration ######\n"+
            " [+] Initial Obj: %.5e"%(iLQR_real_system.get_obj_fun_value()))
    
    F_matrix = nn_dynamic_model.evaluate_gradient_dynamic_model_function(trajectory)
    iLQR_real_system.update_F_matrix(F_matrix)
    iLQR_real_system.backward_pass()
    (_, _, _, _, trajectory) = iLQR_real_system.forward_pass()
    trajectory_last = trajectory.copy()
    F_matrix = nn_dynamic_model.evaluate_gradient_dynamic_model_function(trajectory)
    iLQR_real_system.update_F_matrix(F_matrix)
    time1 = tm.time()
    for iter_no in range(100):
        time2 = tm.time()
        iLQR_real_system.backward_pass()
        trajectory_last = trajectory.copy()
        (obj, isStop, _, _, trajectory) = iLQR_real_system.forward_pass()
        # dataset_train.update_dataset(trajectory)
        # nn_dynamic_model.re_train(dataset_train)
        F_matrix = nn_dynamic_model.evaluate_gradient_dynamic_model_function(trajectory)
        iLQR_real_system.update_F_matrix(F_matrix)
        time3 = tm.time()
        print(" [+] Iter.No.:%3d  Iter.Time:%.3e   Obj.Val.:%.8e   TrajectoryDiffNorm:%.8e"%(
                      iter_no,     time3-time2,             obj,   np.linalg.norm((trajectory_last - trajectory).reshape(-1))))
        if isStop:
            break
    time5 = tm.time()
    print(" [!] Completed!. All time:%.5e"%(time5-time1))

    io.savemat("cart-pole_networks.mat",{"result": iLQR_real_system.trajectory})  

# %%