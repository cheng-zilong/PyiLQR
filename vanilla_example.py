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
    ##### Model of the vehicle ######
    #################################
    T_int = 100
    n_int = 4
    m_int = 2
    initial_states = np.asarray([0,0,0,4],dtype=np.float64).reshape(-1,1)
    vehicle, x_u, n_int, m_int = DynamicModel.vehicle()
    dynamic_model = DynamicModel.DynamicModelWrapper(vehicle, x_u, initial_states, np.zeros((T_int,m_int,1)), T_int)
    #################################
    ###### Weighting Matrices #######
    #################################
    C_matrix = np.diag([0.,1.,0.,1.,10.,10.])
    r_vector = np.asarray([0.,1.,0.,4.,0.,0.])
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u - r_vector)@C_matrix@(x_u - r_vector), x_u)
    #################################
    ## Parameters of the vehicle ####
    #################################
    iLQR_vanilla = iLQR.iLQRWrapper(dynamic_model, objective_function)
    

    x0_lower_bound = [-0, -1, -0.3, 0]
    x0_upper_bound = [10, 1, 0.3, 8]
    u0_lower_bound = [-0.3,-3]
    u0_upper_bound = [0.3,3]
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_lower_bound, x0_upper_bound, u0_lower_bound, u0_upper_bound, 
                        dataset_size=100)
    dataset_train.update_train_set(dynamic_model.evaluate_trajectory())
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_lower_bound, x0_upper_bound, u0_lower_bound, u0_upper_bound, 
                        dataset_size=10) 
    nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(DynamicModel.DummyNetwork(n_int+m_int, n_int),initial_states,n_int,m_int,T_int, lr = 0.001)
    nn_dynamic_model.train(dataset_train, dataset_vali, max_epoch=50000, stopping_criterion=1e-4, model_name = "Vehicle.model")
    
    # print(  "################################\n"+
    #         "#######Starting Iteration#######\n"+
    #         "################################\n"+
    #         "Initial Cost: %.5e"%(iLQR_vanilla.objective_function_value_last))
    # iLQR_vanilla.backward_pass()
    # iLQR_vanilla.forward_pass()
    # time1 = tm.time()
    # for i in range(100):
    #     time2 = tm.time()
    #     iLQR_vanilla.backward_pass()
    #     time3 = tm.time()
    #     (obj, isStop) = iLQR_vanilla.forward_pass()
    #     time4 = tm.time()
    #     print("Iteration No.%3d\t Backward Time:%.3e\t Forward Time:%.3e\t Obj. Value:%.8e\t"%(i,time3-time2,time4-time3,obj))
    #     if isStop:
    #         break
    # time5 = tm.time()
    # print("Completed! All Time:%.5e"%(time5-time1))
    # io.savemat("test.mat",{"result": iLQR_vanilla.trajectory_list})  

#%%
trajectory = nn_dynamic_model.evaluate_trajectory(np.zeros((100,2,1)))
jacobian_matrix = nn_dynamic_model.evaluate_gradient_dynamic_model_function(trajectory)
# %%

