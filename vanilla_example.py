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
    T = 100
    n = 4
    m = 2
    initial_states = np.asarray([0,0,0,4],dtype=np.float64).reshape(-1,1)
    initial_inputs = np.zeros((T,m,1))
    vehicle, x_u, n, m = DynamicModel.vehicle()
    dynamic_model = DynamicModel.DynamicModelWrapper(vehicle, x_u, initial_states, initial_inputs, T)
    intial_trajectory = dynamic_model.evaluate_trajectory()
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
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_lower_bound, x0_upper_bound, u0_lower_bound, u0_upper_bound, dataset_size=100)
    dataset_train.update_train_set(intial_trajectory)
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_lower_bound, x0_upper_bound, u0_lower_bound, u0_upper_bound, dataset_size=10) 
    nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(DynamicModel.DummyNetwork(n+m, n),initial_states,initial_inputs,n,m,T)
    nn_dynamic_model.train(dataset_train, dataset_vali, max_epoch=50000, stopping_criterion=1e-4, lr = 0.001, model_name = "Vehicle.model")
    
    iLQR_networks = iLQR.iLQRWrapper(nn_dynamic_model, objective_function)
    iLQR_real_system = iLQR.iLQRWrapper(dynamic_model, objective_function)

    for iter_no in range(1):
        print(  "################################\n"+
                "#######Starting Iteration#######\n"+
                "################################\n"+
                "Initial Network Obj: %.5e\n"%(iLQR_networks.get_objective_function_value())+
                "Initial Real System Obj: %.5e"%(iLQR_real_system.get_objective_function_value()))

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

