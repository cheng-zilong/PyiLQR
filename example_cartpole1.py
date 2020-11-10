#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
from iLQRSolver import DynamicModel, ObjectiveFunction, iLQR, iLQRExample
from loguru import logger
from datetime import datetime
import torch
from torch import nn, optim
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter
import sys
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}")

class SmallNetwork(nn.Module):
    """Here is a dummy network that can work well on the vehicle model
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no=128
        layer2_no=64
        self.layer = nn.Sequential( 
                                    nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(), 
                                    nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(layer2_no), nn.ReLU(), 
                                    nn.Linear(layer2_no, out_dim))
    def forward(self, x):
        x = self.layer(x)
        return x

class LargeNetwork(nn.Module):
    """Here is a dummy network that can work well on the vehicle model
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no=800
        layer2_no=400
        self.layer = nn.Sequential( 
                                    nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(), 
                                    nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(layer2_no), nn.ReLU(), 
                                    nn.Linear(layer2_no, out_dim))
    def forward(self, x):
        x = self.layer(x)
        return x

def cartpole1_vanilla(T = 300, max_iter = 10000, is_check_stop = True):
    file_name = "cartpole1_vanilla_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger_id = iLQRExample.loguru_start(file_name = file_name, T=T, max_iter = max_iter, is_check_stop = is_check_stop)
    #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi,0,0,0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 0.1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ######### iLQR Solver ###########
    #################################
    cartpole1_example = iLQRExample.iLQRExample(dynamic_model, objective_function)
    cartpole1_example.vanilla_iLQR(file_name, max_iter = max_iter, is_check_stop = is_check_stop)
    iLQRExample.loguru_end(logger_id)

def cartpole1_dd_iLQR(   T = 300, 
                        trial_no = 100,
                        stopping_criterion = 1e-4,
                        max_iter=100,
                        decay_rate=0.99,
                        decay_rate_max_iters=300):
    file_name = "cartpole1_dd_iLQR_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger_id = iLQRExample.loguru_start(file_name = file_name, T=T, trial_no = trial_no,  stopping_criterion = stopping_criterion,  max_iter = max_iter, decay_rate = decay_rate, decay_rate_max_iters = decay_rate_max_iters)
    #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi,0,0,0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 0.1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ########## Training #############
    #################################
    x0_u_lower_bound = [np.pi, -0, -0, -0, -15]
    x0_u_upper_bound = [np.pi,  0,  0,  0,  15]
    x0_u_bound = (x0_u_lower_bound, x0_u_upper_bound)
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=trial_no)
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=10) 
    nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallNetwork(n+m, n), init_state, init_input, T)
    nn_dynamic_model.pre_train(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_small.model")
    # nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(LargeNetwork(n+m, n), init_state, init_input, T)
    # nn_dynamic_model.pre_train(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_large.model")
    #################################
    ######### iLQR Solver ###########
    #################################
    cartpole1_example = iLQRExample.iLQRExample(dynamic_model, objective_function)
    cartpole1_example.dd_iLQR(file_name, nn_dynamic_model, dataset_train,
                                    re_train_stopping_criterion=stopping_criterion, 
                                    max_iter=max_iter,
                                    decay_rate=decay_rate,
                                    decay_rate_max_iters=decay_rate_max_iters)
    iLQRExample.loguru_end(logger_id)

def cartpole1_net_iLQR( T = 300, 
                        trial_no=100,
                        is_check_stop = False, 
                        stopping_criterion = 1e-4,
                        is_re_train = True,  
                        max_iter=100,
                        decay_rate=0.99,
                        decay_rate_max_iters=300):
    file_name = "cartpole1_net_iLQR_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger_id = iLQRExample.loguru_start(file_name = file_name, T=T, trial_no = trial_no, is_check_stop = is_check_stop, stopping_criterion = stopping_criterion, is_re_train = is_re_train, max_iter = max_iter, decay_rate = decay_rate, decay_rate_max_iters = decay_rate_max_iters)
     #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi,0,0,0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 0.1, 0.1, 1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 100, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ########## Training #############
    #################################
    x0_u_lower_bound = [np.pi, -0, -0, -0, -10]
    x0_u_upper_bound = [np.pi,  0,  0,  0,  10]
    x0_u_bound = (x0_u_lower_bound, x0_u_upper_bound)
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=100)
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=10) 
    # nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallNetwork(n+m, n), init_state, init_input, T)
    # nn_dynamic_model.pre_train(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_small.model")
    nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallNetwork(n+m, n), init_state, init_input, T)
    nn_dynamic_model.pre_train(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_small.model")
    #################################
    ######### iLQR Solver ###########
    #################################
    cartpole1_example = iLQRExample.iLQRExample(dynamic_model, objective_function)
    cartpole1_example.net_iLQR(file_name, nn_dynamic_model, dataset_train, 
                                    is_re_train = is_re_train, 
                                    re_train_stopping_criterion=stopping_criterion, 
                                    max_iter=max_iter,
                                    is_check_stop=is_check_stop,
                                    decay_rate=decay_rate,
                                    decay_rate_max_iters=decay_rate_max_iters)
    iLQRExample.loguru_end(logger_id)

# cartpole1 Neural Gradient
def cartpole1_neural_gradient(  T = 300,
                                is_check_stop = False, 
                                stopping_criterion = 1e-5,
                                is_re_train = True, 
                                max_iter=300,
                                decay_rate=0.995,
                                decay_rate_max_iters=200):
    file_name = "cartpole1_neural_gradient_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi,0,0,0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 0.1, 0.1, 1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 100, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ########## Training #############
    #################################
    x0_u_lower_bound = [np.pi, -0, -0, -0, -10]
    x0_u_upper_bound = [np.pi,  0,  0,  0,  10]
    x0_u_bound = (x0_u_lower_bound, x0_u_upper_bound)
    dataset_train = DynamicModel.DynamicModelGradientDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=100)
    dataset_vali = DynamicModel.DynamicModelGradientDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=10) 
    nn_dynamic_model = DynamicModel.NeuralGradientDynamicModelWrapper(SmallNetwork(n+m, n*(n+m)),init_state,init_input,T)
    nn_dynamic_model.pre_train(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_gradient.model")
    #################################
    ######### iLQR Solver ###########
    #################################
    cartpole1_example = iLQRExample.iLQRExample(dynamic_model, objective_function)

    cartpole1_example.dd_iLQR(file_name, nn_dynamic_model, dataset_train, 
                                    max_iter = max_iter,
                                    is_re_train = is_re_train, 
                                    re_train_stopping_criterion = stopping_criterion, 
                                    is_check_stop = is_check_stop,
                                    decay_rate= decay_rate,
                                    decay_rate_max_iters=decay_rate_max_iters)
# %%
if __name__ == "__main__":
    # cartpole1_vanilla(T = 150, max_iter=10000, is_check_stop = True)

    cartpole1_dd_iLQR(  T = 150,
                        trial_no=100,
                        stopping_criterion = 1e-4,
                        max_iter=1000,
                        decay_rate=0.98,
                        decay_rate_max_iters=200)

    # cartpole1_net_iLQR(T = 300,
    #                 trial_no=300,
    #                 is_check_stop = False, 
    #                 stopping_criterion = 1e-4,
    #                 is_re_train = True, 
    #                 max_iter=1000,
    #                 decay_rate=0.99,
    #                 decay_rate_max_iters=300)

    # cartpole1_neural_gradient(  T = 300,
    #                             is_check_stop = False, 
    #                             stopping_criterion = 1e-5,
    #                             is_re_train = True, 
    #                             max_iter=3000,
    #                             decay_rate=0.995,
    #                             decay_rate_max_iters=200)
# %%
