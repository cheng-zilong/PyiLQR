#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import cvxpy as cp
from numba import njit, jitclass, jit
import numba
import torch
from torch import nn, optim
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter
import os

torch.manual_seed(42)
device = "cuda:0"

# class Residual(nn.Module):  #@save
#     """The Residual block of ResNet."""
#     def __init__(self, input_channels, output_channels, is_shorcut = True):
#         super().__init__()
#         self.is_shorcut = is_shorcut
#         self.linear1 = nn.Linear(input_channels, output_channels)
#         self.bn1 = nn.BatchNorm1d(output_channels)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(output_channels, output_channels)
#         self.bn2 = nn.BatchNorm1d(output_channels)

#         self.shorcut = nn.Linear(input_channels, output_channels)
#         self.main_track = nn.Sequential(self.linear1, self.bn1, self.relu, self.linear2, self.bn2)
#     def forward(self, X):
#         if self.is_shorcut:
#             Y = self.main_track(X) + self.shorcut(X)
#         else:
#             Y = self.main_track(X) + X
#         return torch.nn.functional.relu(Y)
        
# class DummyNetwork(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         layer1_no=64
#         layer2_no=32
#         layer3_no=16
#         layer4_no=8

#         self.layer = nn.Sequential( Residual(in_dim, layer1_no),
#                                     Residual(layer1_no, layer1_no, is_shorcut=False),
#                                     Residual(layer1_no, layer2_no),
#                                     Residual(layer2_no, layer2_no, is_shorcut=False),
#                                     Residual(layer2_no, layer3_no),
#                                     Residual(layer3_no, layer3_no, is_shorcut=False),
#                                     Residual(layer3_no, layer4_no),
#                                     Residual(layer4_no, layer4_no, is_shorcut=False),
#                                     nn.Linear(layer4_no, out_dim))

#     def forward(self, x):
#         x = self.layer(x)
#         return x
        
class DummyNetwork(nn.Module):
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

class DynamicModelDataSetWrapper(object):
    """This class generates the dynamic model data for training the neural networks
    """
    def __init__(self, dynamic_model, x0_u_bound, Trial_No = 100, additional_parameters = None):
        """ Initialization
            
            Parameters
            ---------
            dynamic_model : DynamicModelWrapper
                The system you want to generate data
            x0_u_bound : tuple (x0_u_lower_bound, x0_u_upper_bound)
                x0_u_lower_bound : list(m+n)
                x0_u_upper_bound : list(m+n)
                Since you are generating the data with random initial states and inputs, 
                you need to give
                the range of the initial system state variables, and
                the range of the system input variables
            Trial_No : int
                The number of trials 
                The dataset size is Trial_No*(T-1)
            additional_parameters : array(T, q)
                The additional arguments for the system dynamic function, 
                if you dont have the additional_parameters, 
                leave it to be none
        """
        self.Trial_No = Trial_No
        self.T = dynamic_model.T
        self.n = dynamic_model.n
        self.m = dynamic_model.m
        self.dataset_size = self.Trial_No*(self.T-1)
        self.dataset_x = torch.zeros((Trial_No, self.T-1, self.n+self.m,1)).cuda()
        self.dataset_y = torch.zeros((Trial_No, self.T-1, self.n,1)).cuda()
        # The index of the dataset for the next time updating
        self.update_index = 0
        x0_u_lower_bound, x0_u_upper_bound = x0_u_bound
        for i in range(Trial_No):
            x0 = np.random.uniform(x0_u_lower_bound[:self.n], x0_u_upper_bound[:self.n]).reshape(-1,1)
            input_trajectory = np.expand_dims(np.random.uniform(x0_u_lower_bound[self.n:], x0_u_upper_bound[self.n:], (self.T, self.m)), axis=2)
            new_trajectory = torch.from_numpy(dynamic_model.evaluate_trajectory(x0, input_trajectory,additional_parameters)).float().cuda()
            self.dataset_x[i] = new_trajectory[:self.T-1]
            self.dataset_y[i] = new_trajectory[1:, :self.n]
        self.X = self.dataset_x.view(self.dataset_size, self.n+self.m)
        self.Y = self.dataset_y.view(self.dataset_size, self.n)

    def update_dataset(self, new_trajectory):
        """ Insert new data to the dataset and delete the oldest data

            Parameter
            -------
            new_trajectory : array(T, n+m, 1)
                The new trajectory inserted in to the dataset
        """
        print("###### Dataset Updating ######")
        print(" [+] Dataset is updated!")
        self.dataset_x[self.update_index] = torch.from_numpy(new_trajectory[:self.T-1]).float().cuda()
        self.dataset_y[self.update_index] = torch.from_numpy(new_trajectory[1:,:self.n]).float().cuda()
        if self.update_index < self.Trial_No - 1:
            self.update_index = self.update_index + 1
        else:
            self.update_index  = 0
    
    def get_data(self):
        """ Return the data from the dataset

            Return
            ---------
            X : tensor(dataset_size, n+m)\\
            Y : tensor(dataset_size, n)
        """
        return self.X, self. Y


class DynamicModelWrapper(object):
    """ This is a wrapper class for the dynamic model
    """
    def __init__(self, dynamic_model_function, x_u_var, initial_state, initial_input_trajectory, T, additional_parameters_var = None):
        """ Initialization
            
            Parameters
            ---------------
            dynamic_model_function : sympy.array with symbols
                The model dynamic function defined by sympy symbolic array
            x_u_var : tuple with sympy.symbols 
                State and input variables in the model
            initial_state : array(n, 1)
                The initial state vector of the system
            initial_input_trajectory : array(T, m, 1) 
                The initial input vector
            T : int
                The prediction horizon
            additional_var : tuple with sympy.symbols 
                For the use of designing new algorithms
        """
        self.initial_state = initial_state
        self.initial_input_trajectory = initial_input_trajectory
        self.n = int(initial_state.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = T
        if additional_parameters_var is None:
            additional_parameters_var = sp.symbols("no_use")
        self.dynamic_model_lamdify = njit(sp.lambdify([x_u_var, additional_parameters_var], dynamic_model_function, "math"))
        gradient_dynamic_model_array = sp.transpose(sp.derive_by_array(dynamic_model_function, x_u_var))
        self.gradient_dynamic_model_lamdify = njit(sp.lambdify([x_u_var, additional_parameters_var], gradient_dynamic_model_array, "math"))

    def evaluate_trajectory(self, initial_state = None, input_trajectory = None, additional_parameters = None):
        """ Evaluate the system trajectory by given initial states and input vector

            Parameters
            -----------------
            initial_state : array(n, 1)

            input_trajectory : array(T, n, 1)

            additional_parameters : array(T, p_int)
                For the purpose of new method design

            Return
            ---------------
            trajectory : array(T, m+n, 1)
                The whole trajectory
        """
        if initial_state is None:
            initial_state = self.initial_state
        if input_trajectory is None:
            input_trajectory = self.initial_input_trajectory
        return self._evaluate_trajectory_static(self.dynamic_model_lamdify, initial_state, input_trajectory, additional_parameters, self.m, self.n)

    def update_trajectory(self, old_trajectory, K_matrix_all, k_vector_all, alpha, additional_parameters = None): 
        """ Update the trajectory by using iLQR
            Parameters
            -----------------
            old_trajectory : array(T, m+n, 1)
                The trajectory in the last iteration
            K_matrix_all : array(T, m, n)\\
            k_vector_all : array(T, m, 1)\\
            alpha : double
                Step size in this iteration
            additional_parameters : array(T, p_int)
                For the purpose of new method design
            Return
            ---------------
            new_trajectory : array(T, m+n, 1) 
                The updated trajectory
        """
        return self._update_trajectory_static(self.dynamic_model_lamdify, self.m, self.n, old_trajectory, K_matrix_all, k_vector_all, alpha, additional_parameters)

    def evaluate_gradient_dynamic_model_function(self, trajectory, additional_parameters=None):
        """ Return the matrix of the gradient of the dynamic_model
            Parameters
            -----------------
            trajectory : array(T, m+n, 1)
                System trajectory
            additional_parameters : array(T, p_int)
                For the purpose of new method design
            Return
            ---------------
            grad : array(T, m, n)
                The gradient of the dynamic_model
        """
        return self._evaluate_gradient_dynamic_model_function_static(self.gradient_dynamic_model_lamdify, trajectory, additional_parameters)
    
    @staticmethod
    @njit
    def _evaluate_trajectory_static(dynamic_model_lamdify, initial_states, input_trajectory, additional_parameters, m, n):
        T = int(input_trajectory.shape[0])
        if additional_parameters == None:
            additional_parameters = np.zeros((T,1))
        trajectory = np.zeros((T, m+n, 1))
        trajectory[0] = np.vstack((initial_states, input_trajectory[0]))
        for tau in range(T-1):
            trajectory[tau+1, :n, 0] = np.asarray(dynamic_model_lamdify(trajectory[tau,:,0], additional_parameters[tau]),dtype=np.float64)
            trajectory[tau+1, n:] = input_trajectory[tau+1]
        return trajectory

    @staticmethod
    @njit
    def _update_trajectory_static(dynamic_model_lamdify, m, n, old_trajectory, K_matrix_all, k_vector_all, alpha, additional_parameters):
        T = int(K_matrix_all.shape[0])
        if additional_parameters == None:
            additional_parameters = np.zeros((T,1))
        new_trajectory = np.zeros((T, m+n, 1))
        new_trajectory[0] = old_trajectory[0] # initial states are the same
        for tau in range(T-1):
            # The amount of change of state x
            delta_x = new_trajectory[tau, 0:n] - old_trajectory[tau, 0:n]
            # The amount of change of input u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real input of next iteration
            input_u = old_trajectory[tau, n:n+m] + delta_u
            new_trajectory[tau,n:] = input_u
            new_trajectory[tau+1,0:n] = np.asarray(dynamic_model_lamdify(new_trajectory[tau,:,0], additional_parameters[tau]),dtype=np.float64).reshape(-1,1)
            # dont care the input at the last time stamp, because it is always zero
        return new_trajectory

    @staticmethod
    @njit
    def _evaluate_gradient_dynamic_model_function_static(gradient_dynamic_model_lamdify, trajectory, additional_parameters):
        T = int(trajectory.shape[0])
        if additional_parameters == None:
            additional_parameters = np.zeros((T,1))
        F_matrix_initial =  gradient_dynamic_model_lamdify(trajectory[0,:,0], additional_parameters[0])
        F_matrix_list = np.zeros((T, len(F_matrix_initial), len(F_matrix_initial[0])))
        F_matrix_list[0] = np.asarray(F_matrix_initial, dtype = np.float64)
        for tau in range(1, T):
            F_matrix_list[tau] = np.asarray(gradient_dynamic_model_lamdify(trajectory[tau,:,0], additional_parameters[tau]), dtype = np.float64)
        return F_matrix_list


class NeuralDynamicModelWrapper(DynamicModelWrapper):
    """ This is a class to create system model using neural networks
    """
    def __init__(self, networks, initial_state, initial_input_trajectory, T):
        """ Initialization
            networks : nn.module
                Networks used to train the system dynamic model
            initial_state : array(n,1)
                Initial system state
            initial_input_trajectory : array(T, m, 1)
                Initial input trajectory used to generalize the initial trajectory
            T : int
                Prediction horizon
        """
        self.initial_input_trajectory = initial_input_trajectory
        self.initial_state = initial_state
        self.n = initial_state.shape[0]
        self.m = initial_input_trajectory.shape[1]
        self.T = T
        self.model = networks.cuda()
        self.F_matrix_list = torch.zeros(self.T, self.n, self.n+self.m).cuda()
        
        self.const_param = torch.eye(self.n).cuda()
    def pre_train(self, dataset_train, dataset_validation, max_epoch=50000, stopping_criterion = 1e-3, lr = 0.001, model_name = "NeuralDynamic.model"):
        """ Pre-train the model by using randomly generalized data

            Parameters
            ------------
            dataset_train : DynamicModelDataSetWrapper
                Data set for training
            dataset_validation : DynamicModelDataSetWrapper
                Data set for validation
            max_epoch : int
                Maximum number of epochs if stopping criterion is not reached
            stopping_criterion : double
                If the objective function of the training set is less than 
                the stopping criterion, the training is stopped
            lr : double
                Learning rate
            model_name : string
                When the stopping criterion, 
                the model with the given name will be saved as a file
        """
        print("###### Pre-training ######")
        # if the model exists, load the model directly
        if not os.path.exists(model_name):
            print(" [+] Model file does NOT exist. Pre-traning starts...")
            self.writer = SummaryWriter()
            loss_fun = nn.MSELoss()
            # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            X_train, Y_train = dataset_train.get_data()
            X_vali, Y_vali = dataset_validation.get_data()  
            for epoch in range(max_epoch):
                #################
                #### Training ###
                #################
                self.model.train()
                optimizer.zero_grad()
                Y_prediction = self.model(X_train)         
                obj_train = loss_fun(Y_prediction, Y_train) 
                obj_train.backward()                   
                optimizer.step()
                # Print every 100 epoch
                if epoch % 100 == 0:
                    #################
                    ## Evaluation ###
                    #################
                    self.model.eval()
                    Y_prediction = self.model(X_vali)         # Forward Propagation
                    obj_vali = loss_fun(Y_prediction, Y_vali)
                    #################
                    ##### Print #####
                    #################
                    print("[Epoch: %5d]     Train Obj: %.5e     Vali Obj:%.5e"%(
                            epoch + 1,      obj_train.item(),  obj_vali.item()))
                    self.writer.add_scalar('Obj/train', obj_train.item(), epoch)
                    self.writer.add_scalar('Obj/Vali', obj_vali.item(), epoch)
                    if obj_train.item() < stopping_criterion:
                        print(" [+] Pre-training finished! Model file \"" + model_name + "\" is saved!")
                        torch.save(self.model.state_dict(), model_name)
                        self.model.eval()
                        return
            raise Exception("Maximum epoch is reached!")
        else:
            print(" [+] Model file \"" + model_name + "\" exists. Loading....")
            self.model.load_state_dict(torch.load(model_name))
            print(" [+] Loading Completed!")
            self.model.eval()

    def re_train(self, dataset, max_epoch=10000, stopping_criterion = 1e-3, lr = 0.001):
        print("###### Re-training ######")
        print(" [+] Re-traning starts...")
        loss_fun = nn.MSELoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        X_train, Y_train = dataset.get_data()
        for epoch in range(max_epoch):
            #################
            #### Training ###
            #################
            self.model.train()
            optimizer.zero_grad()
            Y_prediction = self.model(X_train)         
            obj_train = loss_fun(Y_prediction, Y_train)  
            obj_train.backward()                   
            optimizer.step()
            if epoch % 10 == 0:
                print("[Epoch: %5d]     Train Obj: %.5e"%(
                        epoch + 1,      obj_train.item()))
                if obj_train.item() < stopping_criterion:
                    print(" [+] Re-training finished!")
                    self.model.eval()
                    return
        self.model.eval()
        raise Exception("Maximum epoch is reached!")

    def next_state(self, current_state_and_input):
        if isinstance(current_state_and_input, list):
            current_state_and_input = np.asarray(current_state_and_input)
        if current_state_and_input.shape[0] != 1:
            current_state_and_input = current_state_and_input.reshape(1,-1)
        x_u = torch.from_numpy(current_state_and_input).float().cuda()
        return self.model(x_u).detach().numpy().reshape(-1,1)

    def evaluate_trajectory(self, initial_states=None, input_trajectory=None, additional_parameters=None):
        if initial_states is None:
            initial_states = self.initial_state
        if input_trajectory is None:
            input_trajectory = self.initial_input_trajectory
        input_trajectory_cuda = torch.from_numpy(input_trajectory).float().cuda()
        trajectory = torch.zeros(self.T, self.n+self.m).cuda()
        trajectory[0] = torch.from_numpy(np.vstack((initial_states, input_trajectory[0]))).float().cuda().reshape(-1)
        for tau in range(self.T-1):
            trajectory[tau+1, :self.n] = self.model(trajectory[tau,:].reshape(1,-1))
            trajectory[tau+1, self.n:] = input_trajectory_cuda[tau+1,0]
        return trajectory.cpu().double().detach().numpy().reshape(self.T, self.m+self.n, 1)

    def update_trajectory(self, old_trajectory, K_matrix_all, k_vector_all, alpha, additional_parameters=None): 
        # Not test!!!!
        # Not test!!!!!
        # Cannot work!!!!
        new_trajectory = np.zeros((self.T, self.m+self.n, 1))
        new_trajectory[0] = old_trajectory[0] # initial states are the same
        for tau in range(self.T-1):
            # The amount of change of state x
            delta_x = new_trajectory[tau, :self.n] - old_trajectory[tau, :self.n]
            # The amount of change of input u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real input of next iteration
            input_u = old_trajectory[tau, self.n:self.n+self.m] + delta_u
            new_trajectory[tau,self.n:] = input_u
            new_trajectory[tau+1,0:self.n,0] = self.model(torch.from_numpy(new_trajectory[tau,:]).float()).detach().numpy()
            # dont care the input at the last time stamp, because it is always zero
        return new_trajectory

    def evaluate_gradient_dynamic_model_function(self, trajectory, additional_parameters=None):
        trajectory_cuda = torch.from_numpy(trajectory[:,:,0]).float().cuda()
        # def get_batch_jacobian(net, x, noutputs):
        #     x = x.unsqueeze(1) # b, 1 ,in_dim
        #     n = x.size()[0]
        #     x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
        #     x.requires_grad_(True)
        #     y = net(x)
        #     input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1)
        #     y.backward(input_val)
        #     return x.grad.data
        for tau in range(0, self.T):
            x = trajectory_cuda[tau]
            x = x.repeat(self.n, 1)
            x.requires_grad_(True)
            y = self.model(x)
            y.backward(self.const_param)
            self.F_matrix_list[tau] = x.grad.data
            # F_matrix_list[tau] = jacobian(self.model, torch.from_numpy().float()).squeeze().numpy()
        # get_batch_jacobian(self.model, trajectory_cuda, 4)
        return self.F_matrix_list.cpu().double().numpy()
        

def vehicle(h_constant = 0.1):
    """Model of a vehicle

        Paramters
        --------
        h_constant : float
            step size

        Return
        --------
        system
        x_u
        n
        m
        
    """
    x_u = sp.symbols('x_u:6')
    d_constanT = 3
    h_d_constanT = h_constant/d_constanT
    b_function = d_constanT \
                + h_constant*x_u[3]*sp.cos(x_u[4]) \
                -sp.sqrt(d_constanT**2 
                    - (h_constant**2)*(x_u[3]**2)*(sp.sin(x_u[4])**2))
    system = sp.Array([  
                x_u[0] + b_function*sp.cos(x_u[2]), 
                x_u[1] + b_function*sp.sin(x_u[2]), 
                x_u[2] + sp.asin(h_d_constanT*x_u[3]*sp.sin(x_u[4])), 
                x_u[3]+h_constant*x_u[5]
            ])
    return system, x_u, 4, 2

def cart_pole(h_constant = 0.02):
    x_u = sp.symbols('x_u:5') # x0: theta x1:dot_theta x2:x x3 dot_x x4:F

# %%
