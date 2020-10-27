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
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.linear1 = nn.Linear(input_channels, input_channels)
#         self.bn1 = nn.BatchNorm1d(input_channels)
#         self.relu = nn.Sigmoid()
#         self.linear2 = nn.Linear(input_channels, output_channels)
#         self.bn2 = nn.BatchNorm1d(output_channels)

#         self.shorcut = nn.Linear(input_channels, output_channels)
#         self.main_track = nn.Sequential(self.linear1, self.bn1, self.relu, self.linear2, self.bn2)
#     def forward(self, X):
#         Y = self.main_track(X) + self.shorcut(X)
#         return torch.nn.functional.sigmoid(Y)
        
# class DummyNetwork(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         layer1_no=512
#         layer2_no=256
#         layer3_no=128

#         self.layer0 = nn.BatchNorm1d(in_dim)
#         self.layer1 = Residual(in_dim, layer1_no)
#         self.layer2 = Residual(layer1_no, layer2_no)
#         self.layer3 = Residual(layer2_no, layer3_no)
#         self.layer6 = nn.Linear(layer3_no, out_dim)

#     def forward(self, x):
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer6(x)
#         return x
        
class DummyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no=800
        layer2_no=400
        self.layer = nn.Sequential( nn.BatchNorm1d(in_dim), 
                                    nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(), 
                                    nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(layer2_no), nn.ReLU(), 
                                    nn.Linear(layer2_no, out_dim))

    def forward(self, x):
        x = self.layer(x)
        return x

class DynamicModelDataSetWrapper(object):
    def __init__(self, dynamic_model, x0_lower_bound, x0_upper_bound, u0_lower_bound, u0_upper_bound, additional_parameters = None, dataset_size = 100):
        self.dataset_size = dataset_size
        self.T = dynamic_model.T
        self.n = dynamic_model.n
        self.m = dynamic_model.m
        self.dataset_x = torch.zeros((dataset_size, self.T-1, self.n+self.m,1)).cuda()
        self.dataset_y = torch.zeros((dataset_size, self.T-1, self.n,1)).cuda()
        self.update_index = 0
        for i in range(dataset_size):
            x0 = np.random.uniform(x0_lower_bound, x0_upper_bound).reshape(-1,1)
            u_all = np.expand_dims(np.random.uniform(u0_lower_bound, u0_upper_bound, (self.T,self.m)),axis=2)
            new_trajectory = torch.from_numpy(dynamic_model.evaluate_trajectory(x0,u_all,additional_parameters)).float().cuda()
            self.dataset_x[i] = new_trajectory[:self.T-1]
            self.dataset_y[i] = new_trajectory[1:, :self.n]
        self.X = self.dataset_x.view((self.dataset_size)*(self.T-1), self.n+self.m)
        self.Y = self.dataset_y.view((self.dataset_size)*(self.T-1), self.n)

    def update_train_set(self, new_trajectory):
        ###### can be done only by tensor
        ###### To be done in the future
        self.dataset_x[self.update_index] = torch.from_numpy(new_trajectory[:self.T-1]).float().cuda()
        self.dataset_y[self.update_index] = torch.from_numpy(new_trajectory[1:,:self.n]).float().cuda()
        if self.update_index < self.dataset_size - 1:
            self.update_index = self.update_index + 1
        else:
            self.update_index  = 0
    
    def get_dataset(self):
        return self.X, self. Y


class DynamicModelWrapper(object):
    """ This is a wrapper class for the dynamic model
    """
    def __init__(self, dynamic_model_function, x_u_var, initial_states, initial_input_trajectory, T, additional_parameters_var = None, lr = 0.01):
        """ Initialization
            
            Parameters
            ---------------
            dynamic_model_function : sympy.array with symbols
            x_u_var : tuple with sympy.symbols 
                State and input variables in the model
            initial_states : array(n, 1)
                The initial state vector of the system
            initial_input_trajectory : array(T, m, 1) 
                The initial input vector
            T : int
                The prediction horizon
            additional_var : tuple with sympy.symbols 
                For the use of designing new algorithms

            lr : double
                learning rate for the networks
        """
        self.initial_states = initial_states
        self.initial_input_trajectory = initial_input_trajectory
        self.n = int(initial_states.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = T
        if additional_parameters_var is None:
            additional_parameters_var = sp.symbols("no_use")
        self.dynamic_model_lamdify = njit(sp.lambdify([x_u_var, additional_parameters_var], dynamic_model_function, "math"))
        gradient_dynamic_model_array = sp.transpose(sp.derive_by_array(dynamic_model_function, x_u_var))
        self.gradient_dynamic_model_lamdify = njit(sp.lambdify([x_u_var, additional_parameters_var], gradient_dynamic_model_array, "math"))

        ###################
        ##### NN Model ####
        ###################


    def evaluate_trajectory(self, initial_states = None, input_trajectory = None, additional_parameters = None):
        """ Evaluate the system trajectory by given initial states and input vector

            Parameters
            -----------------
            initial_states : array(n, 1)

            input_trajectory : array(T, n, 1)

            additional_parameters : array(T, p_int)
                For the purpose of new method design

            Return
            ---------------
            trajectory : array(T, m+n, 1)
                The whole trajectory
        """
        if initial_states is None:
            initial_states = self.initial_states
        if input_trajectory is None:
            input_trajectory = self.initial_input_trajectory
        return self._evaluate_trajectory_static(self.dynamic_model_lamdify, initial_states, input_trajectory, additional_parameters, self.m, self.n)

    def update_trajectory(self, old_trajectory, K_matrix_all, k_vector_all, alpha, additional_parameters = None): 
        """ Update the trajectory by using iLQR

            Parameters
            -----------------
            old_trajectory : array(T, m+n, 1)
                The trajectory in the last iteration

            K_matrix_all : array(T, m, n)

            k_vector_all : array(T, m, 1)

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
        """Return the matrix of the gradient of the dynamic_model

            Parameters
            -----------------
            trajectory : array(T, m+n, 1) 
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
        trajectory_list = np.zeros((T, m+n, 1))
        trajectory_list[0] = np.vstack((initial_states, input_trajectory[0]))
        for tau in range(T-1):
            trajectory_list[tau+1, :n, 0] = np.asarray(dynamic_model_lamdify(trajectory_list[tau,:,0], additional_parameters[tau]),dtype=np.float64)
            trajectory_list[tau+1, n:] = input_trajectory[tau+1]
        return trajectory_list

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
    def _evaluate_gradient_dynamic_model_function_static(gradient_dynamic_model_lamdify, trajectory_list, additional_parameters):
        T = int(trajectory_list.shape[0])
        if additional_parameters == None:
            additional_parameters = np.zeros((T,1))
        F_matrix_initial =  gradient_dynamic_model_lamdify(trajectory_list[0,:,0], additional_parameters[0])
        F_matrix_list = np.zeros((T, len(F_matrix_initial), len(F_matrix_initial[0])))
        F_matrix_list[0] = np.asarray(F_matrix_initial, dtype = np.float64)
        for tau in range(1, T):
            F_matrix_list[tau] = np.asarray(gradient_dynamic_model_lamdify(trajectory_list[tau,:,0], additional_parameters[tau]), dtype = np.float64)
        return F_matrix_list


class NeuralDynamicModelWrapper(DynamicModelWrapper):
    def __init__(self, networks, initial_states, initial_input_trajectory, n, m, T):
        self.initial_input_trajectory = initial_input_trajectory
        self.initial_states = initial_states
        self.n = n
        self.m = m
        self.T = T
        self.model = networks

        self.loss_fun = nn.MSELoss()
        
    def train(self, dataset_train, dataset_validation, max_epoch=10000, stopping_criterion = 1e-3, lr = 0.01, model_name = "NeuralNetworks.model"):
        if not os.path.exists(model_name):
            print(" [+] Model File Does NOT Exist. Traning Start...")
            self.writer = SummaryWriter()
            self.model.cuda()
            self.model.train()
            # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            X_train, Y_train = dataset_train.get_dataset()
            for epoch in range(max_epoch):
                self.optimizer.zero_grad()             
                Y_prediction = self.model(X_train)         
                cost_train = self.loss_fun(Y_prediction, Y_train) 
                cost_train.backward()                   
                self.optimizer.step()

                cost_vali = self.validate(dataset_validation)
                print("[Epoch: %5d] \t Train Obj: %.5e \t Vali Obj:%.5e"%(
                        epoch + 1,     cost_train.item(),  cost_vali.item()))
                self.writer.add_scalar('Obj/train', cost_train.item(), epoch)
                self.writer.add_scalar('Obj/Vali', cost_vali.item(), epoch)
                if cost_vali.item() < stopping_criterion:
                    print(" [+] Training finished! Model File \"" + model_name + "\" Saved!")
                    torch.save(self.model.state_dict(), model_name)
                    self.model.cpu()
                    return
            raise Exception("Maximum Epoch!!!")
        else:
            print(" [+] Model File \"" + model_name + "\" Exists. Loading....")
            self.model.load_state_dict(torch.load(model_name))
            self.model.cpu()
            self.model.eval()
            print(" [+] Loading Completed!")

    def validate(self, dataset):
        self.model.eval()
        X, Y = dataset.get_dataset()  
        with torch.no_grad():             # Zero Gradient Container
            Y_prediction = self.model(X)         # Forward Propagation
            cost_vali = self.loss_fun(Y_prediction, Y)
            return cost_vali

    def next_state(self, current_state_and_input):
        if isinstance(current_state_and_input, list):
            current_state_and_input = np.asarray(current_state_and_input)
        if current_state_and_input.shape[0] != 1:
            current_state_and_input = current_state_and_input.reshape(1,-1)
        x_u = torch.from_numpy(current_state_and_input).float()
        return self.model(x_u).detach().numpy().reshape(-1,1)

    def evaluate_trajectory(self, initial_states=None, input_trajectory=None, additional_parameters=None):
        if initial_states is None:
            initial_states = self.initial_states
        if input_trajectory is None:
            input_trajectory = self.initial_input_trajectory

        trajectory_list = np.zeros((self.T, self.n+self.m, 1))
        trajectory_list[0] = np.vstack((initial_states, input_trajectory[0]))
        for tau in range(self.T-1):
            trajectory_list[tau+1, :self.n, 0] = self.model(torch.from_numpy(trajectory_list[tau,:].reshape(1,-1)).float()).detach().numpy()
            trajectory_list[tau+1, self.n:] = input_trajectory[tau+1]
        return trajectory_list

    def update_trajectory(self, old_trajectory_list, K_matrix_all, k_vector_all, alpha, additional_parameters=None): 
        new_trajectory_list = np.zeros((self.T, self.m+self.n, 1))
        new_trajectory_list[0] = old_trajectory_list[0] # initial states are the same
        for tau in range(self.T-1):
            # The amount of change of state x
            delta_x = new_trajectory_list[tau, :self.n] - old_trajectory_list[tau, :self.n]
            # The amount of change of input u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real input of next iteration
            input_u = old_trajectory_list[tau, self.n:self.n+self.m] + delta_u
            new_trajectory_list[tau,self.n:] = input_u
            new_trajectory_list[tau+1,0:self.n,0] = self.model(torch.from_numpy(new_trajectory_list[tau,:]).float()).detach().numpy()
            # dont care the input at the last time stamp, because it is always zero
        return new_trajectory_list

    def evaluate_gradient_dynamic_model_function(self, trajectory_list, additional_parameters=None):
        F_matrix_list = np.zeros((self.T, self.n, self.n+self.m))
        for tau in range(0, self.T):
            F_matrix_list[tau] = jacobian(self.model, torch.from_numpy(trajectory_list[tau,:].reshape(1,-1)).float()).squeeze().numpy()
        return F_matrix_list


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
