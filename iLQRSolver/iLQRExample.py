#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import os
import torch
from iLQRSolver import DynamicModel, ObjectiveFunction, iLQR
from loguru import logger
from scipy.ndimage import gaussian_filter1d
import atexit 

def loguru_start(example_name="N.A.", 
                T = "N.A.",
                trial_no = "N.A.",
                max_iter="N.A.", 
                is_re_train="N.A.", 
                stopping_criterion="N.A.", 
                is_check_stop="N.A.", 
                decay_rate="N.A.", 
                decay_rate_max_iters="N.A."):
    logger_id = logger.add("log\\" + example_name + ".log", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}")
    logger.debug("[+] Starting Iteration: " + example_name)
    logger.debug("[+] prediction_horizon: " + str(T))
    logger.debug("[+] max_iter: " + str(max_iter))
    logger.debug("[+] trial_no: " + str(trial_no))
    logger.debug("[+] is_re_train: " + str(is_re_train))
    logger.debug("[+] stopping_criterion: " + str(stopping_criterion))
    logger.debug("[+] is_check_stop: " + str(is_check_stop))
    logger.debug("[+] decay_rate: " + str(decay_rate))
    logger.debug("[+] decay_rate_max_iters: " + str(decay_rate_max_iters))
    return logger_id
    
def loguru_end(logger_id):
    logger.remove(logger_id)

class iLQRExample(object):
    """This is an example class
    """
    def __init__(self, dynamic_model, obj_fun):
        """ Initialization

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR
        """
        self.dynamic_model = dynamic_model
        self.obj_fun = obj_fun
        self.real_system_iLQR = iLQR.iLQRWrapper(self.dynamic_model, self.obj_fun)

    def log_barrier_iLQR(self, example_name, max_iter = 100, is_check_stop = True):
        """ Solve the constraint problem with log barrier iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            max_iter : int
                The max iteration of iLQR
            is_check_stop : boolean
                Whether check the stopping criterion, if False, then max_iter number of iterations are performed
        """
        iLQR_log_barrier = iLQR.iLQRLogBarrier(self.dynamic_model, self.obj_fun)
        logger.debug("[+ +] Initial Obj.Val.: %.5e"%(iLQR_log_barrier.get_obj_fun_value()))
        iLQR_log_barrier.clear_obj_fun_value_last()
        iLQR_log_barrier.backward_pass()
        iLQR_log_barrier.forward_pass()
        start_time = tm.time()
        for j in [0.5, 1., 2., 5., 10., 20., 50., 100.]:
            self.obj_fun.update_t(j)
            for i in range(max_iter):
                iter_start_time = tm.time()
                iLQR_log_barrier.backward_pass()
                backward_time = tm.time()
                obj, isStop = iLQR_log_barrier.forward_pass(line_search = "feasibility")
                forward_time = tm.time()
                logger.debug("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                            i,  backward_time-iter_start_time,forward_time-backward_time,obj))
                if isStop and is_check_stop:
                    iLQR_log_barrier.clear_obj_fun_value_last()
                    logger.debug("[+ +] Complete One Inner Loop! The log barrier parameter t is %.5f"%(j) + " in this iteration!")
                    logger.debug("[+ +] Iteration No.\t Backward Time \t Forward Time \t Objective Value")
                    break
        end_time = tm.time()
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))
        io.savemat("mat\\" + example_name + ".mat",{"result": iLQR_log_barrier.get_traj()})

    def vanilla_iLQR(self, example_name, max_iter = 100, is_check_stop = True):
        """ Solve the problem with classical iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            max_iter : int
                The max number of iterations of iLQR
            is_check_stop : boolean
                Whether check the stopping criterion, if False, then max_iter number of iterations are performed
        """
        logger.debug("[+ +] Initial Obj.Val.: %.5e"%(self.real_system_iLQR.get_obj_fun_value()))
        self.real_system_iLQR.backward_pass()
        self.real_system_iLQR.forward_pass()
        start_time = tm.time()
        for i in range(max_iter):
            iter_start_time = tm.time()
            self.real_system_iLQR.backward_pass()
            backward_time = tm.time()
            obj, isStop = self.real_system_iLQR.forward_pass()
            forward_time = tm.time()
            logger.debug("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                         i,  backward_time-iter_start_time,forward_time-backward_time,obj))
            if isStop and is_check_stop:
                break
        end_time = tm.time()
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))
        io.savemat("mat\\" + example_name + ".mat",{"result": self.real_system_iLQR.get_traj()})
    
    def dd_iLQR(self, example_name, nn_dynamic_model, dataset_train, re_train_stopping_criterion = 1e-5, max_iter = 100, is_check_stop = True, decay_rate = 1, decay_rate_max_iters = 300):
        """ Solve the problem with nerual network iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            nn_dynamic_model : NeuralDynamicModelWrapper
                The neural network dynamic model
            dataset_train : DynamicModelDataSetWrapper
                Data set for training
            re_train_stopping_criterion : double
                The stopping criterion during re-training
            max_iter : int
                The max number of iterations of iLQR
            decay_rate : double
                Re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            decay_rate_max_iters : int
                The max iterations the decay_rate existing
        """
        logger.debug("[+ +] Initial Obj.Val.: %.5e"%(self.real_system_iLQR.get_obj_fun_value()))
        logger.debug("[+ +] Create Folder: %s"%("mat\\" + example_name))
        os.mkdir("mat\\" + example_name)
        trajectory = self.real_system_iLQR.get_traj()
        new_data = []
        for i in range(int(max_iter)):
            if i == 1:  # skip the compiling time 
                start_time = tm.time()
            iter_start_time = tm.time()
            F_matrix = nn_dynamic_model.eval_grad_dynamic_model(trajectory)
            F_matrix = gaussian_filter1d(F_matrix, 1, axis=0)
            self.real_system_iLQR.update_F_matrix(F_matrix)
            self.real_system_iLQR.backward_pass()
            obj_val, isStop = self.real_system_iLQR.forward_pass()
            if i < decay_rate_max_iters:
                re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            iter_end_time = tm.time()
            iter_time = iter_end_time-iter_start_time
            logger.debug("[+ +] Iter.No.:%3d  Iter.Time:%.3e   Obj.Val.:%.5e"%(
                                    i,        iter_time,       obj_val,   ))
            trajectory = self.real_system_iLQR.get_traj()
            if not isStop: # The last trajectory not updated because it is the same as the previous one
                new_data += [trajectory]
            else: # The last trajectory is updated with noise
                if len(new_data) != 0: # Ensure the optimal trajectroy being in the dataset
                    trajectory_noisy = trajectory
                else:
                    trajectory_noisy = self.dynamic_model.eval_traj(input_traj = (trajectory[:,self.dynamic_model.n:]+np.random.normal(0,1,[self.dynamic_model.m,1])))
                new_data += [trajectory_noisy]
                dataset_train.update_dataset(new_data[-int(self.dynamic_model.T/5):])
                io.savemat("mat\\" + example_name + "\\" + str(i) +".mat",{"optimal_trajectory": self.real_system_iLQR.get_traj(), "trajectroy_noisy": trajectory_noisy})
                nn_dynamic_model.re_train(dataset_train, max_epoch = 100000, stopping_criterion = re_train_stopping_criterion)
                new_data = []
                
        end_time = tm.time()
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))
        


    def net_iLQR(self, example_name, nn_dynamic_model, dataset_train, is_re_train = True, re_train_stopping_criterion = 1e-5, max_iter = 100, is_check_stop = True, decay_rate = 1, decay_rate_max_iters = 300):
        """ Solve the problem with nerual network iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            nn_dynamic_model : NeuralDynamicModelWrapper
                The neural network dynamic model
            dataset_train : DynamicModelDataSetWrapper
                Data set for training
            is_re_train : boolean
                Whether re-train the model during iterations
            re_train_stopping_criterion : double
                The stopping criterion during re-training
            max_iter : int
                The max number of iterations of iLQR
            is_check_stop : boolean
                Whether check the stopping criterion, if False, then max_iter number of iterations are performed
            decay_rate : double
                Re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            decay_rate_max_iters : int
                The max iterations the decay_rate existing
        """
        net_system_iLQR = iLQR.iLQRWrapper(nn_dynamic_model, self.obj_fun)
        logger.debug("[+ +] Initial Real.Obj.Val.: %.5e"%(self.real_system_iLQR.get_obj_fun_value()))
        for i in range(max_iter):
            if i == 1:  # skip the compiling time 
                start_time = tm.time()
            iter_start_time = tm.time()
            net_system_iLQR.backward_pass()
            network_obj, isStop = net_system_iLQR.forward_pass()
            trajectory = self.dynamic_model.eval_traj(input_traj = net_system_iLQR.trajectory[:,self.dynamic_model.n:])
            real_obj = self.obj_fun.eval_obj_fun(trajectory)
            dataset_train.update_dataset(trajectory)
            if i < decay_rate_max_iters:
                re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            iter_end_time = tm.time()
            iter_time = iter_end_time-iter_start_time
            logger.debug("[+ +] Iter.No.:%3d  Iter.Time:%.3e   Net.Obj.Val.:%.5e   Real.Obj.Val.:%.5e"%(
                                i,            iter_time,       network_obj,        real_obj))
            if isStop and is_check_stop:
                break
            if isStop and is_re_train:
                nn_dynamic_model.re_train(dataset_train, max_epoch = 100000, stopping_criterion = re_train_stopping_criterion)
                net_system_iLQR.clear_obj_fun_value_last()

        end_time = tm.time()
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))
        io.savemat("mat\\" + example_name + ".mat",{"result": trajectory})

# %%
