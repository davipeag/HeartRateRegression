
import os
import time
import numpy as np
import torch
import pickle

from collections import OrderedDict
import random
import numpy as np
import os

from torch import (nn, optim)

from torch.utils.data import DataLoader
from torch.utils import data
from torch.backends import cudnn

from sklearn import metrics

from matplotlib import pyplot as plt

from model_objects import (
  PredictorHolder2,
  ISAEHolder2,
  TSAEHolder,
  TrainerParameters,
  JointTrainer,
  EpochTrainer,
  HarHolder
)
from loaders import (DoubleLoaders, MultiReductors, LeaveOneTrialOutCrossValidation)



from joint_models import JointModelsV1


from repositories import (
  ModuleLocalRepository,
  JointModelLocalRepository,
  JointTrainerInitializer,
  TrainerExecutionParameterRepository
)

from repositories import JsonObjectRepository


from model_utils import (compute_har_accuracy,
                         compute_har_accuracy_drop0)

from callers import *



class RandomParameterSelector():
  def __init__(self, options: OrderedDict):
    self._options = options
  
  @property
  def options(self):
    return self._options
  
  def sample_parameter(self):
    choice = dict()
    for k,v in self.options.items():
      choice[k] = random.choice(v)
    return choice
  
  def indices_from_parameter(self, parameter):
    options = self.options
    choice_index = dict()
    for k,v in options.items():
      choice_index[k] = v.index(parameter[k])
    return choice_index

  def parameter_from_indices(self, indices):
    options = self.options
    choice = dict()
    for k,v in options.items():
      choice[k] = v[indices[k]]
    return choice
  
  def name_from_indices(self, indices):
    return "-".join([str(indices[k]) 
                       for k in self.options.keys()])

  def indices_from_name(self, name):
    indices_values = [int(v) for v in name.split("-")]
    indices = dict()
    for k, v in zip(self.options.keys(), indices_values):
      indices[k] = v
    return indices
  
  def parameter_from_name(self, name):
    indices = self.indices_from_name(name)
    return self.parameter_from_indices(indices)
  
  def name_from_parameter(self, parameter):
    indices = self.indices_from_parameter(parameter)
    return  self.name_from_indices(indices)

class TrainingRepository():
  
  def __init__(self, base_dir, parameters, device,
              har_loss_func=nn.CrossEntropyLoss,
              hr_loss_func=nn.L1Loss, 
              tsae_loss_func=nn.L1Loss,
              isae_loss_func=nn.L1Loss,
              optimizer_func = optim.Adam 
              ):
    
    self.har_loss_func= har_loss_func
    self.hr_loss_func= hr_loss_func 
    self.tsae_loss_func=tsae_loss_func
    self.isae_loss_func=isae_loss_func
    self.optimizer_func = optimizer_func 
    
    self._base_dir = base_dir
    os.makedirs(base_dir, exist_ok=True)

    modules_base_dir = self.nested_dir("modules_repo")
    trainer_exec_repo_base_dir = self.nested_dir(
      "trainer_exec_repo")
    epoch_exec_repo_base_dir = self.nested_dir(
      "epoch_exec_repo" 
    )

    self.device = device

    self.module_repo = JointModelLocalRepository(
      modules_base_dir, JointModelsV1)

    self.trainer_exec_repo = TrainerExecutionParameterRepository(
      trainer_exec_repo_base_dir)

    self.epoch_exec_repo = TrainerExecutionParameterRepository(
      epoch_exec_repo_base_dir)

    self.parameters_repo = JsonObjectRepository(
      self.nested_dir("parameters"))
    
    self.parameters_repo.save(parameters, "parameters.json")

    self.parameters = parameters
    pars = parameters
    self.loaders = DoubleLoaders(
      label_columns=['heart_rate','activity_id'],
      ts_size = pars["ts_size"],
      ts_overlap = pars["ts_overlap"],
      sample_size = pars["sample_size"],
      sample_overlap = pars["sample_overlap"],
      label_reduce_function = MultiReductors(),
      feature_reduce_function = lambda v: v.to_numpy().transpose(),
      batch_size = pars["batch_size"],
      feature_columns = [
        'h_temperature', 'a_temperature', 'c_temperature',
        'h_xacc16','h_yacc16', 'h_zacc16',
        'h_xacc6', 'h_yacc6', 'h_zacc6',
        'h_xgyr', 'h_ygyr', 'h_zgyr', 
        'c_xacc16', 'c_yacc16', 'c_zacc16',
        'c_xacc6', 'c_yacc6', 'c_zacc6',
        'c_xgyr', 'c_ygyr', 'c_zgyr',
        'a_xacc16', 'a_yacc16', 'a_zacc16',
        'a_xacc6','a_yacc6', 'a_zacc6',
        'a_xgyr', 'a_ygyr', 'a_zgyr'],
      data_dir = "tmp/normalized",
      train_subjects= pars["train_subjects"],
      validation_subjects= pars["validation_subjects"],
      test_subjects=[1,5,6,8,9],
      num_workers = 0,
      shuffle=True)

    self.loader_train = self.loaders.train_loader()

    self.loader_validation = self.loaders.validation_loader()

  def nested_dir(self, subdir,):
    return os.path.join(self._base_dir, subdir)

  def get_joint_trainer_initializer(self):
    pars = self.parameters
    def get_indexes(source, items):
      return [source.index(i) for i in items]

    def filter_by_indexes(source, indexes):
      return [source[i] for i in indexes]

    assembler_names = ["har_net", "hr_net", "ts_autoencoder_net",
        "is_autoencoder_net"]


    optimzer_kargs = {
      "lr": pars["lr"],
      "weight_decay":pars['weight_decay']}

    loss_weights = pars["loss_weights"]
    loss_functions = [self.har_loss_func(), self.hr_loss_func(),
        self.tsae_loss_func(), self.isae_loss_func()]
    optimizers_classes = [self.optimizer_func for _ in range(4)]
    optimizers_kargs = [optimzer_kargs,optimzer_kargs,
        optimzer_kargs, optimzer_kargs]
    holders = [HarHolder, PredictorHolder2, TSAEHolder,
        ISAEHolder2]
    holder_names =["HAR", "HR", "TSAE", "ISAE"]  


    f_indexes = get_indexes(
      assembler_names, pars["used_assemblers"])

    

    return JointTrainerInitializer(
      filter_by_indexes(assembler_names, f_indexes),
      filter_by_indexes(loss_weights, f_indexes), 
      filter_by_indexes(loss_functions, f_indexes),
      filter_by_indexes(optimizers_classes, f_indexes),
      filter_by_indexes(optimizers_kargs, f_indexes),
      filter_by_indexes(holders, f_indexes),
      filter_by_indexes(holder_names, f_indexes),
      self.device)

  def load(self, fname):
    jm = self.module_repo.load(fname)
    assemblers = jm.assemblers
    jt_initializer = self.get_joint_trainer_initializer()
    
    jt = jt_initializer.initialize(assemblers)
    self.trainer_exec_repo.load(jt, fname)

    et = EpochTrainer(
      self.loader_train, self.loader_validation, [jt])
    self.epoch_exec_repo.load(et, fname)
    return jm, jt, et
  
  def initialize(self):
    pars = self.parameters
    jm = JointModelsV1(
      feature_count = pars["feature_count"],
      ts_size = pars["ts_size"],
      is_size = pars["is_size"],
      lstm_hidden_size = pars["lstm_hidden_size"],
      lstm_input_size = pars["lstm_input_size"],
      ts_h_size = pars["ts_h_size"],
      is_h_size = pars["is_h_size"],
      lstm_num_layers = pars["lstm_num_layers"],
      initial_points = pars["initial_points"],
      fc_net_output_size = pars["fc_net_output_size"],
      fc_net_middle_size = pars["fc_net_middle_size"],
      hr_label_count = pars["hr_label_count"],
      har_label_count = pars["har_label_count"])
    assemblers = jm.assemblers
    jt = self.get_joint_trainer_initializer().initialize(
      assemblers
    )
    et = EpochTrainer(
      self.loader_train, self.loader_validation, [jt])
  
    # et.train_step()
    # et.validation_step()
    return jm, jt, et
  
  def save(self, fname, jm, jt, et):

    try:
      nlt = et._named_losses_train
      epochs =  len(next(iter(nlt.items()))[1])
      nlt = et._named_losses_validation
      epochs =  len(next(iter(nlt.items()))[1])
    except StopIteration:
      print("""empty train or validation state dict, 
             perform a train and/or validation step before attempting to save
             ---not saving ---""")
      return 
    self.module_repo.save(jm, fname)

    self.trainer_exec_repo.save(jt, fname)
  
    self.epoch_exec_repo.save(et, fname)

  def load_or_initialize(self, fname):
    try:
      jm, jt, et = self.load(fname)
    except ValueError:
      jm, jt, et = self.initialize()
    return jm, jt, et


class TrainingRepositoryLeaveOneTrialOut():
  
  def __init__(self, base_dir, parameters, device,
              har_loss_func=nn.CrossEntropyLoss,
              hr_loss_func=nn.L1Loss, 
              tsae_loss_func=nn.L1Loss,
              isae_loss_func=nn.L1Loss,
              optimizer_func = optim.Adam 
              ):
    
    self.har_loss_func= har_loss_func
    self.hr_loss_func= hr_loss_func 
    self.tsae_loss_func=tsae_loss_func
    self.isae_loss_func=isae_loss_func
    self.optimizer_func = optimizer_func 
    
    self._base_dir = base_dir
    os.makedirs(base_dir, exist_ok=True)

    modules_base_dir = self.nested_dir("modules_repo")
    trainer_exec_repo_base_dir = self.nested_dir(
      "trainer_exec_repo")
    epoch_exec_repo_base_dir = self.nested_dir(
      "epoch_exec_repo" 
    )

    self.device = device

    self.module_repo = JointModelLocalRepository(
      modules_base_dir, JointModelsV1)

    self.trainer_exec_repo = TrainerExecutionParameterRepository(
      trainer_exec_repo_base_dir)

    self.epoch_exec_repo = TrainerExecutionParameterRepository(
      epoch_exec_repo_base_dir)

    self.parameters_repo = JsonObjectRepository(
      self.nested_dir("parameters"))
    
    self.parameters_repo.save(parameters, "parameters.json")

    self.parameters = parameters
    pars = parameters

    try:
      pars["seed"]
    except KeyError:
      pars["seed"] = 1735

    self.loaders = LeaveOneTrialOutCrossValidation(
      label_columns=['heart_rate','activity_id'],
      ts_size = pars["ts_size"],
      ts_overlap = pars["ts_overlap"],
      sample_size = pars["sample_size"],
      sample_overlap = pars["sample_overlap"],
      label_reduce_function = MultiReductors(),
      feature_reduce_function = lambda v: v.to_numpy().transpose(),
      batch_size = pars["batch_size"],
      feature_columns = [
        'h_temperature', 'a_temperature', 'c_temperature',
        'h_xacc16','h_yacc16', 'h_zacc16',
        'h_xacc6', 'h_yacc6', 'h_zacc6',
        'h_xgyr', 'h_ygyr', 'h_zgyr', 
        'c_xacc16', 'c_yacc16', 'c_zacc16',
        'c_xacc6', 'c_yacc6', 'c_zacc6',
        'c_xgyr', 'c_ygyr', 'c_zgyr',
        'a_xacc16', 'a_yacc16', 'a_zacc16',
        'a_xacc6','a_yacc6', 'a_zacc6',
        'a_xgyr', 'a_ygyr', 'a_zgyr'],
      data_dir = "tmp/normalized",
      subjects= pars["subjects"],
      test_subjects=[],
      num_folds=pars["num_folds"],
      samples_per_trial=pars["samples_per_trial"],
      current_fold=pars["current_fold"],
      num_workers = 0,
      shuffle=True,
      seed=pars["seed"])

                
    self.loader_train = self.loaders.train_loader()

    self.loader_validation = self.loaders.validation_loader()

  def nested_dir(self, subdir,):
    return os.path.join(self._base_dir, subdir)

  def get_joint_trainer_initializer(self):
    pars = self.parameters
    def get_indexes(source, items):
      return [source.index(i) for i in items]

    def filter_by_indexes(source, indexes):
      return [source[i] for i in indexes]

    assembler_names = ["har_net", "hr_net", "ts_autoencoder_net",
        "is_autoencoder_net"]


    optimzer_kargs = {
      "lr": pars["lr"],
      "weight_decay":pars['weight_decay']}

    loss_weights = pars["loss_weights"]
    loss_functions = [self.har_loss_func(), self.hr_loss_func(),
        self.tsae_loss_func(), self.isae_loss_func()]
    optimizers_classes = [self.optimizer_func for _ in range(4)]
    optimizers_kargs = [optimzer_kargs,optimzer_kargs,
        optimzer_kargs, optimzer_kargs]
    holders = [HarHolder, PredictorHolder2, TSAEHolder,
        ISAEHolder2]
    holder_names =["HAR", "HR", "TSAE", "ISAE"]  


    f_indexes = get_indexes(
      assembler_names, pars["used_assemblers"])

    

    return JointTrainerInitializer(
      filter_by_indexes(assembler_names, f_indexes),
      filter_by_indexes(loss_weights, f_indexes), 
      filter_by_indexes(loss_functions, f_indexes),
      filter_by_indexes(optimizers_classes, f_indexes),
      filter_by_indexes(optimizers_kargs, f_indexes),
      filter_by_indexes(holders, f_indexes),
      filter_by_indexes(holder_names, f_indexes),
      self.device)

  def load(self, fname):
    jm = self.module_repo.load(fname)
    assemblers = jm.assemblers
    jt_initializer = self.get_joint_trainer_initializer()
    
    jt = jt_initializer.initialize(assemblers)
    self.trainer_exec_repo.load(jt, fname)

    et = EpochTrainer(
      self.loader_train, self.loader_validation, [jt])
    self.epoch_exec_repo.load(et, fname)
    return jm, jt, et
  
  def initialize(self):
    pars = self.parameters
    jm = JointModelsV1(
      feature_count = pars["feature_count"],
      ts_size = pars["ts_size"],
      is_size = pars["is_size"],
      lstm_hidden_size = pars["lstm_hidden_size"],
      lstm_input_size = pars["lstm_input_size"],
      ts_h_size = pars["ts_h_size"],
      is_h_size = pars["is_h_size"],
      lstm_num_layers = pars["lstm_num_layers"],
      initial_points = pars["initial_points"],
      fc_net_output_size = pars["fc_net_output_size"],
      fc_net_middle_size = pars["fc_net_middle_size"],
      hr_label_count = pars["hr_label_count"],
      har_label_count = pars["har_label_count"])
    assemblers = jm.assemblers
    jt = self.get_joint_trainer_initializer().initialize(
      assemblers
    )
    et = EpochTrainer(
      self.loader_train, self.loader_validation, [jt])
  
    # et.train_step()
    # et.validation_step()
    return jm, jt, et
  
  def save(self, fname, jm, jt, et):

    try:
      nlt = et._named_losses_train
      epochs =  len(next(iter(nlt.items()))[1])
      nlt = et._named_losses_validation
      epochs =  len(next(iter(nlt.items()))[1])
    except StopIteration:
      print("""empty train or validation state dict, 
             perform a train and/or validation step before attempting to save
             ---not saving ---""")
      return 
    self.module_repo.save(jm, fname)

    self.trainer_exec_repo.save(jt, fname)
  
    self.epoch_exec_repo.save(et, fname)

  def load_or_initialize(self, fname):
    try:
      jm, jt, et = self.load(fname)
    except ValueError:
      jm, jt, et = self.initialize()
    return jm, jt, et

class TrainManagerHarV1():

  def __init__(self, repository_dir, max_epochs, parameters, device,
                har_loss_func=nn.CrossEntropyLoss,
                hr_loss_func=nn.L1Loss, 
                tsae_loss_func=nn.L1Loss,
                isae_loss_func=nn.L1Loss,
                optimizer_func = optim.Adam,
                training_repository_class=TrainingRepository
      ):
      self.parameters = parameters
      self.max_epochs = max_epochs
      self.repository_dir = repository_dir
      self.device = device


      self.tr = training_repository_class(repository_dir, parameters, device,
                                   har_loss_func,
                                    hr_loss_func, 
                                    tsae_loss_func,
                                    isae_loss_func,
                                    optimizer_func)    

      best_validation_fname = "best_validation"
      best_train_fname = "best_train"
      execution_fname = "execution"

      jm, jt, et = self.tr.load_or_initialize(execution_fname)

      self.tr.save(execution_fname, jm, jt, et)

      self.epoch_trainer = et
      har_train_accuracy = None
      har_validation_accuracy = None

      try:
        jm_btrain = self.tr.module_repo.load(best_train_fname)
        har_train_accuracy = compute_har_accuracy_drop0(self.tr.loader_train, jm_btrain.assemblers["har_net"]) 
      except ValueError:
        print("No train file found")

      try:
        jm_bval = self.tr.module_repo.load(best_validation_fname)
        har_validation_accuracy = compute_har_accuracy_drop0(self.tr.loader_validation, jm_bval.assemblers["har_net"]) 
      except ValueError:
        print("No validation file found")



      save_caller = SavingCaller(
        self.tr, jm, jt, et, "execution"
      )


      har_accuracy = MetricComputer(
        self.tr.loader_train,
        compute_har_accuracy_drop0,
        jm.assemblers["har_net"])

      compute_epoch = EpochComputer(et)


      save_train = CallerPipeline(
        ConstantCaller(best_train_fname), save_caller,
        PrintCaller("saving "))

      save_validation = CallerPipeline(
        ConstantCaller(best_validation_fname), save_caller,
        PrintCaller("saving "))

      save_execution = CallerPipeline(
        ConstantCaller("execution"), save_caller,
        PrintCaller("saving "))


      print_epoch = CallerPipeline(
        compute_epoch, PrintCaller("Epoch: ")
      )

      print_har_validation_accuracy = CallerPipeline(
        ConstantCaller(self.tr.loader_validation),
        har_accuracy, PrintCaller("validation accuracy: ")
      )

      print_har_train_accuracy = CallerPipeline(
        ConstantCaller(self.tr.loader_train),
        har_accuracy, PrintCaller("train accuracy: ")
      )

      best_train_cond = MetricCondition(
              self.tr.loader_train,
              compute_har_accuracy_drop0,
              jm.assemblers["har_net"],
              maximizing =True, last_metric=har_train_accuracy)

      best_valid_cond = MetricCondition(
              self.tr.loader_validation,
              compute_har_accuracy_drop0,
              jm.assemblers["har_net"],
              maximizing =True,
              last_metric=har_validation_accuracy)

      git_push_caller = GitPushCaller()
      self.callers = [
        ConditionalCaller(best_train_cond, 
                          CallerPipeline(print_epoch, save_train, print_har_train_accuracy)
                        ),
        ConditionalCaller(best_valid_cond,
                          CallerPipeline(print_epoch, save_validation, print_har_validation_accuracy)
                          ),
        ConditionalCaller(
          EpochCondition(et, lambda e: e% 10==0), et.validation_step),
        ConditionalCaller(
          EpochCondition(et, lambda e: e% 50==0),
                        CallerPipeline(print_epoch, save_execution)),
        ConditionalCaller(
          EpochCondition(et, lambda e: e% 100==0), et.plot_history),
        ConditionalCaller(
          EpochCondition(et, lambda e: e% 100==0), git_push_caller),
        ConditionalCaller(EpochCondition(et, lambda e: e > max_epochs),
                          CallerPipeline(save_execution, interrupt_caller)),
      ]

  def train(self):
    self.epoch_trainer.train_step()
    self.epoch_trainer.validation_step()
    while(True):
      try:
        for caller in self.callers:
          caller()
      except StopIteration:
        print("#### END OF TRAINING ####")
        break
      self.epoch_trainer.train_step()


