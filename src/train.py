import random
import torch
import numpy as np
import os
def set_seed(seed):
  """ Set all seeds to make results reproducible (deterministic mode).
      When seed is a false-y value or not supplied, disables deterministic mode. """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('Deterministic Mode!! Seed set to {}'.format(seed))
set_seed(11212)

from argsUtils import argparseNloop
from trainer_chooser import trainer_chooser
import gc
import pdb
import torch

def loop(args, exp_num):
  sample_all_styles = args.sample_all_styles
  finetune_quantile_sample =  args.finetune_quantile_sample
  args_subset = ['exp', 'cpk', 'speaker', 'model', 'note']
  args_dict_update = {'sample_all_styles':0}

  ## Choose Trainer
  Trainer = trainer_chooser(args)

  ## TRAIN
  ## -----------------------------
  trainer = Trainer(args, args_subset, args_dict_update)
  trainer.start_exp()  ## Start Log
  trainer.book._set_seed()

  trainer.train(exp_num)  ## Train

  ## FINE TUNE over quantile
  ## --------------------------------
  if finetune_quantile_sample is not None:
    ## Load best model
    try:
      trainer.book._load_model(trainer.model)
    except:
      pass
    
    ## update train_sampler
    trainer.data.quantile_sample = finetune_quantile_sample
    trainer.data.train_sampler = trainer.data.get_train_sampler(trainer.data.dataset_train,
                                                               trainer.data.train_intervals_dict)
    ## update dataloader
    trainer.data.update_dataloaders(trainer.data.time, trainer.data.window_hop)
    trainer.data_train = trainer.data.train
    trainer.data_dev = trainer.data.dev
    trainer.data_test = trainer.data.test

    ## update args, trainer.args.weighted, trainer.args.epochs
    trainer.args.__dict__.update({'weighted':0, 'num_epochs':20})
    trainer.num_epochs = 20

    ## update bookkeeper to start the training afresh
    trainer.book.best_dev_score = np.inf * trainer.book.dev_sign
    trainer.book.stop_count = 0

    ## Reset optims and learning rates
    trainer.G_optim, trainer.D_optim = trainer.get_optims()
    trainer.schedulers = trainer.get_scheduler()

    ## train again
    trainer.train(exp_num)

  ## SAMPLE ALL STYLES
  ## -----------------------------
  args.__dict__.update({'load':trainer.book.name(trainer.book.weights_ext[0],
                                                 trainer.book.weights_ext[1],
                                                 trainer.args.save_dir)})
  
  if sample_all_styles != 0:
    del trainer
    gc.collect()

    print('Sampling all styles!!!')

    ## Sample all styles
    args_dict_update = {'render':args.render, 'window_hop':0, 'sample_all_styles':sample_all_styles}
    trainer = Trainer(args, args_subset, args_dict_update)
    trainer.sample(exp_num)
  
  ## SAMPLE
  ## -----------------------------
  
  ## Sample Prep.
  del trainer
  gc.collect()    

  print('Loading the best model and running the sample loop')
  args_dict_update = {'render':args.render, 'window_hop':0, 'sample_all_styles':0}
  
  ## Sample
  trainer = Trainer(args, args_subset, args_dict_update)
  trainer.sample(exp_num)
  
  ## Finish
  trainer.finish_exp()

  ## Print Experiment No.
  print('\nExperiment Number: {}'.format(args.exp))
  
if __name__ == '__main__':
  argparseNloop(loop)
