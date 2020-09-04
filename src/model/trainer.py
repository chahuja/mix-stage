import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import AdamW
from functools import partial

from model import *

from data import *
from argsUtils import get_args_perm
import evaluation
from animation import save_animation
from parallel import parallel

from pycasper.name import Name
from pycasper.BookKeeper import *
from pycasper import torchUtils

import trainer_chooser
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm
import json
from functools import partial
import itertools
import pickle as pkl
from collections import Counter
import pdb

'''
Class Heirarchy
Trainer Skeleton
  - TrainerBase
     - Trainer
     - TrainerGAN
'''

class TrainerBase():
  def __init__(self, args, args_subset, args_dict_update={}):
    self.book = BookKeeper(args, args_subset, args_dict_update=args_dict_update,
                           tensorboard=args.tb)
    self.args = self.book.args

    ## Training parameters
    self.path2data = self.args.path2data
    self.path2outdata = self.args.path2outdata
    self.speaker = self.args.speaker
    self.modalities = self.args.modalities
    if self.args.input_modalities is None: ## infer inputs and outputs from self.modalities
      self.input_modalities = self.modalities[1:]
    else:
      self.input_modalities = self.args.input_modalities
    if self.args.output_modalities is None:
      self.output_modalities = self.modalities[:1]
    else:
      self.output_modalities = self.args.output_modalities

    self.mask = self.args.mask
    self.mask = list(np.concatenate([np.r_[i] if isinstance(i, int) else np.r_[eval(i)] for i in self.mask])) ## convert ranges to list of numbers
    self.split = self.args.split
    self.batch_size = self.args.batch_size
    self.shuffle = True if self.args.shuffle else False
    self.time = self.args.time
    self.fs_new = self.args.fs_new if isinstance(self.args.fs_new, list) else [self.args.fs_new] * len(self.modalities)
    self.window_hop = self.args.window_hop
    self.num_epochs = self.args.num_epochs
    self.num_clusters = args.num_clusters
    self.feats = self.args.feats
    self.num_training_sample = self.args.num_training_sample
    self.style_losses = self.args.style_losses
    self.style_iters = self.args.style_iters
    self.sample_all_styles = self.args.sample_all_styles
    self.repeat_text = self.args.repeat_text

    self.relative2parent = self.args.relative2parent
    self.quantile_sample = self.args.quantile_sample
    self.quantile_num_training_sample = self.args.quantile_num_training_sample

    self.metrics = self.args.metrics
    self.load_data = self.args.load_data
    self.pretrained_model = self.args.pretrained_model
    self.modelKwargs = {}

    ## parameter to use pad_collate for the dataloaders
    self.text_in_modalities = False
    for modality in self.modalities:
      if 'text' in modality:
        self.text_in_modalities = True

    ## Device
    self.device = torch.device('cuda:{}'.format(self.args.cuda)) if self.args.cuda>=0 else torch.device('cpu')

    ## Get Data
    self.data, self.data_train, self.data_dev, self.data_test = self.get_data()

    ## Get style
    self.style_dict = self.data.style_dict
    self.style_dim = self.args.style_dim

    ## Data shape
    self.data_shape = self.data.shape

    # define input and output modalities TODO hadcoded
    self.output_modality = self.output_modalities[0]

    ## Parents
    self.parents = self.data.modality_classes[self.output_modality].parents

    ## Get cluster Transform for Cluster based models
    # if self.num_clusters is not None or self.args.pos:
    #   self.cluster = self.get_cluster()
    #   if self.args.pos:
    #     self.num_clusters = len(self.cluster.tagset)

    if self.num_clusters is not None:
      self.cluster = self.get_cluster()
    if self.args.pos:
      self.cluster_pos = self.get_pos_cluster()
      self.num_clusters_pos = len(self.cluster_pos.tagset)

    if args.preprocess_only:
      print('Data Preprocessing done')
      exit(1)

    ## Create Model
    self.update_modelKwargs()
    self.model = self.get_model()
    self.model.to(self.device).double()
    #device_ids = list(range(torch.cuda.device_count()))
    #self.model = nn.DataParallel(self.model, device_ids=device_ids)

    self.book._copy_best_model(self.model)
    print('Model Created')

    ## Load model
    if self.args.load:
      print('Loading Model')
      self.book._load_model(self.model)

    ## Loss Function
    self.criterion = self.get_criterion()

    ## Optimizers
    self.G_optim, self.D_optim = self.get_optims()

    ## Scheduler
    self.schedulers = self.get_scheduler()

    ## ZNorm
    self.pre = self.get_pre()

    ## Remove Joints / Reinsert Joints from data
    self.transform = self.get_transforms()

    ## transform the confidence matrix
    self.transform_confidence = self.get_transforms()
    self.confidence_loss = Confidence(beta=1, epsilon=0.5)

    ## label histogram
    if self.num_clusters is not None:
      self.num_styles = len(self.speaker) if self.speaker[0] != 'all' else len(self.data.speakers)
      if self.sample_all_styles: ## if all styles are being sampled, create the permutation of the kwargs_names
        kwargs_names = ['{}_{}'.format(sp1, sp2) for sp2 in self.speaker for sp1 in self.speaker if sp1 != sp2]
      else:
        kwargs_names = ['style']
        kwargs_names.append('same')
      self.labels_hist = {kwargs_name:{desc:{i:torch.zeros(self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in kwargs_names}
      self.labels_hist_tensor = {kwargs_name:{desc:{i:torch.zeros(1, self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in kwargs_names}

      #self.labels_hist = {kwargs_name:{desc:{i:torch.zeros(self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in ['same', 'style']}
      #self.labels_hist_tensor = {kwargs_name:{desc:{i:torch.zeros(1, self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in ['same', 'style']}

    if args.mix and args.load:
      self.Stack = partial(evaluation.Stack, n=len(self.data.speaker), speakers=self.data.speaker, sample_styles=['mix'])
    elif self.args.sample_all_styles != 0 and args.load:
      sample_styles = ['same'] + ['_'.join(list(perm)) for perm in itertools.permutations(self.speaker, 2)]
      self.Stack = partial(evaluation.Stack, n=len(self.data.speaker), speakers=self.data.speaker, sample_styles=sample_styles)
    elif self.args.load:
      self.Stack = partial(evaluation.Stack, n=len(self.data.speaker), speakers=self.data.speaker, sample_styles=['same', 'style'])
    else:
      self.Stack = partial(evaluation.Stack, n=0, speakers=[], sample_styles=['same'])

    ## Metrics
    self.metrics_init()

    ## Counter for reweighting
    self.weight_counter = Counter()

  def get_data(self):
    ## Load data iterables
    data = Data(self.path2data, self.speaker, self.modalities, self.fs_new,
                time=self.time, split=self.split, batch_size=self.batch_size,
                shuffle=self.shuffle, window_hop=self.window_hop, style_iters=self.style_iters,
                num_training_sample=self.num_training_sample,
                load_data=self.load_data, sample_all_styles=self.sample_all_styles,
                repeat_text=self.repeat_text, quantile_sample=self.quantile_sample,
                quantile_num_training_sample=self.quantile_num_training_sample,
                weighted=self.args.weighted, filler=self.args.filler, num_training_iters=self.args.num_training_iters)

    data_train = data.train
    data_dev = data.dev
    data_test = data.test

    print('Data Loaded')

    return data, data_train, data_dev, data_test

  def get_criterion(self):
    return eval('torch.nn.' + self.args.loss)(**self.args.lossKwargs)

  def get_pck(self):
    return self.Stack(evaluation.PCK(num_joints=int(self.data_shape[self.output_modality][-1]/2)))

  def get_l1(self):
    return self.Stack(evaluation.L1())

  def get_VelL1(self):
    return self.Stack(evaluation.VelL1())

  def get_Diversity(self):
    mean = self.pre.transforms[-1].variable_dict[self.output_modality][0]
    remove_joints = RemoveJoints(self.mask)
    mean = remove_joints(mean).squeeze(0)
    return self.Stack(evaluation.Diversity(mean))

  def get_Expressiveness(self):
    mean = self.pre.transforms[-1].variable_dict[self.output_modality][0]
    remove_joints = RemoveJoints(self.mask)
    mean = remove_joints(mean).squeeze(0)
    return self.Stack(evaluation.Expressiveness(mean))

  def get_F1(self):
    cluster = KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=8, mask=self.mask, feats=self.feats)
    return self.Stack(evaluation.F1(num_clusters=8)), cluster

  def get_IS(self):
    speakers_rev = {sp:i for i,sp in enumerate(self.data.speakers)}
    if 'all' in self.speaker:
      speaker = self.data.speakers
    else:
      speaker = self.speaker

      weight = torch.Tensor([speakers_rev[sp.split('|')[0]] for sp in speaker]).double().unsqueeze(-1)
    return self.Stack(evaluation.InceptionScoreStyle(len(self.data.speakers), weight))

  def get_FID(self):
    return self.Stack(evaluation.FID())

  def get_W1(self):
    return self.Stack(evaluation.W1())

  def get_optims(self):
    if self.args.gan !=0:
      model_params = list(self.model.G.parameters())
    else:
      model_params = list(self.model.parameters())

    if self.args.optim_separate is not None: ## TODO  harcoded to work with text_encoder
      if self.args.gan != 0:
        bert_params = self.model.G.text_encoder.parameters()
      else:
        bert_params = self.model.text_encoder.parameters()
      bert_params = list(bert_params)
      G_optim = eval('torch.optim.' + self.args.optim)([{'params': bert_params,
                                                         'lr':self.args.optim_separate},
                                                        {'params': list(set(model_params) \
                                                                        - set(bert_params))}],
                                                       lr=self.args.lr, **self.args.optimKwargs)
    else:
      G_optim = eval('torch.optim.' + self.args.optim)(model_params, lr=self.args.lr, **self.args.optimKwargs)

    if self.args.gan != 0:
      D_optim = eval('torch.optim.' + self.args.optim)(self.model.D.parameters(), lr=self.args.lr, **self.args.optimKwargs)
    else:
      D_optim = None

    return G_optim, D_optim
    #return AdamW(self.model.parameters(), lr=self.args.lr, **self.args.optimKwargs)

  def get_scheduler(self):
    schedulers = []
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
      """ Create a schedule with a learning rate that decreases linearly after
      linearly increasing during a warmup period.
      """

      def lr_lambda(current_step):
        if current_step < num_warmup_steps:
          return float(current_step) / float(max(1, num_warmup_steps))
        return max(
          0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
          )

      return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    if self.args.scheduler == 'linear_decay':
      warmup_steps = self.args.scheduler_warmup_steps
      schedulers.append(get_linear_schedule_with_warmup(self.G_optim, warmup_steps, len(self.data.train)*self.num_epochs))
      if self.D_optim is not None:
        schedulers.append(get_linear_schedule_with_warmup(self.D_optim, warmup_steps, len(self.data.train)*self.num_epochs))
    else:
      schedulers.append(lr_scheduler.ExponentialLR(self.G_optim, gamma=self.args.gamma))
      if self.D_optim is not None:
        schedulers.append(lr_scheduler.ExponentialLR(self.D_optim, gamma=self.args.gamma))
    return schedulers

  def get_pre(self):
    transforms = []
    if self.relative2parent:
      transforms.append(Relative2Parent())
      pre_op = Compose(transforms) ## if the mean variance is being calculated for the first time, it uses the pre_op on each batch before calculating mean var
    else:
      pre_op = None

    ## remove text/tokens
    hidden_modalities = ['text/tokens', 'text/filler', 'audio/silence']
    modalities = [mod for mod in self.modalities if mod not in hidden_modalities]

    transforms.append(ZNorm(modalities, key=self.speaker, data=self.data_train, relative2parent=self.relative2parent, pre=pre_op))
    return Compose(transforms)

  def get_transforms(self):
    return Compose([RemoveJoints(self.mask, self.parents)])

  def get_cluster(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=self.num_clusters, mask=self.mask, feats=self.feats)

  def get_gt(self, path2h5):
    Y, h5 = self.data.load(path2h5, self.output_modality)
    feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    Y = Y[()].reshape(-1, 2, feats_shape)
    Y[..., 0] = 0
    h5.close()
    return Y

  def get_confidence_loss(self, batch, y, y_cap):
    key = 'pose/confidence'
    if key in batch:
      confidence = self.transform_confidence(batch[key].to(self.device))
    else:
      return 0

    confidence = confidence.view(*y.shape)
    confidence_loss = self.confidence_loss(y, y_cap, confidence).mean()

    return confidence_loss

  def _update_labels(self, desc, style, kwargs_name):
    if self.num_clusters is not None:
      if isinstance(self.model, GAN):
        model = self.model.G
      else:
        model = self.model

      if kwargs_name is None:
        kwargs_name = 'same'
      ## update only if labels_cap_soft is an attribute in the model
      try:
        if hasattr(model, 'labels_cap_soft'):
          if desc == 'test':
            self.labels_hist_tensor[kwargs_name][desc][style] = torch.cat([self.labels_hist_tensor[kwargs_name][desc][style], model.labels_cap_soft.squeeze(0).detach().cpu().float()], dim=0)
          label = torch.argmax(model.labels_cap_soft.squeeze(0), dim=-1)
          label = label.detach().cpu()
          emb = torch.nn.Embedding(num_embeddings=self.num_clusters,
                                   embedding_dim=self.num_clusters,
                                   _weight=torch.eye(self.num_clusters))
          self.labels_hist[kwargs_name][desc][style] += emb(label).sum(dim=0)
      except:
        pass

  def _save_labels(self):
    if self.num_clusters is not None:
      speakers = self.data.speakers if self.speaker[0] == 'all' else self.speaker
      labels_hist = {kwargs_name:{desc:{speakers[i]:self.labels_hist[kwargs_name][desc][i].numpy().tolist() for i in self.labels_hist[kwargs_name][desc]} for desc in ['test', 'train', 'dev']} for kwargs_name in self.labels_hist}
      labels_hist_tensor = {kwargs_name:{desc:{speakers[i]:self.labels_hist_tensor[kwargs_name][desc][i].numpy() for i in self.labels_hist_tensor[kwargs_name][desc]} for desc in ['test', 'train', 'dev']} for kwargs_name in self.labels_hist_tensor}

      hist_filename = self.book.name('histogram', 'json', self.book.save_dir)
      json.dump(labels_hist, open(hist_filename, 'w'))
      tensor_filename = self.book.name('style', 'pkl', self.book.save_dir)
      pkl.dump(labels_hist_tensor, open(tensor_filename, 'wb'))

  def metrics_init(self):
    self.pck = self.get_pck()
    self.l1 = self.get_l1()
    self.vel_l1 = self.get_VelL1()
    self.diversity = self.get_Diversity()
    self.expressiveness = self.get_Expressiveness()
    self.f1, self.f1_cluster = self.get_F1()
    if not self.pretrained_model: ## if this is a pretrained model do not get self.IS to avoid a loop
      self.IS = self.get_IS()
    self.fid = self.get_FID()
    self.w1 = self.get_W1()

    self.metrics_objects = [self.pck, self.l1, self.vel_l1, self.diversity, self.expressiveness, self.f1, self.fid, self.w1]

    if hasattr(self, 'IS'):
      self.metrics_objects.append(self.IS)

  def metrics_reset(self):
    for obj in self.metrics_objects:
      obj.reset()

  @property
  def metric_order(self):
    if self.metrics:
      metric_order = ['pck', 'F1',
                      'style_IS']
    else:
      metric_order = []
    return metric_order

  def get_metrics(self, desc):
    metrics = {}
    metrics_split = {}

    for metric in self.metrics_objects:
      avgs = metric.get_averages(desc)
      if isinstance(avgs, tuple):
        metrics.update(avgs[0])
        if not metrics_split:
          metrics_split = {kwargs_name:{speaker:{} for speaker in avgs[1][kwargs_name]} for kwargs_name in avgs[1]}

        for kwargs_name in avgs[1]:
          for speaker in avgs[1][kwargs_name]:
            metrics_split[kwargs_name][speaker].update(avgs[1][kwargs_name][speaker])
      else:
        metrics.update(avgs)
    return metrics, metrics_split

  def _save_metrics(self, metrics, filename='metrics'):
    metrics_filename = self.book.name(filename, 'json', self.book.save_dir)
    json.dump(metrics, open(metrics_filename, 'w'))

  def get_kwargs(self, batch, **kwargs_subset):
    kwargs = {}
    keys = ['text/token_count', 'text/token_duration', 'audio/silence', 'text/filler']
    for key in keys:
      if key in batch:
        kwargs[key] = batch[key].to(self.device)

    ## add speaker name
    kwargs.update({'speaker':self.speaker})

    ## add current epoch
    kwargs.update(kwargs_subset)

    return kwargs

  def update_kwargs(self, kwargs):
    '''
    Update kwargs for sample_loop
    '''
    yield kwargs, None

  def start_exp(self):
    self.book._start_log()

  def finish_exp(self):
    self.book._stop_log()

  def _is_warmup(self, epoch, min_epoch):
    return False

  def mem_usage(self):
    out = subprocess.check_output(['nvidia-smi'])
    used = int(out.decode('utf-8').split('\n')[8].split('|')[2].strip().split('/')[0].strip()[:-3])
    total = int(out.decode('utf-8').split('\n')[8].split('|')[2].strip().split('/')[1].strip()[:-3])
    return used, total, (float(used)/total) * 100

  def detach(self, *args):
    for var in args:
      if isinstance(var, list):
        for va in var:
          del va
      elif isinstance(var, torch.Tensor):
        del var

    # for p in self.model.parameters():
    #   if p.grad is not None:
    #     del p.grad
    #torch.cuda.empty_cache()
    #used, total, percent = self.mem_usage()
    #tqdm.write('{}/{}: {}%'.format(used, total, percent))

  def train(self, exp_num):
    for epoch in tqdm(range(self.num_epochs), ncols=20):
      train_loss, train_metrics, train_metrics_split = self.train_loop(self.data_train, 'train', epoch, num_iters=self.args.num_iters)
      dev_loss, dev_metrics, dev_metrics_split = self.train_loop(self.data_dev, 'dev', num_iters=self.args.num_iters)
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.data_test, 'test', num_iters=self.args.num_iters)
      if self.args.scheduler not in ['linear_decay']: ## update lr after each iteration if training bert
        self.schedulers_step() ## Change the Learning Rate

      ## update the weights for data_train
      if self.args.weighted:
        ## Normalize weights
        max_W = 10
        min_W = 0.1
        W_ = self.data_train.sampler.weights
        W_ = (W_ - W_.mean())/W_.std() + 1
        W_ = torch.min(torch.ones(1)[0].double()*max_W,
                       torch.max(torch.zeros(1)[0].double() + min_W, W_))
        if torch.isnan(W_).any():
          W_ = torch.ones_like(W_) ## reinit to ones if Weights suffer a large variation
        self.data_train.sampler.weights = W_

        W = self.data_train.sampler.weights
        D_prob = self.model.D_prob if hasattr(self.model, 'D_prob') else 0
        tqdm.write('W: {}/{}/{}/{}/{}'.format(W.mean(), W.std(), W.min(), W.max(), D_prob))
        most_common = str(self.weight_counter.most_common()[:5])
        least_common = str(self.weight_counter.most_common()[-5:])
        tqdm.write('samples: {} -- {}'.format(most_common, least_common))

      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()

      ## update tensorboard
      if self.args.tb:
        self.book.update_tb({'scalar':[[f'{self.args.cpk}/train', train_loss, epoch],
                                       [f'{self.args.cpk}/dev', dev_loss, epoch],
                                        [f'{self.args.cpk}/test', test_loss, epoch],
                                       [f'{self.args.cpk}/pck_train',
                                        train_metrics['train_pck'], epoch],
                                       [f'{self.args.cpk}/pck_dev',
                                        dev_metrics['dev_pck'], epoch],
                                       [f'{self.args.cpk}/pck_test',
                                        test_metrics['test_pck'],
                                        epoch],
                                       [f'{self.args.cpk}/train_spatialNorm',
                                        train_metrics['train_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/dev_spatialNorm',
                                        dev_metrics['dev_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/test_spatialNorm',
                                        test_metrics['test_spatialNorm'], epoch]
        ]})
                             #'histogram':[[f'{self.args.cpk}/'+name,
                             #param.clone().cpu().detach().numpy(), epoch]
                             # for name, param in model.named_parameters()]})

      ## print results
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])

#      warmup = self._is_warmup(epoch, np.ceil(len(self.data_train)/self.batch_size))
      if self.book.stop_training(self.model, epoch):
        break

    if self.args.num_iters > 0:
      #get the best model
      self.book._load_model(self.model)
      #calculate test loss for the complete dataset
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.data_test, 'test', 0)
      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()
      print('Final Results')
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])


  def train_loop(self, data, desc, epoch=0, num_iters=0):
    ## init
    self.metrics_reset()
    self.running_loss_init()

    if desc == 'train':
      self.model.train(True)
    else:
      self.model.eval()

    bar_format = '{percentage:3.0f}%[{elapsed}<{remaining}]' + ':{desc}'
    bar_format = '{desc}:' +'{n_fmt}/{total_fmt}[{elapsed}<{remaining}]'
    Tqdm = tqdm(data, desc=self.tqdm_desc(desc), leave=False, ncols=20, bar_format=bar_format)
    for count, batch in enumerate(Tqdm):
      self.zero_grad()

      ## update weight counter
      self.weight_counter.update(batch['idx'].numpy())

      ## Transform batch before using in the model
      x, y_, y = self.get_processed_batch(batch)

      ## get kwargs like style
      kwargs = self.get_kwargs(batch, epoch=epoch, sample_flag=0, description=desc)

      ## add noise to output to improve robustness of the model
      noise = torch.randn_like(y) * self.args.noise if self.args.noise > 0 else 0

      y_cap, internal_losses, args = self.forward_pass(desc, x, y+noise, **kwargs)
      args = args[0] if len(args)>0 else {} ## dictionary of args returned by model

      ## check if there are weights in *args
      if args.get('W') is not None and desc=='train' and self.args.weighted > 0:
        W = args['W']
        W_min = 0.1
        self.data_train.sampler.weights[batch['idx']] = torch.max(torch.zeros(1)[0].double() + W_min, W.cpu()) ## clip the weights to positive values

      ## Get mask to calculate the loss function
      src_mask_loss = args.get('src_mask_loss')
      src_mask_loss = src_mask_loss.unsqueeze(-1) if src_mask_loss is not None else torch.ones_like(y[:, :, 0:1])

      ## get confidence values and
      ## calculate confidence loss
      confidence_loss = self.get_confidence_loss(batch, y, y_cap)

      loss = self.calculate_loss(x, (y+noise)*src_mask_loss, y_cap*src_mask_loss, internal_losses)

      ## update tqdm
      losses = [l/c for l,c in zip(self.running_loss, self.running_count)] + [confidence_loss]
      Tqdm.set_description(self.tqdm_desc(desc, losses))
      Tqdm.refresh()

      if np.isnan(losses[0]):
        pdb.set_trace()
      if desc == 'train':
        self.optimize(loss + confidence_loss)

      ## Detach Variables to avoid memory leaks
      #x = x.detach()
      #y = y.detach()
      #loss = loss.detach()
      #y_cap = y_cap.detach()

      ## Evalutation
      y_cap = y_cap.to('cpu')
      src_mask_loss = src_mask_loss.to('cpu')
      with torch.no_grad():
        self.calculate_metrics(y_cap*src_mask_loss, y_*src_mask_loss, 'same', **kwargs)
      self.detach(x, y, loss, y_cap, internal_losses)

      if count>=self.args.debug and self.args.debug: ## debugging by overfitting
        break

      ## if self.args.num_iters > 0, break training
      if count >= num_iters and num_iters > 0 and desc != 'train':
        Tqdm.close()
        break

    metrics = {}
    if self.metrics:
      metrics, metrics_split = self.get_metrics(desc)
    else:
      metrics, metrics_split = {}, {}

    return losses[0], metrics, metrics_split
    #return sum(losses), metrics

  def weight_estimate_loop(self, data, desc, epoch=0, num_iters=0):
    self.model.eval()

    bar_format = '{percentage:3.0f}%[{elapsed}<{remaining}]' + ':{desc}'
    bar_format = '{desc}:' +'{n_fmt}/{total_fmt}[{elapsed}<{remaining}]'
    Tqdm = tqdm(data, desc='update weights: '+self.tqdm_desc(desc), leave=False, ncols=20, bar_format=bar_format)
    W = []
    for count, batch in enumerate(Tqdm):
      ## Transform batch before using in the model
      x, y_, y = self.get_processed_batch(batch)

      ## get kwargs like style
      kwargs = self.get_kwargs(batch, epoch=0, sample_flag=0, description=desc)

      w = self.forward_pass_weight(desc, x, y, **kwargs)
      W.append(w)

      if count>=self.args.debug and self.args.debug: ## debugging by overfitting
        break

      ## if self.args.num_iters > 0, break training
      if count >= num_iters and num_iters > 0:
        break

    ## update the weights for data sampler
    W = torch.cat(W)
    return W

  def sample(self, exp_num):
    ## Create Output Directory
    self.dir_name = self.book.name.dir(self.args.save_dir)

    ## Load best Model
    self.book._load_model(self.model)

    test_loss, test_metrics, test_metrics_split = self.sample_loop(self.data_test.dataset.datasets, 'test')
    train_loss, train_metrics, train_metrics_split = self.sample_loop(self.data_train.dataset.datasets, 'train')
    dev_loss, dev_metrics, dev_metrics_split = self.sample_loop(self.data_dev.dataset.datasets, 'dev')

    if self.sample_all_styles == 0: ## if all styles are sampled, then the results change, hence we don't update it in this case
      ## Save labels histogram
      self._save_labels()

      ## Save sample time metrics
      self._save_metrics(test_metrics_split, 'metrics')
      self._save_metrics(test_metrics, 'cummMetrics')

    print('Sampled- Train:{:.4f}/{:.4f}, '.format(train_loss, train_metrics['train_pck']) + \
          'Dev:{:.4f}/{:.4f}, '.format(dev_loss, dev_metrics['dev_pck']) + \
          'Test:{:.4f}/{:.4f}'.format(test_loss, test_metrics['test_pck']))

    ## print results
    self.book.print_res(epoch=0,
                        key_order=['train', 'dev', 'test'],
                        metric_order=self.metric_order,
                        exp=exp_num,
                        lr=0)

    # self.book.print_res(epoch=0, key_order=['train', 'dev', 'test',
    #                                         'train_pck', 'dev_pck', 'test_pck',
    #                                         'train_VelL1', 'dev_VelL1', 'test_VelL1'],
    #                     exp=exp_num, lr=0)

  def sample_loop(self, data, desc):
    self.metrics_reset()
    self.running_loss_init()
    self.model.eval()

    intervals = []
    start = []
    y_outs = []
    y_animates = []
    filenames = []
    keys = []

    ## collate function
    #if not self.repeat_text:
    if self.text_in_modalities:
      pad_keys = ['text/w2v', 'text/bert', 'text/token_duration', 'text/tokens']
      collate_fn = partial(collate_fn_pad, pad_key=pad_keys, dim=0)
    else:
      collate_fn = None

    len_data = len(data)
    bar_format = '{percentage:3.0f}%|' + '|' + ':{desc}'
    bar_format = '{percentage:3.0f}%[{elapsed}<{remaining}]' + ':{desc}'
    Tqdm = tqdm(data, desc=self.tqdm_desc(desc), leave=False, ncols=20, bar_format=bar_format)
    for count, loader in enumerate(Tqdm):
      ### load ground truth
      Y = self.get_gt(loader.path2h5)

      if len(loader) > 0:
        loader = DataLoader(loader, len(loader), shuffle=False, collate_fn=collate_fn)
        Y_cap = []

        for batch in loader:
          with torch.no_grad():
            ## Transform batch before using in the model
            x, y_, y = self.get_processed_batch(batch)
            kwargs = self.get_kwargs(batch, epoch=0, sample_flag=1, description=desc)

        batch_size = y.shape[0]
        X_ = [x_.view(1, -1, x_.shape[-1]) for x_ in x[:len(self.input_modalities)]]
        for x_ in x[len(self.input_modalities):]: ## hardcoded for auxillary labels
          X_.append(x_.view(1, -1))
        #if len(x) > len(self.input_modalities):
        #  X_.append(x[-1].view(1, -1))

        y = y.view(1, -1, y.shape[-1])
        x = X_

          ## based on kwargs_batch_size, repeat x, and y
          #y = torch.cat([y]*kwargs_batch_size, dim=0)
          #x = [torch.cat([x_]*kwargs_batch_size, dim=0) for x_ in x]
        for kwargs, kwargs_name in self.update_kwargs(kwargs): ## update kwargs like style
          with torch.no_grad():
            ## Forward pass
            y_cap, internal_losses, args = self.forward_pass(desc, x, y, **kwargs)

            ## update labels histogram ## only update when the speaker is sampled with it's style
            self._update_labels(desc=desc, style=int(batch['style'][0, 0].item()), kwargs_name=kwargs_name)

            ## get confidence loss
            confidence_loss = self.get_confidence_loss(batch, y, y_cap)

            loss = self.calculate_loss(x, y, y_cap, internal_losses)

            ## Calculates PCK and reinserts data removed before training
            y_cap = y_cap.to('cpu')
            with torch.no_grad():
              y_cap = y_cap.view(batch_size, -1, y_cap.shape[-1])
              y_cap = self.calculate_metrics(y_cap, y_, kwargs_name, **kwargs)
            Y_cap.append(y_cap)

            ## update tqdm
            losses = [l/c for l,c in zip(self.running_loss, self.running_count)] + [confidence_loss]
            Tqdm.set_description(self.tqdm_desc(desc, losses))
            Tqdm.refresh()

            self.detach(x, y, y_cap, loss, internal_losses)

          if Y_cap:
            intervals.append(batch['meta']['interval_id'][0])
            start.append(torch.Tensor([0]).to(torch.float))
            y_outs.append(torch.cat(Y_cap, dim=0))
            y_animates.append([torch.cat(Y_cap, dim=0), Y])

            dir_name = 'keypoints' if kwargs_name is None else 'keypoints_{}'.format(kwargs_name)
            filenames.append((Path(self.dir_name)/dir_name/'{}/{}/{}.h5'.format(desc,
                                                                                self.data.getSpeaker(intervals[-1]),
                                                                                intervals[-1])).as_posix())
            keys.append(self.output_modality)
            Y_cap = []
            #keys += [self.output_modality] * len(intervals)

      ## Save Keypoints
      if (count + 1) % 100 == 0 or count == len_data - 1: ## save files every 100 batches to prevent memory errors
        parallel(self.data.modality_classes[self.output_modality].append, # fn
                 -1, # n_jobs
                 filenames, keys, y_outs) # fn_args
        intervals = []
        start = []
        y_outs = []
        y_animates = []
        filenames = []
        keys = []

    if self.metrics:
      metrics, metrics_split = self.get_metrics(desc)
    else:
      metrics, metrics_split = {}, {}

    return losses[0], metrics, metrics_split

  def get_processed_batch(self, batch):
    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the first joint
    y = self.transform(y)

    return x, y_, y

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    if kwargs_name is None:
      kwargs_name = 'same'
    #feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    if 'style' in kwargs:
      idx = int(kwargs['style'].view(-1)[0].detach().cpu().item())
      style_vector = kwargs['style'].detach().cpu()
    else:
      idx = 0
      style_vector = torch.zeros(y_cap.shape[0], y_cap.shape[1]).long()

    try:
      self.IS(y_cap, style_vector, self.mask, idx=idx, kwargs_name=kwargs_name)
    except:
      pass

    ## Re-insert Joints
    y_cap = self.transform(y_cap, inv=True, batch_gt=y_)

    ## calculate L1
    self.l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.vel_l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.fid(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## undo normalization
    y_cap = self.pre({self.output_modality:y_cap}, inv=True)[self.output_modality]
    y_cap = y_cap.view(y_cap.shape[0], y_cap.shape[1], 2, -1) ## (B, T, 2, feats)
    y_ = self.pre({self.output_modality:y_}, inv=True)[self.output_modality]
    y_ = y_.view(y_.shape[0], y_.shape[1], 2, -1) ## (B, T, 2, feats)

    ## calculate wasserstein_distance-1 for avg velocity and accelaration
    self.w1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## Hardcode root as (0,0) for eternity for y and gt
    y_cap = y_cap.view(-1, 2, y_cap.shape[-1]) ## (BxT, 2, feats)
    y_cap[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity
    y_cap_out = y_cap

    y_gt = y_.view(-1, 2, y_cap.shape[-1])
    y_gt[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity

    ## calculate and add pck to the average meter
    self.pck(y_cap, y_gt, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## calculate STEEr, SEA and MoCA-{self.num_clusters} scores
    y_cap = self.transform(y_cap.view(1, y_cap.shape[0], -1), save_insert=False)
    y_gt = self.transform(y_gt.view(1, y_gt.shape[0], -1), save_insert=False)
    self.diversity(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.expressiveness(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.f1(self.f1_cluster(y_cap), self.f1_cluster(y_gt), idx=idx, kwargs_name=kwargs_name)
    return y_cap_out

  def get_model(self):
    raise NotImplementedError

  def update_modelKwargs(self):
    raise NotImplementedError

  # def debug_model(self, model):
  #   try:
  #     model()
  #   except RuntimeError as e:
  #     if 'out of memory' in str(e):
  #       print('| WARNING: ran out of memory, retrying batch',sys.stdout)
  #       sys.stdout.flush()
  #       for p in model.parameters():
  #         if p.grad is not None:
  #           del p.grad  # free some memory
  #       torch.cuda.empty_cache()
  #       y= model()
  #     else:
  #       raise e

  def running_loss_init(self):
    raise NotImplementedError

  def tqdm_desc(self):
    raise NotImplementedError

  def zero_grad(self):
    raise NotImplementedError

  def forward_pass(self):
    raise NotImplementedError

  def calculate_loss(self):
    raise NotImplementedError

  def optimize(self, loss):
    if self.args.scheduler in ['linear_decay']:
      self.schedulers_step()

  def schedulers_step(self):
    for sched in self.schedulers:
      sched.step()

class Trainer(TrainerBase):
  '''
  Single modality Trainer with early fusion
  '''
  def __init__(self, args, args_subset, args_dict_update={}):
    super(Trainer, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_model(self):
    return eval(self.args.model)(**self.modelKwargs)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'shape':self.data_shape})

  def running_loss_init(self):
    self.running_loss = [0]
    self.running_count = [1e-10]

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' {:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' {:.4f} H:{:.4f}'.format(0, 0)

  def zero_grad(self):
    self.model.zero_grad()
    self.G_optim.zero_grad()
    if self.D_optim is not None:
      self.D_optim.zero_grad()

  def forward_pass(self, desc, x, y, **kwargs):
    x = torch.cat(x, dim=-1) ## Early Fusion
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y)

    return y_cap, internal_losses, args

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = self.criterion(y_cap, y)
    for i_loss in internal_losses:
      loss += i_loss

    self.running_loss[0] += loss.item() * y_cap.shape[0]
    self.running_count[0] += y_cap.shape[0]

    return loss

  def optimize(self, loss):
    loss.backward()
    self.G_optim.step()
    super().optimize(loss)

class TrainerLate(Trainer):
  '''
  the inputs are not concatenated, passed as a list to the model
  '''
  def __init__(self, args, args_subset, args_dict_update={}):
    super(TrainerLate, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, **kwargs)

    return y_cap, internal_losses, args

TrainerJointLate = TrainerLate
TrainerJoint = Trainer

class TrainerGAN(TrainerBase):
  def __init__(self, args, args_subset, args_dict_update):
    super(TrainerGAN, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_model(self):
    ## Generator
    G = eval(self.args.model)(**self.modelKwargs)

    ## Discriminator
    if self.args.discriminator is None: ## infer the name of the discriminator
      D_modelname = '_'.join(self.args.model.split('_')[:-1] + ['D'])
    else:
      D_modelname = self.args.discriminator

    ## GAN Wrapper
    D_modelKwargs = {}
    if self.args.weighted:
      GANWrapper = GANWeighted
      D_modelKwargs.update({'out_shape':2})
    else:
      GANWrapper = GAN

    ### add input_shape for self.args.joint
    input_shape = 0
    if self.args.joint:
      for mod in self.input_modalities:
        input_shape += self.data_shape[mod][-1]

    D_modelKwargs.update({'in_channels':self.data_shape[self.output_modality][-1]-2*len(self.mask) + input_shape})
    if 'p' in self.modelKwargs: ## get the dropout parameter in the discrimiator as well
      D_modelKwargs.update({'p':self.args.modelKwargs['p']})

    try:
      D = eval(D_modelname)(**D_modelKwargs)
    except:
      print('{} not defined, hence defaulting to Speech2Gesture_D'.format(D_modelname))
      D = eval('Speech2Gesture_D')(**D_modelKwargs)

    ## GAN
    model = GANWrapper(G, D, lr=self.args.lr, criterion=self.args.loss, optim=self.args.optim,
                       dg_iter_ratio=self.args.dg_iter_ratio, lambda_gan=self.args.lambda_gan,
                       lambda_D=self.args.lambda_D, joint=self.args.joint, input_modalities=self.input_modalities,
                       update_D_prob_flag=self.args.update_D_prob_flag, no_grad=self.args.no_grad)
    return model

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'shape':self.data_shape})

  def running_loss_init(self):
    self.running_loss = [0]*4
    self.running_count = [1e-10]*4

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} H:{:.4f}'.format(0, 0, 0, 0, 0)

  def zero_grad(self):
    self.model.zero_grad()
    self.G_optim.zero_grad()
    self.D_optim.zero_grad()

  def forward_pass(self, desc, x, y, **kwargs):
    x = torch.cat(x, dim=-1) ## Early Fusion
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, **kwargs)
    return y_cap, internal_losses, args

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = 0
    for i, i_loss in enumerate(internal_losses):
      if i < 2:
        if self.model.G_flag: ## TODO
          self.running_loss[i] += i_loss.item() * y_cap.shape[0]
          self.running_count[i] += y_cap.shape[0]
        else:
          self.running_loss[i+2] += i_loss.item() * y_cap.shape[0]
          self.running_count[i+2] += y_cap.shape[0]

      loss += i_loss
    return loss

  def get_norm(self, model):
    params = []
    for param in model.parameters():
      params.append(param.grad.view(-1))
    return torch.norm(torch.cat(params))

  def optimize(self, loss):
    loss.backward()
    if self.model.G_flag: ## TODO
      torch.nn.utils.clip_grad_norm_(self.model.G.parameters(), 1) ## TODO
      self.G_optim.step() ## TODO
    else:
      torch.nn.utils.clip_grad_norm_(self.model.D.parameters(), 1) ## TODO
      self.D_optim.step() ## TODO
    super().optimize(loss)

class TrainerLateGAN(TrainerGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def forward_pass_weight(self, desc, x, y, **kwargs):
    w = self.model.estimate_weights(x, y, input_modalities=self.input_modalities, **kwargs)
    return w

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, desc=desc, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, desc=desc, **kwargs)

    return y_cap, internal_losses, args


class TrainerLateCluster(TrainerLate):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.transform_cluster = self.get_transforms()

  def running_loss_init(self):
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' pose:{:.4f} label:{:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' pose:{:.4f} label:{:.4f} H:{:.4f}'.format(0, 0, 0)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = self.criterion(y_cap, y)
    self.running_loss[0] += loss.item() * y_cap.shape[0]
    self.running_count[0] += y_cap.shape[0]
    for i, i_loss in enumerate(internal_losses):
      self.running_loss[i+1] += i_loss.item() * y_cap.shape[0]
      self.running_count[i+1] += y_cap.shape[0]
      loss += i_loss

    return loss

  # def update_modelKwargs(self):
  #   modelKwargs = {}
  #   modelKwargs.update(self.args.modelKwargs)
  #   modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
  #                       'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
  #                       'num_clusters':self.num_clusters,
  #                       'cluster':self.cluster,
  #                       'shape':self.data_shape})

  # def get_model(self):

  #   return eval(self.args.model)(**modelKwargs)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape})


  def get_cluster(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=self.num_clusters, mask=self.mask, feats=self.feats)

  def get_processed_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)
    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    ## Append cluster labels
    x.append(labels)

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the masked joints
    y = self.transform(y)

    return x, y_, y

TrainerJointLateCluster = TrainerLateCluster

class TrainerLateClusterGAN(TrainerLateGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super(TrainerGAN, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]
    self.transform_cluster = self.get_transforms()

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape})

  def running_loss_init(self):
    self.running_loss = [0]*5
    self.running_count = [1e-10]*5

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'.format(0, 0, 0, 0, 0, 0)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = 0
    for i, i_loss in enumerate(internal_losses):
      if i < 2:
        if self.model.G_flag: ## TODO
          self.running_loss[i] += i_loss.item() * y_cap.shape[0]
          self.running_count[i] += y_cap.shape[0]
        else:
          if not self.model.fake_flag and i == 1:
            pass
          else:
            self.running_loss[i+2] += i_loss.item() * y_cap.shape[0]
            self.running_count[i+2] += y_cap.shape[0]
      else:
        self.running_loss[i+2] = i_loss.item() * y_cap.shape[0]
        self.running_count[i+2] += y_cap.shape[0]
      loss += i_loss
    return loss

  def get_cluster(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=self.num_clusters, mask=self.mask, feats=self.feats)

  def get_processed_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)
    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    ## Append cluster labels
    x.append(labels)

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the masked joints
    y = self.transform(y)

    return x, y_, y

TrainerJointLateClusterGAN = TrainerLateClusterGAN


class TrainerStyleClassifier(Trainer):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'in_channels':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'shape':self.data_shape,
                             'style_dict':self.style_dict})

  def get_processed_batch(self, batch):
    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch['style'].long()[:,0]

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the first joint
    x = [self.transform(x_) for x_ in x]

    return x, y_, y

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    return y_cap


class TrainerLateClusterStyleGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'style_dict':self.style_dict,
                             'style_dim':self.style_dim})

  def get_kwargs(self, batch, **kwargs_subset):
    kwargs = super().get_kwargs(batch, **kwargs_subset)

    ## Style Vector
    kwargs.update({'style':batch['style'].long().to(self.device)})
    return kwargs

  def update_kwargs(self, kwargs):
    if self.sample_all_styles:
      style_id = kwargs['style'].view(-1)[0].cpu().item()
      kwargs_list = [kwargs.copy()]
      kwargs_names = [None]
      for style_shift in range(1, self.num_styles):
        kwargs_temp = kwargs.copy()
        kwargs_temp['style'] = (kwargs_temp['style'] + style_shift) % self.num_styles
        kwargs_list.append(kwargs_temp)
        style_shift_id = (style_id + style_shift) % self.num_styles
        kwargs_names.append('{}_{}'.format(self.speaker[style_id], self.speaker[style_shift_id]))
        #kwargs_names = [None, 'style']
    else:
      kwargs_list = [kwargs.copy()]
      kwargs['style'] = (kwargs['style'] + 1) % self.num_styles
      kwargs_list.append(kwargs)
      kwargs_names = [None, 'style']

    for kwargs_, kwargs_name in zip(kwargs_list, kwargs_names):
      yield kwargs_, kwargs_name

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'label',
            'id_in', 'id_out',
            'H']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
    if not losses:
      losses = [0]* len(self.running_loss)

    return desc + loss_str.format(*losses)

  # def update_kwargs(self, kwargs):
  #   kwargs_list = [kwargs.copy()]
  #   kwargs['style'] = (kwargs['style'] + 1) % self.num_styles
  #   kwargs_list.append(kwargs)
  #   kwargs_names = [None, 'style']
  #   for kwargs, kwargs_name in zip(kwargs_list, kwargs_names):
  #     yield kwargs, kwargs_name

TrainerJointLateClusterStyleGAN = TrainerLateClusterStyleGAN


class TrainerLateClusterStyleDisentangleGAN(TrainerLateClusterStyleGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'style_dict':self.style_dict,
                             'style_dim':self.style_dim,
                             'style_losses':self.style_losses})


  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'con_+', 'con_-',
            'id_a', 'id_p',
            'c_a', 'c_p',
            'st_a', 'st_p',
            'rec_a', 'rec_p',
            'H']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
    if not losses:
      losses = [0]* len(self.running_loss)

    return desc + loss_str.format(*losses)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = 0
    for i, i_loss in enumerate(internal_losses):
      if i < 2:
        if self.model.G_flag: ## TODO
          self.running_loss[i] += i_loss.item() * y_cap.shape[0]
          self.running_count[i] += y_cap.shape[0]
        else:
          self.running_loss[i+2] += i_loss.item() * y_cap.shape[0]
          self.running_count[i+2] += y_cap.shape[0]
      else:
        self.running_loss[i+2] = i_loss.item() * y_cap.shape[0]
        self.running_count[i+2] += y_cap.shape[0]
      loss += i_loss
    return loss

TrainerJointLateClusterStyleDisentangleGAN = TrainerLateClusterStyleDisentangleGAN

