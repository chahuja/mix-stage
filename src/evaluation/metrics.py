import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from pathlib import Path
import time
import pdb

from argsUtils import *

import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import linalg
import scipy.stats

from argparse import Namespace
from argsUtils import get_args_perm
from pycasper.BookKeeper import BookKeeper
from pathlib import Path
import copy
import trainer_chooser

def get_model(path2weights):
  args_new = Namespace(load=path2weights, cuda=-1, save_dir=Path(path2weights).parent.as_posix(), pretrained_model=1)
  args, args_perm = get_args_perm()
  args.__dict__.update(args_perm[0])
  args.__dict__.update(args_new.__dict__)
  book = BookKeeper(args, [], args_dict_update = {'load_data':0, 'pretrained_model':1, 'sample_all_styles':0, 'mix':0, 'optim_separate':None, 'path2data':args.path2data})
  Trainer = trainer_chooser.trainer_chooser(book.args)
  trainer = Trainer(args, [], args_dict_update = {'load_data':0, 'pretrained_model':1, 'path2data':args.path2data})
  trainer.model.eval()
  return trainer.model

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()
    
  def reset(self):
    self.val = 0
    self.avg = torch.Tensor([0])[0]
    self.sum = 0
    self.count = 0
    self.val2 = 0
    self.sum_energy = 0
    self.avg_energy = 0
    
  def update(self, val, n=1, val2=None):
    self.count += n
    self.val = val
    self.sum += val * n
    self.avg = self.sum / self.count
    self.val2 = val2
    if val2 is not None:
      self.sum_energy += val2 * n
      self.avg_energy = self.sum_energy / self.count
      
  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class Stack():
  def __init__(self, metric, n=0, speakers=[], sample_styles=['same']):
    self.metric = metric
    if n == 0:
      self.metrics = {}
    else:
      self.metrics = {i:[copy.deepcopy(metric) for i in range(n)] for i in sample_styles}
    self.speakers = speakers
    assert len(self.speakers) == n

  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9], idx=0, kwargs_name='same'):
    self.metric(y, gt, mask_idx)
    if self.metrics:
      self.metrics[kwargs_name][idx](y, gt, mask_idx)

  def reset(self):
    self.metric.reset()
    for metric_key in self.metrics:
      for metric in self.metrics[metric_key]:
        metric.reset()

  def get_averages(self, desc):
    if self.metrics:
      return self.metric.get_averages(desc), {metric_key: {self.speakers[i]:metric.get_averages(desc) for i, metric in enumerate(self.metrics[metric_key])} for metric_key in self.metrics}
    else:
      return self.metric.get_averages(desc)    
  
class L1():
  def __init__(self):
    self.average_meter = AverageMeter('L1')

  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9]):
    mask = sorted(list(set(range(int(y.shape[-1]/2))) - set(mask_idx)))
    y = y.view(y.shape[0], y.shape[1], 2, -1) ## (B, T, 2, feats)
    gt = gt.view(gt.shape[0], gt.shape[1], 2, -1) ## (B, T, 2, feats)

    self.average_meter.update(torch.nn.functional.l1_loss(y[..., mask], gt[..., mask]), n=y.shape[0])

  def reset(self):
    self.average_meter.reset()

  def get_averages(self, desc):
    return {'{}_L1'.format(desc):self.average_meter.avg.item()}

class VelL1():
  def __init__(self):
    self.average_meter = AverageMeter('VelL1')

  def get_vel(self, x):
    return x[:, 1:] - x[:, :-1]
  
  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9]):
    mask = sorted(list(set(range(int(y.shape[-1]/2))) - set(mask_idx)))
    y = y.view(y.shape[0], y.shape[1], 2, -1) ## (B, T, 2, feats)
    gt = gt.view(gt.shape[0], gt.shape[1], 2, -1) ## (B, T, 2, feats)
    y_vel = self.get_vel(y)
    gt_vel = self.get_vel(gt)
    self.average_meter.update(torch.nn.functional.l1_loss(y_vel[..., mask], gt_vel[..., mask]), n=y.shape[0])

  def reset(self):
    self.average_meter.reset()

  def get_averages(self, desc):
    return {'{}_VelL1'.format(desc):self.average_meter.avg.item()}


class F1():
  def __init__(self, num_clusters=8):
    self.num_clusters = num_clusters
    self.reset()
    self.labels = list(range(num_clusters))
    
  def __call__(self, y, gt, mask_idx=None):
    self.cm += confusion_matrix(gt.reshape(-1), y.reshape(-1), labels=self.labels)
  
  def reset(self):
    self.cm = np.zeros((self.num_clusters, self.num_clusters))

  def get_precision(self):
    precision = np.diag(self.cm)/np.sum(self.cm, axis=0)
    return np.nan_to_num(precision)

  def get_recall(self):
    recall = np.diag(self.cm)/np.sum(self.cm, axis=1)
    return np.nan_to_num(recall)
  
  def get_F1(self):
    # returns weighted F1 score
    precision = self.get_precision()
    recall = self.get_recall()
    f1 = 2*(precision*recall/(precision + recall))
    try:
      f1 = np.average(np.nan_to_num(f1), weights=self.cm.sum(axis=1))
    except:
      f1 = 0
    return f1

  def get_acc(self):
    return np.diag(self.cm).sum()/self.cm.sum()
  
  def get_averages(self, desc):
    return {'{}_acc'.format(desc):self.get_acc(),
            '{}_F1'.format(desc):self.get_F1(),
            '{}_precision'.format(desc):np.mean(self.get_precision()),
            '{}_recall'.format(desc):np.mean(self.get_recall())}

class Diversity():
  def __init__(self, mean):
    self.div = AverageMeter(name='diversity')
    self.div_gt = AverageMeter(name='diversity_gt')
    self.mean = mean
    
  def reset(self):
    self.div.reset()
    self.div_gt.reset()

  def __call__(self, y, gt, mask_idx=None):
    ### (B, feats), (B, feats), (1, feats)
    self.div.update(torch.nn.functional.l1_loss(y, self.mean.expand_as(y)), n=y.shape[0])
    self.div_gt.update(torch.nn.functional.l1_loss(gt, self.mean.expand_as(gt)), n=y.shape[0])

  def get_averages(self, desc):
    return {'{}_diversity'.format(desc):self.div.avg.item(),
    '{}_diversity_gt'.format(desc):self.div_gt.avg.item()}

class Expressiveness():
  def __init__(self, mean):
    self.spatial = AverageMeter(name='spatial')
    self.spatial_norm = AverageMeter(name='spatial_norm')
    self.energy = AverageMeter(name='energy')
    self.power = AverageMeter(name='power')
    self.mean = mean
    
  def reset(self):
    self.spatial.reset()
    self.energy.reset()
    self.power.reset()

  def get_dist(self, y, mean):
    y = y.reshape(y.shape[0], 2, -1)
    mean = mean.reshape(mean.shape[0], 2, -1)
    return (((y-mean)**2).sum(dim=-2)**0.5).mean(-1)
    
  def get_expressivity(self, y, gt, mean):
    return ((self.get_dist(y, mean) - self.get_dist(gt, mean))**2).mean(-1)**0.5

  def get_vel(self, x):
    return x[1:] - x[:-1]

  def window_smoothing(self, x, k=5):
    x = x.view(1, x.shape[0], x.shape[1]).transpose(2, 1)
    weight = torch.ones(x.shape[-2], 1, k).double()/k
    padding = int((k-1)/2)
    with torch.no_grad():
      x = torch.nn.functional.conv1d(x, weight, padding=padding, groups=x.shape[-2])
    return x.squeeze(0).transpose(1, 0)
  
  def __call__(self, y, gt, mask_idx=None):
    self.spatial.update(self.get_expressivity(y, gt, self.mean), n=y.shape[0])
    self.spatial_norm.update(self.get_expressivity(self.mean, gt, self.mean),
                             n=y.shape[0])
    y_v, gt_v = self.get_vel(y), self.get_vel(gt)
    #gt_v = self.window_smoothing(gt_v)
    self.energy.update(self.get_expressivity(y_v, gt_v, torch.zeros_like(y_v)), n=y_v.shape[0])
    y_a, gt_a = self.get_vel(y_v), self.get_vel(gt_v)
    #gt_a = self.window_smoothing(gt_a)
    self.power.update(self.get_expressivity(y_a, gt_a, torch.zeros_like(y_a)), n=y_a.shape[0])

    #self.spatial.update()

  def get_averages(self, desc):
    if self.spatial_norm.avg.item() > 0:
      spatialNorm = self.spatial.avg.item()/self.spatial_norm.avg.item()
    else:
      spatialNorm = 1000
    return {'{}_spatialNorm'.format(desc):spatialNorm,
            '{}_spatial'.format(desc):self.spatial.avg.item(),
            '{}_energy'.format(desc):self.energy.avg.item(),
            '{}_power'.format(desc):self.power.avg.item()}
  
class PCK():
  '''Computes PCK for different values of alpha and for each joint and returns it as a dictionary'''
  def __init__(self, alphas=[0.1, 0.2], num_joints=52):
    self.alphas = alphas
    self.num_joints = num_joints
    self.avg_meters = {'pck_{}_{}'.format(al, jnt):AverageMeter('pck_{}_{}'.format(al, jnt)) for al in alphas for jnt in range(num_joints)}
    self.avg_meters.update({'pck_{}'.format(alpha):AverageMeter('pck_{}'.format(alpha)) for alpha in self.alphas})
    self.avg_meters.update({'pck':AverageMeter('pck')})

  '''
  y:  (B, 2, joints)
  gt: (B, 2, joints)
  '''
  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9]):
    B = y.shape[0]
    dist = (((y - gt)**2).sum(dim=1)**0.5)
    for alpha in self.alphas:
      thresh = self.get_thresh(gt, alpha)
      pck = self.pck(dist, thresh)
      for jnt in range(self.num_joints):
        key = 'pck_{}_{}'.format(alpha, jnt)
        self.avg_meters[key].update(pck.mean(dim=0)[jnt], n=B)

      mask = sorted(list(set(range(self.num_joints)) - set(mask_idx)))
      self.avg_meters['pck_{}'.format(alpha)].update(pck[:, mask].mean(), n=B*len(mask))

    for alpha in self.alphas:
      self.avg_meters['pck'].update(self.avg_meters['pck_{}'.format(alpha)].avg, n=B*len(mask))
      
  def pck(self, dist, thresh):
    return (dist < thresh).to(torch.float)
    
  def get_thresh(self, gt, alpha):
    h = gt[:, 0, :].max(dim=-1).values - gt[:, 0, :].min(dim=-1).values
    w = gt[:, 1, :].max(dim=-1).values - gt[:, 1, :].min(dim=-1).values
    thresh = alpha * torch.max(torch.stack([h, w], dim=-1), dim=-1, keepdim=True).values
    return thresh

  def get_averages(self, desc):
    averages = {}
    for alpha in self.alphas:
      for jnt in range(self.num_joints):
        key = 'pck_{}_{}'.format(alpha, jnt)
        out_key = '{}_pck_{}_{}'.format(desc, alpha, jnt)
        averages.update({out_key:self.avg_meters[key].avg.item()})

      key = 'pck_{}'.format(alpha)
      out_key = '{}_pck_{}'.format(desc, alpha)
      averages.update({out_key:self.avg_meters[key].avg.item()})
    key = 'pck'
    out_key = '{}_pck'.format(desc)
    averages.update({out_key:self.avg_meters[key].avg.item()})
    return averages

  def reset(self):
    for key in self.avg_meters:
      self.avg_meters[key].reset()

class InceptionScoreStyle():
  def __init__(self, num_clusters, weight, eps=1E-6):
    self.p_y = AverageMeter('p_y')
    self.p_yx = AverageMeter('p_yx')
    self.p_y_subset = AverageMeter('p_y')
    self.p_yx_subset = AverageMeter('p_yx')
    self.f1 = F1(num_clusters=num_clusters)
    self.f1_subset = F1(num_clusters=weight.shape[0])
    self.cce = AverageMeter('cce')
    self.cce_subset = AverageMeter('cce')
    
    self.eps = eps
    self.classifier = get_model("save/inception_score/exp_1503_cpk_m_speaker_['all']_model_StyleClassifier_G_weights.p")
    self.classifier.eval()
    self.weight = weight.long().squeeze(-1)
    self.emb = torch.nn.Embedding(weight.shape[0], weight.shape[1], _weight=weight)

  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9]):
    #mask = sorted(list(set(range(int(y.shape[-1]/2))) - set(mask_idx)))
    #y = y.view(y.shape[0], y.shape[1], 2, -1) ## (B, T, 2, feats)
#    gt = gt.view(gt.shape[0], gt.shape[1], 2, -1) ## (B, T, 2, feats)
    y = y.view(-1, 64, y.shape[-1]) ## must have 64 time steps
    y = self.classifier(y, None)[0]
    p_y = torch.nn.functional.softmax(y, dim=-1)
    p_y_subset = torch.nn.functional.softmax(y[:, self.weight], dim=-1)
    self.f1_subset(p_y[:, self.weight].argmax(-1), gt[:, 0]) ## assuming that there are only speakers this is being trained form
    self.cce_subset.update(torch.nn.functional.cross_entropy(y[:, self.weight], gt[:, 0], reduction='mean'), n=y.shape[0])

    ## Inception Score Updates
    self.update_IS(p_y, self.p_y, self.p_yx)
    self.update_IS(p_y_subset, self.p_y_subset, self.p_yx_subset)
    
    gt = self.emb(gt[:, 0]).squeeze(-1).long()
    self.f1(p_y.argmax(-1), gt)
    self.cce.update(torch.nn.functional.cross_entropy(y, gt, reduction='mean'), n=y.shape[0])
    
  def update_IS(self, p_y, meter_p_y, meter_p_yx):
    meter_p_y.update(p_y.mean(0), n=p_y.shape[0])
    meter_p_yx.update((p_y * torch.log(p_y + self.eps)).mean(0), n=p_y.shape[0])

  def get_IS(self, p_y, p_yx):
    p_y = p_y.avg
    p_yx = p_yx.avg
    kl_d = p_yx - p_y * torch.log(p_y + self.eps)
    is_score = torch.exp(kl_d.sum()).item()
    return is_score
    
  def reset(self):
    self.p_y.reset()
    self.p_yx.reset()
    self.p_y_subset.reset()
    self.p_yx_subset.reset()
    self.f1.reset()
    self.f1_subset.reset()
    self.cce.reset()
    self.cce_subset.reset()

  def get_averages(self, desc):
    is_score = self.get_IS(self.p_y, self.p_yx)
    is_score_subset = self.get_IS(self.p_y_subset, self.p_yx_subset)
    avgs = {'{}_style_IS'.format(desc): is_score,
            '{}_style_IS_subset'.format(desc): is_score_subset,
            '{}_style_cce'.format(desc): self.cce.avg.item(),
            '{}_style_cce_subset'.format(desc): self.cce_subset.avg.item()}
    avgs.update(self.f1.get_averages(desc+'_style'))
    avgs.update(self.f1_subset.get_averages(desc+'_style_subset'))
    return avgs


class FID():
  def __init__(self):
    self.gt_sum_meter = AverageMeter('gt_sum')
    self.gt_square_meter = AverageMeter('gt_square')
    self.y_sum_meter = AverageMeter('y_sum')
    self.y_square_meter = AverageMeter('y_square')

  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9]):
    mask = sorted(list(set(range(int(y.shape[-1]/2))) - set(mask_idx)))
    y = y.view(y.shape[0], y.shape[1], 2, -1)[..., mask].view(y.shape[0]*y.shape[1], -1) ## (B, T, 2, feats) -> (B*T, masked_feats*2) 
    gt = gt.view(gt.shape[0], gt.shape[1], 2, -1)[..., mask].view(gt.shape[0]*gt.shape[1], -1) ## (B, T, 2, feats) -> (B*T, masked_feats*2) 
    self.gt_sum_meter.update(gt.mean(0, keepdim=True), n=gt.shape[0])
    self.y_sum_meter.update(y.mean(0, keepdim=True), n=y.shape[0])
    self.gt_square_meter.update(gt.T.matmul(gt)/gt.shape[0], n=gt.shape[0])
    self.y_square_meter.update(y.T.matmul(y)/y.shape[0], n=y.shape[0])
    
  def reset(self):
    self.gt_sum_meter.reset()
    self.y_sum_meter.reset()
    self.gt_square_meter.reset()
    self.y_square_meter.reset()

  def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance. 
    Borrowed from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
      'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
      'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
      msg = ('fid calculation produces singular product; '
             'adding %s to diagonal of cov estimates') % eps
      print(msg)
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
      if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        m = np.max(np.abs(covmean.imag))
        raise ValueError('Imaginary component {}'.format(m))
      covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

  def get_averages(self, desc):
    try:
      N = self.gt_sum_meter.count
      gt_mu = self.gt_sum_meter.avg.squeeze()
      y_mu = self.y_sum_meter.avg.squeeze()

      gt_sum = self.gt_sum_meter.sum
      y_sum = self.y_sum_meter.sum

      gt_square = self.gt_square_meter.sum
      y_square = self.y_square_meter.sum

      gt_cross = gt_sum.T.matmul(gt_sum)
      y_cross = y_sum.T.matmul(y_sum)

      gt_sigma = (gt_square - gt_cross/N)/(N-1)
      y_sigma = (y_square - y_cross/N)/(N-1) ## divide by N-1 for no bias in the estimator

      fid = self.calculate_frechet_distance(gt_mu.numpy(), gt_sigma.numpy(), y_mu.numpy(), y_sigma.numpy())
    except:
      fid = 1000
    return {'{}_FID'.format(desc):fid}

## Wasserstein - 1 Distance between average speeds and accelerations
class W1():
  def __init__(self):
    self.gt_vel_meter = AverageMeter('gt_vel')
    self.gt_acc_meter = AverageMeter('gt_acc')
    self.y_vel_meter = AverageMeter('y_vel')
    self.y_acc_meter = AverageMeter('y_acc')
    self.ranges = np.arange(0, 300, 0.1)
    
  def get_vel_acc(self, y):
    diff = lambda x:x[:, 1:] - x[:, :-1]
    absolute = lambda x:((x**2).sum(2)**0.5).mean(-1).view(-1)
    vel = diff(y)
    acc = diff(vel)
    vel = absolute(vel)  ## average speed accross all joints
    acc = absolute(acc)
    return vel, acc
    
  def __call__(self, y, gt, mask_idx=[0, 7, 8, 9]):
    mask = sorted(list(set(range(int(y.shape[-1]))) - set(mask_idx)))
    y = y.view(y.shape[0], y.shape[1], 2, -1)[..., mask] ## (B, T, 2, feats) -> (B*T, masked_feats*2) 
    gt = gt.view(gt.shape[0], gt.shape[1], 2, -1)[..., mask] ## (B, T, 2, feats) -> (B*T, masked_feats*2)

    y_vel, y_acc = self.get_vel_acc(y)
    gt_vel, gt_acc = self.get_vel_acc(gt)

    ## make histogram
    y_vel, _ = np.histogram(y_vel, bins=self.ranges)
    y_acc, _ = np.histogram(y_acc, bins=self.ranges)
    gt_vel, _ = np.histogram(gt_vel, bins=self.ranges)
    gt_acc, _ = np.histogram(gt_acc, bins=self.ranges)
    
    self.y_vel_meter.update(y_vel, n=1)
    self.y_acc_meter.update(y_acc, n=1)
    self.gt_vel_meter.update(gt_vel, n=1)
    self.gt_acc_meter.update(gt_acc, n=1)
    
  def reset(self):
    self.y_vel_meter.reset()
    self.y_acc_meter.reset()
    self.gt_vel_meter.reset()
    self.gt_acc_meter.reset()

  def get_averages(self, desc):
    N = self.ranges[:-1]
    try:
      W1_vel = scipy.stats.wasserstein_distance(N, N,
                                                self.y_vel_meter.sum,
                                                self.gt_vel_meter.sum)
      W1_acc = scipy.stats.wasserstein_distance(N, N,
                                                self.y_acc_meter.sum,
                                                self.gt_acc_meter.sum)
    except:
      W1_vel = 1000
      W1_acc = 1000
      
    return {'{}_W1_vel'.format(desc): W1_vel,
            '{}_W1_acc'.format(desc): W1_acc}
