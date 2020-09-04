import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb

import torch
import torch.nn as nn
from transformers import BertModel
import logging
logging.getLogger('transformers').setLevel(logging.CRITICAL)

def num_powers_of_two(x):
  num_powers = 0
  while x>1:
    if x % 2 == 0:
      x /= 2
      num_powers += 1
    else:
      break
  return num_powers

def next_multiple_power_of_two(x, power=5):
  curr_power = num_powers_of_two(x)
  if curr_power < power:
    x = x * (2**(power-curr_power))
  return x

class ConvNormRelu(nn.Module):
  def __init__(self, in_channels, out_channels,
               type='1d', leaky=False,
               downsample=False, kernel_size=None, stride=None,
               padding=None, p=0, groups=1):
    super(ConvNormRelu, self).__init__()
    if kernel_size is None and stride is None:
      if not downsample:
        kernel_size = 3
        stride = 1
      else:
        kernel_size = 4
        stride = 2

    if padding is None:
      if isinstance(kernel_size, int) and isinstance(stride, tuple):
        padding = tuple(int((kernel_size - st)/2) for st in stride)
      elif isinstance(kernel_size, tuple) and isinstance(stride, int):
        padding = tuple(int((ks - stride)/2) for ks in kernel_size)
      elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
        assert len(kernel_size) == len(stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(len(kernel_size), len(stride))
        padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, kernel_size))
      else:
        padding = int((kernel_size - stride)/2)


    in_channels = in_channels*groups
    out_channels = out_channels*groups
    if type == '1d':
      self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm1d(out_channels)
      self.dropout = nn.Dropout(p=p)
    elif type == '2d':
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm2d(out_channels)
      self.dropout = nn.Dropout2d(p=p)
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=0.2)
    else:
      self.relu = nn.ReLU()

  def forward(self, x, **kwargs):
    return self.relu(self.norm(self.dropout(self.conv(x))))

class UNet1D(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)

  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super(UNet1D, self).__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, return_bottleneck=False):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    bn = x
    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)

    if return_bottleneck:
      return x, bn
    else:
      return x

class AudioEncoder(nn.Module):
  '''
  input_shape:  (N, C, time, frequency)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=1, kernel_size=None, stride=None, p=0, groups=1):
    super(AudioEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(128, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=(3,8), stride=1, p=p, groups=groups))

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

class PoseEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1):
    super(PoseEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class PoseStyleEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, t)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1, num_speakers=4):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, num_speakers, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.mean(-1)
    x = x.squeeze(dim=-1)
    return x
    
class PoseDecoder(nn.Module):
  '''
  input_shape:  (N, channels, time)
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=256, style_dim=10, num_clusters=8, out_feats=96, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.num_clusters = num_clusters
    self.style_dim = style_dim
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels+style_dim,
                                                                   input_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=num_clusters)
                                                      for i in range(4)]))
    self.pose_logits = nn.Conv1d(input_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

  def forward(self, x, **kwargs):
    style = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])[:, -self.style_dim:]
    for i, model in enumerate(self.pose_decoder):
      #x = torch.split(x, int(x.shape[1]/self.num_clusters), dim=1)
      #x = torch.cat([torch.cat([x_, kwargs['style']], dim=1) for x_ in x], dim=1)
      x = model(x)
      if i < len(self.pose_decoder) - 1: ## last module
        x = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])
        x = torch.cat([x, style], dim=1).view(x.shape[0], -1, x.shape[-1])
    return self.pose_logits(x)

class StyleDecoder(nn.Module):
  '''
  input_shape:  (N, channels, time)
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=256, num_clusters=10, out_feats=96, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.num_clusters = num_clusters
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels,
                                                                   input_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=num_clusters)
                                                      for i in range(2)]))
    self.pose_logits = nn.Conv1d(input_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

  def forward(self, x, **kwargs):
    x = self.pose_decoder(x)
    return self.pose_logits(x)


#TODO Unify Encoders via input_channel size?
class TextEncoder1D(nn.Module):
  '''
  input_shape:  (N, time, text_features: 300)
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])

    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, time_steps=None, **kwargs):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

class Transpose(nn.Module):
  def __init__(self, idx):
    super().__init__()
    self.param = torch.nn.Parameter(torch.ones(1))
    self.idx = idx

  def forward(self, x, *args, **kwargs):
    return x.transpose(*self.idx)
  
class AudioEncoder1D(nn.Module):
  '''
  input_shape:  (N, time, audio_features: 128)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=128, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, time_steps=None):
    #x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x


        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class LatentEncoder(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels=2, p=0):
    super().__init__()
    enc1 = nn.ModuleList([ConvNormRelu(in_channels, hidden_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(1)])
    enc2 = nn.ModuleList([ConvNormRelu(hidden_channels, hidden_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(2)])
    enc3 = nn.ModuleList([ConvNormRelu(hidden_channels, out_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(1)])
    self.enc = nn.Sequential(*enc1, *enc2, *enc3)

  def forward(self, x):
    x = self.enc(x)
    return x


class ClusterClassify(nn.Module):
  '''
  input_shape: (B, C, T)
  output_shape: (B, num_clusters, T)
  '''
  def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
    super().__init__()
    self.conv = nn.ModuleList()
    self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                             kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in range(5)])

    self.logits = nn.Conv1d(256*groups, num_clusters*groups, kernel_size=1, stride=1, groups=groups)

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    x = self.logits(x)
    return x

class Confidence(nn.Module):
  '''
  0 < confidence <= 1
  '''
  def __init__(self, beta=0.1, epsilon=1e-8):
    super().__init__()
    self.beta = beta
    self.epsilon = epsilon

  def forward(self, y, y_cap, confidence):
    if isinstance(confidence, int):
      confidence = torch.ones_like(y)
    sigma = self.get_sigma(confidence)
    P_YCAP_Y = self.p_ycap_y(y, y_cap, sigma)
    sigma_ycap = self.get_sigma(P_YCAP_Y)
    return self.get_entropy(sigma_ycap)

  def p_ycap_y(self, y, y_cap, sigma):
    diff = -(y-y_cap)**2
    diff_normalized = diff/(2*sigma**2)
    prob = torch.exp(diff_normalized)
    prob_normalized = prob*(1/(2*math.pi*sigma))
    return prob_normalized

  def get_sigma(self, confidence):
    mask = (confidence < self.epsilon).double()
    confidence = (1 - mask) * confidence + torch.ones_like(confidence)*self.epsilon*mask
    sigma = 1/(2*math.pi*confidence)
    return sigma

  ## entropy of a guassian
  def get_entropy(self, sigma):
    return 0.5*(torch.log(2*math.pi*math.e*(sigma**2)))*self.beta

class Repeat(nn.Module):
  def __init__(self, repeat, dim=-1):
    super().__init__()
    self.dim = dim
    self.repeat = repeat
    #self.temp = torch.nn.Parameter(torch.zeros(1))

  def forward(self, x):
    return x.repeat_interleave(self.repeat, self.dim)


class BatchGroup(nn.Module):
  '''
  Group conv networks to run in parallel
  models: list of instantiated models

  Inputs:
    x: list of list of inputs; x[group][batch], len(x) == groups, and len(x[0]) == batches
    labels: uses these labels to give a soft attention on the outputs. labels[batch], len(labels) == batches
            if labels is None, return a list of outputs
    transpose: if true, model needs a transpose of the input
  '''
  def __init__(self, models, groups=1):
    super().__init__()
    if not isinstance(models, list):
      models = [models]
    self.models = nn.ModuleList(models)
    self.groups = groups

  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward(self, x, labels=None, transpose=True, **kwargs):
    if not isinstance(x, list):
      raise 'x must be a list'
    if not isinstance(x[0], list):
      raise 'x must be a list of lists'
    if labels is not None:
      assert isinstance(labels, list), 'labels must be a list'

    groups = len(x)
    assert self.groups == groups, 'input groups should be the same as defined groups'
    batches = len(x[0])

    x = [torch.cat(x_, dim=0) for x_ in x] # batch
    x = torch.cat(x, dim=1)  # group

    if transpose:
      x = x.transpose(-1, -2)
    for model in self.models:
      if kwargs:
        x = model(x, **kwargs)
      else:
        x = model(x)

    is_tuple = isinstance(x, tuple)
    if labels is not None:
      assert not is_tuple, 'labels is not None does not work with is_tuple=True'
      labels = torch.cat(labels, dim=0) # batch
      x = [self.index_select_outputs(x, labels).transpose(-1, -2)]
    else: # separate the groups
      if is_tuple:
        channels = [int(x[i].shape[1]/groups) for i in range(len(x))]
        x = [torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]
        #x = list(zip(*[torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]))
        #x = [tuple([x_[:, start*channels[i]:(start+1)*channels[i]] for i, x_ in enumerate(x)]) for start in range(groups)]
      else:
        channels = int(x.shape[1]/groups)
        x = list(torch.split(x, channels, dim=1))
        #x = [x[:, start*channels:(start+1)*channels] for start in range(groups)]

    if is_tuple:
      channels = int(x[0][0].shape[0]/batches)
      x = tuple([[torch.split(x__, channels, dim=0) for x__ in x_] for x_ in x])
      #x = [[tuple([x__[start*channels:(start+1)*channels] for x__ in x_]) for start in range(batches)] for x_ in x]
    else:
      channels = int(x[0].shape[0]/batches)
      x = [list(torch.split(x_, channels, dim=0)) for x_ in x]
      #x = [[x_[start*channels:(start+1)*channels] for start in range(batches)] for x_ in x]
    return x


class Group(nn.Module):
  '''
  Group conv networks to run in parallel
  models: list of instantiated models
  groups: groups of inputs
  dim: if dim=0, use batch a set of inputs along batch dimension (group=1 always)
       elif dim=1, combine the channel dimension (group=num_inputs)

  Inputs:
    x: list of inputs
    labels: uses these labels to give a soft attention on the outputs. Use only with dim=1.
            if labels is None, return a list of outputs
    transpose: if true, model needs a transpose of the input
  '''
  def __init__(self, models, groups=1, dim=1):
    super().__init__()
    if not isinstance(models, list):
      models = [models]
    self.models = nn.ModuleList(models)
    self.groups = groups
    self.dim = dim

  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward(self, x, labels=None, transpose=True, **kwargs):
    if self.dim == 0:
      self.groups = len(x)
    if isinstance(x, list):
      x = torch.cat(x, dim=self.dim) ## concatenate along channels
    if transpose:
      x = x.transpose(-1, -2)
    for model in self.models:
      if kwargs:
        x = model(x, **kwargs)
      else:
        x = model(x)
    if labels is not None:
      x = self.index_select_outputs(x, labels).transpose(-1, -2) ## only for dim=1
      return x
    else:
      channels = int(x.shape[self.dim]/self.groups)
      dim = self.dim % len(x.shape)
      if dim == 2:
        x = [x[:, :, start*channels:(start+1)*channels] for start in range(self.groups)]
      elif dim == 1:
        x = [x[:, start*channels:(start+1)*channels] for start in range(self.groups)]
      elif dim == 0:
        x = [x[start*channels:(start+1)*channels] for start in range(self.groups)]
      return x

class EmbLin(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x, mode='lin'):
    if mode == 'lin':
      return x.matmul(self.emb.weight)
    elif mode == 'emb':
      return self.emb(x)


class Style(nn.Module):
  '''
  input_shape: (B, )
  output_shape: (B, )
  '''
  def __init__(self, num_speakers=1):
    self.style_emb = nn.Embedding(num_embeddings=num_speakers, embedding_dim=256)

  def forward(self, x):
    pass

class Curriculum():
  def __init__(self, start, end, num_iters):
    self.start = start
    self.end = end
    self.num_iters = num_iters
    self.iters = 0
    self.diff = (end-start)/num_iters
    self.value = start

  def step(self, flag=True):
    if flag:
      value_temp = self.value
      if self.iters < self.num_iters:
        self.value += self.diff
        self.iters += 1
        return value_temp
      else:
        return self.end
    else:
      return self.value