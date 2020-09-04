import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *

import torch
import torch.nn as nn

class Speech2Gesture_G(nn.Module):
  '''
  Baseline: http://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/

  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0):
    super(Speech2Gesture_G, self).__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)
                                  for i in range(4)]))
    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

  def forward(self, x, y, time_steps=None, **kwargs):
    if x.dim() == 3:
      x = x.unsqueeze(dim=1)
    x = self.audio_encoder(x, time_steps)
    x = self.unet(x)
    x = self.decoder(x)
    x = self.logits(x)

    internal_losses = []
    return x.transpose(-1, -2), internal_losses

class Speech2Gesture_D(nn.Module):
  '''
  Baseline: http://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/

  input_shape:  (N, time, pose_feats)
  output_shape: (N, *, 1) ## discriminator scores
  '''
  def __init__(self, in_channels=104, out_channels=64,  n_downsampling=2, p=0, groups=1, **kwargs):
    super(Speech2Gesture_D, self).__init__()
    self.conv1 = nn.Sequential(torch.nn.Conv1d(in_channels*groups, out_channels*groups, 4, 2, padding=1, groups=groups),
                               torch.nn.LeakyReLU(negative_slope=0.2))
    self.conv2 = nn.ModuleList([])
    for n in range(1, n_downsampling):
      ch_mul = min(2**n, 8)
      self.conv2.append(ConvNormRelu(out_channels, out_channels*ch_mul,
                                     type='1d', downsample=True, leaky=True, p=p, groups=groups))

    self.conv2 = nn.Sequential(*self.conv2)
    ch_mul_new = min(2**n_downsampling, 8)
    self.conv3 = ConvNormRelu(out_channels*ch_mul, out_channels*ch_mul_new,
                              type='1d', leaky=True, kernel_size=4, stride=1, p=p, groups=groups)

    out_shape = 1 if 'out_shape' not in kwargs else kwargs['out_shape']
    self.logits = nn.Conv1d(out_channels*ch_mul_new*groups, out_shape*groups, kernel_size=4, stride=1, groups=groups)

  def forward(self, x):
    x = x.transpose(-1, -2)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.logits(x)

    internal_losses = []
    return x.transpose(-1, -2).squeeze(dim=-1), internal_losses

