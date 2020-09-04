import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import math

from pycasper.torchUtils import LambdaScheduler
from tqdm import tqdm

import pdb
  
class GAN(nn.Module):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True,
               **kwargs):
    super(GAN, self).__init__()
    self.G = G
    self.D = D
    self.D_prob = dg_iter_ratio/(dg_iter_ratio + 1) ## discriminator generator iteration ratio
    self.lambda_D = lambda_D
    self.lambda_gan = lambda_gan
    self.lambda_scheduler = LambdaScheduler([self.lambda_D, self.lambda_gan],
                                            kind='incremental', max_interval=300,

                                            max_lambda=2)
    self.G_flag = True
    self.lr = lr

#    self.G_optim = eval('torch.optim.' + optim)(self.G.parameters(), lr=self.lr)
#    self.D_optim = eval('torch.optim.' + optim)(self.D.parameters(), lr=self.lr)

    self.criterion = eval('torch.nn.' + criterion)(reduction='none')

    self.joint = joint
    self.input_modalities = kwargs['input_modalities']
    self.update_D_prob_flag = update_D_prob_flag
    self.no_grad = no_grad

  def get_velocity(self, x, x_audio):
    x_v = x[..., 1:, :] - x[..., :-1, :]
    if self.joint:
      return torch.cat([torch.cat([torch.zeros_like(x[..., 0:1, :]), x_v], dim=-2), torch.cat(x_audio[:len(self.input_modalities)], dim=-1)], dim=-1)
    else:
      return torch.cat([torch.zeros_like(x[..., 0:1, :]), x_v], dim=-2)

  def get_velocity_archive(self, x, x_audio):
    x_v = x[..., 1:] - x[..., :-1]
    return torch.cat([torch.zeros_like(x[..., 0:1]), x_v], dim=-1)

  def get_real_gt(self, x):
    return torch.ones_like(x)
  
  def get_fake_gt(self, x):
    return torch.zeros_like(x)

  def sample_wise_weight_mean(self, loss, W):
    W = W.view([W.shape[0]] + [1]*(len(loss.shape) - 1))
    loss = (W.expand_as(loss) * loss).mean()
    return loss
  
  def get_gan_loss(self, y_cap, y, W):
    loss = self.criterion(y_cap, y)
    return self.sample_wise_weight_mean(loss, W)

  def get_loss(self, y_cap, y, W):
    loss = self.criterion(y_cap, y)
    return self.sample_wise_weight_mean(loss, W)

  def estimate_weights(self, x_audio, y_pose, **kwargs):
    return torch.ones(y_pose.shape[0]).to(y_pose.device), None

  def estimate_weights_loss(self, W):
    return W

  def update_D_prob(self, W):
    pass
    
  def forward(self, x_audio, y_pose, **kwargs):
    internal_losses = []

    ## get confidence values
    if 'confidence' in kwargs:
      confidence = kwargs['confidence']
    else:
      confidence = 1

    W, outputs = self.estimate_weights(x_audio, y_pose, **kwargs)
    W_loss = self.estimate_weights_loss(W)
    if self.update_D_prob_flag:
      self.update_D_prob(W)
    
    if self.training:
    #if True:
      ## update lambdas
      self.lambda_D, self.lambda_gan = self.lambda_scheduler.step()

      if torch.rand(1).item() < self.D_prob: ## run discriminator
        self.G.eval() ## generator must be in eval mode to activate eval mode of bn and dropout
        with torch.no_grad():
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
        self.G.train(self.training) ## take back generator to it's parent's mode
        real_pose = y_pose

        ## convert pose to velocity
        real_pose_v = self.get_velocity(real_pose, x_audio)
        fake_pose_v = self.get_velocity(fake_pose, x_audio)

        self.fake_flag = True
        #if torch.rand(1).item() < 0.5:
        if True:
          fake_pose_score, _ = self.D(fake_pose_v.detach())
          fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones_like(1/W_loss))
          self.fake_flag = True
        else:
          fake_D_loss = torch.zeros(1)[0].to(fake_pose_v.device)
          self.fake_flag = False
        real_pose_score, _ = self.D(real_pose_v)
        real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones_like(W_loss))

        internal_losses.append(real_D_loss)
        internal_losses.append(fake_D_loss)
        internal_losses += partial_i_loss
        self.G_flag = False

      else:
        fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
        args = args[0] if len(args)>0 else {}
        ## convert pose to velocity
        fake_pose_v = self.get_velocity(fake_pose, x_audio)
        if self.no_grad:
          with torch.no_grad():
            fake_pose_score, _ = self.D(fake_pose_v)
        else:
          fake_pose_score, _ = self.D(fake_pose_v)

        G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), 1/W_loss)
          
        pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss)

        internal_losses.append(pose_loss)
        internal_losses.append(G_gan_loss)
        internal_losses += partial_i_loss
        self.G_flag = True
    else:
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones_like(W_loss))

      internal_losses.append(pose_loss)
      internal_losses.append(torch.tensor(0))
      internal_losses += partial_i_loss
      self.G_flag = True

    args.update(dict(W=W))
    return fake_pose, internal_losses, args
