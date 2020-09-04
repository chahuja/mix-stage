import sys
sys.path.insert(0, '..')

## Layers for different models
from .layers import *

## Models
from .speech2gesture import *
from .joint_late_cluster_soft_style import *
from .style_classifier import *

## GAN model which is a combination of Generator and Discriminator from Models
from .gan import *

## Trainers - BaseTrainer, Trainer, GANTrainer
from .trainer import *

'''
speech2gesture baselines
  - non-gan
  - gans

(snl2pose) - speech and language
  - non-gan
  - gans
'''
