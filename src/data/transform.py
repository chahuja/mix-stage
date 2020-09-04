import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle as pkl
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import sklearn.cluster
import sklearn.mixture
import numpy as np
import pdb

from dataUtils import DummyData
from text import POStagging
from common import HDF5
from skeleton import Skeleton2D

import torch
from torch.utils.data import Dataset, DataLoader
from pycasper import torchUtils


class TransformDict():
  '''
  Convert a Transform Class which accept dictionaries as inputs

  Args:
    transform: ``Transform`` object

  Example:
  >>> TranslateDict = TransformDict(Translate(1))
  >>> print(TranslateDict({'tensor': torch.zeros(3)}))
  >>> {'tensor': Tensor([1., 1., 1.])}
  '''
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, batch, **kwargs):
    batch_new = {}
    for variable in batch:
      batch_new[variable] = self.transform(batch[variable], **kwargs)
    return batch_new

  def __repr__(self):
    format_string = self.__class__.__name__ + '({})'.format(self.transform)
    return format_string

class Compose():
  """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
      >>> Compose([
      >>>     ZNorm(['pose/data'], key='oliver'),
      >>>     TransformDict(Translate(10))
      >>> ])
  """

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img, inv=False, **kwargs):
    transforms = self.transforms if not inv else self.transforms[::-1]

    for t in transforms:
      if isinstance(t, TransformDict):
        if t.transform.__call__.__code__.co_argcount == 2:
          img = t(img)
        elif t.transform.__call__.__code__.co_argcount == 3 and kwargs:
          img = t(img, inv=inv, **kwargs)
        elif t.transform.__call__.__code__.co_argcount == 3:
          img = t(img, inv=inv)
        else:
          assert 0, 'all transforms must have only one or two arguments'
      else:
        if t.__call__.__code__.co_argcount == 2:
          img = t(img)
        elif t.__call__.__code__.co_argcount == 3 and kwargs:
          img = t(img, inv=inv, **kwargs)
        elif t.__call__.__code__.co_argcount == 3:
          img = t(img, inv=inv)
        else:
          assert 0, 'all transforms must have only one or two arguments'
    return img

  def __repr__(self):
    format_string = self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += '    {0}'.format(t)
    format_string += '\n)'
    return format_string


class ZNorm():
  '''
  Class to calculate ZNorm on the complete data

  Arguments:
    variable_list (list of str, optional):
    savepath (str): path to the folder where the mean and variances are stored
    key (str): name of the file
    data (DataLoader): a data loader which iterates through all of the data
    num_dims (int, optional): number of dimensions from the left along
        which mean/var is calculated. For example, if ``2``, an input
        of shape ``(10, 20, 30, 40)`` would be reduced to a mean
        of shape ``(1, 1, 30, 40)``. (default: ``2``)

  Example::
    from data import ZNorm
    from torch.utils.data import DataLoader
    from data import DummyData

    variable_list = ['pose', 'audio']
    data = DataLoader(DummyData(variable_list=variable_list, random=True), batch_size=1000)
    pre = ZNorm(['audio', 'pose'], savepath='./preprocessing_temp', data=data)
    for batch in data:
      break
    mean = pre(batch)['pose'].mean()
    std = pre(batch)['pose'].std()
    print('Mean: {}, std: {} after Ztransform'.format(mean, std))
  '''
  def __init__(self, variable_list=[], savepath='./preprocessing/muvar', key='key', data=None, num_dims=2, verbose=True, relative2parent=0, pre=None):
    os.makedirs(savepath, exist_ok=True)
    self.variable_list = variable_list
    self.savepath = savepath
    self.key = '_'.join(key) if isinstance(key, list) else key
    self.data = data
    self.relative2parent = relative2parent
    self.pre = pre ## use a pre on the data before calculating mean var
    self.hdf5 = HDF5()

    self.variable_dict = {}


    for variable in variable_list:
      if relative2parent:
        path2file = Path(savepath)/'{}_relative2parent.h5'.format(self.key)
      else:
        path2file = Path(savepath)/(self.key+'.h5')
      if path2file.exists():
        muvar = self.loadfile(path2file, variable)
        if muvar is None:
          muvar = self.cal_muvar(path2file, variable, num_dims=num_dims)
          if verbose:
            print('Calculating Mean-Variance for {}'.format(variable))
        else:
          if verbose:
            print('Loading Mean-Variance for {}'.format(variable))
      else:
        muvar = self.cal_muvar(path2file, variable, num_dims=num_dims)
        if verbose:
          print('Calculating Mean-Variance for {}'.format(variable))
      self.variable_dict[variable] = muvar

  def loadfile(self, path2file, variable):
    if not self.hdf5.isDatasetInFile(path2file, variable):
      return None
    mu = torch.from_numpy(
      self.hdf5.load(
        path2file,
        self.hdf5.add_key(variable, ['mean'])
      )[0][()]
    ).to(torch.double)
    var = torch.from_numpy(
      self.hdf5.load(
        path2file,
        self.hdf5.add_key(variable, ['var'])
      )[0][()]
    ).to(torch.double)
    return [mu, var]

  def savefile(self, obj, path2file, variable):
    self.hdf5.append(path2file,
                     self.hdf5.add_key(variable, ['mean']),
                     obj[0])
    self.hdf5.append(path2file,
                     self.hdf5.add_key(variable, ['var']),
                     obj[1])

  '''
  Calculate the mean and variance of the dataset

  Arguments:
    path2file (str): path to the file where the mean and variances are stored
    variable (str): variable for which the mean and variance has to be calculated
    num_dims (int, optional): number of dimensions from the left along
        which mean/var is calculated. For example, if ``2``, an input
        of shape ``(10, 20, 30, 40)`` would be reduced to a mean
        of shape ``(1, 1, 30, 40)``. (default: ``2``)
  '''
  def cal_muvar(self, path2file, variable, num_dims=2):
    mean = 0
    energy = 0
    count = 0
    collate_fn = self.data.collate_fn
    ## sample thorugh the complete dataset irrespective of the sampler
    dataloader = torch.utils.data.DataLoader(self.data.dataset, batch_size=32, collate_fn=collate_fn)
    #self.data.sampler = torch.utils.data.SequentialSampler(self.data.dataset)
    for batch in tqdm(dataloader, desc='mu+var for {}'.format(variable)):
      if self.pre is not None:
        batch = self.pre(batch)[variable]
      else:
        batch = batch[variable]
      dims = list(range(num_dims))
      mean += batch.sum(dim=dims, keepdim=True)
      energy += (batch**2).sum(dim=dims, keepdim=True)
      count += np.prod(np.array(batch.shape)[dims])
    mean = mean/count
    energy = energy/count
    var = energy - mean**2

    muvar = [mean, var]
    self.savefile(muvar, path2file, variable)
    return muvar

  def znorm(self, x, muvar, eps=1e-8):
    mask_std = (muvar[1] >= 0).to(torch.double)
    std = (muvar[1]*mask_std)**0.5
    mask = (std == 0).to(torch.double)
    std = (mask * eps) + (1-mask)*std
    return (x - muvar[0])/std

  def inv_znorm(self, x, muvar):
    return x*(muvar[1]**0.5) + muvar[0]

  def __call__(self, batch, inv=False):
    batch_new = {}
    for variable in batch:
      if variable in self.variable_dict:
        if not inv:
          batch_new[variable] = self.znorm(batch[variable], self.variable_dict[variable])
        else:
          batch_new[variable] = self.inv_znorm(batch[variable], self.variable_dict[variable])
      else:
        batch_new[variable] = batch[variable]
    return batch_new

  def __repr__(self):
    return self.__class__.__name__ + '(variable_list={}, key={})'.format(self.variable_list, self.key)


class KMeans():
  '''
  ### TODO: Does not work with Compose at the moment
  Class to calculate ZNorm on the complete data

  Arguments:
    variable_list (list of str, optional):
    savepath (str): path to the folder where the mean and variances are stored
    key (str): name of the file
    data (DataLoader): a data loader which iterates through all of the data
    num_dims (int, optional): number of dimensions from the left along
        which mean/var is calculated. For example, if ``2``, an input
        of shape ``(10, 20, 30, 40)`` would be reduced to a mean
        of shape ``(1, 1, 30, 40)``. (default: ``2``)

  Example::
    from data import KMeans
    from torch.utils.data import DataLoader
    from data import DummyData

    variable_list = ['pose', 'audio']
    device = 'cuda'
    data = DataLoader(DummyData(variable_list=variable_list, random=True), batch_size=1000)
    kmeans = KMeans(savepath='./preprocessing_temp', data=data, device=device)
    for batch in data:
      break
    predictions = kmeans(batch['pose/data'])
  '''
  def __init__(self, variable_list = [], savepath='./preprocessing/kmeans', key='key', data=None, num_clusters=8, mask=[0, 7, 8, 9], feats=['pose', 'velocity'], verbose=True):
    os.makedirs(savepath, exist_ok=True)
    self.variable_list = variable_list
    self.variable = variable_list[0]
    self.savepath = savepath
    self.key = '_'.join(key) if isinstance(key, list) else key
    self.data = data
    self.num_clusters = num_clusters
    self.mask = mask
    self.remove_joints = RemoveJoints(self.mask)
    self.feats = feats

    # self.muvarpath = './preprocessing/muvar'
    # self.hdf5 = HDF5()
    # self.variable_dict = {}

    # for v in variable_list:
    #   path2file = Path(self.muvarpath)/(self.key+'.h5')
    #   if path2file.exists():
    #     muvar = ZNorm.loadfile(self, path2file, v)
    #     if muvar is None:
    #       muvar = ZNorm.cal_muvar(path2file, v, num_dims=num_dims)
    #       if verbose:
    #         print('Calculating Mean-Variance for {}'.format(v))
    #     else:
    #       if verbose:
    #         print('Loading Mean-Variance for {}'.format(v))
    #   else:
    #     muvar = ZNorm.cal_muvar(path2file, v, num_dims=num_dims)
    #     if verbose:
    #       print('Calculating Mean-Variance for {}'.format(v))
    #   self.variable_dict[v] = muvar

    pre = ZNorm(variable_list, key=key, data=self.data, verbose=False)
    self.variable_dict = pre.variable_dict
    for var in variable_list:
      if var in ['pose/data', 'pose/normalize']:
        self.output_modality = var
        break
      else:
        raise 'pose variable not found in variable_list'


    self.hdf5 = HDF5()

    path2file = Path(savepath)/(self.key+'.h5')
    self.centers = None
    key_name = 'centers/{}'.format(self.num_clusters) + '_{}'*len(self.feats)
    key_name = key_name.format(*self.feats)
    key_name += '_{}'*len(self.mask)
    key_name = key_name.format(*self.mask)
    key_name += '_{}'
    key_name = key_name.format('_'.join(self.variable.split('/')))

    if path2file.exists():
      if self.hdf5.isDatasetInFile(path2file, key_name):
        self.centers, h5 = self.hdf5.load(path2file, key_name)
        self.centers = self.centers[()]
        h5.close()
        if verbose:
          print('Loading KMeans model for {}/{}'.format(key, key_name))
      else:
        if verbose:
          print('Calculating KMeans model for {}/{}'.format(key, key_name))
        with torch.no_grad():
          self.centers = self.get_kmeans()
        self.hdf5.append(path2file, key_name, self.centers)
    else:
      if verbose:
        print('Calculating KMeans model for {}/{}'.format(key, key_name))
      with torch.no_grad():
        self.centers = self.get_kmeans()
      self.hdf5.append(path2file, key_name, self.centers)

    self.centers = torch.from_numpy(self.centers)


  def get_feats(self, x):
    pose_list = []
    for feat in self.feats:
      if feat == 'pose':
        pose_list.append(x)
      if feat == 'velocity':
        pose_v = torch.zeros_like(x)
        pose_v[:, 1:, :] = x[:, 1:] - x[:, :-1]
        pose_list.append(pose_v)
      if feat == 'speed':
        pose_s = torch.zeros_like(x)
        pose_s[:, 1:, :] = x[:, 1:] - x[:, :-1]
        pose_s = pose_s.reshape(pose_s.shape[0], pose_s.shape[1], 2, -1)
        pose_s = (pose_s**2).sum(dim=-2) ** 0.5 ## calculating speed from velocity
        pose_list.append(pose_s)
      if feat == 'acceleration':
        pose_v = torch.zeros_like(x)
        pose_v[:, 1:, :] = x[:, 1:] - x[:, :-1]
        pose_a = torch.zeros_like(x)
        pose_a[:, 1:, :] = pose_v[:, 1:] - pose_v[:, :-1]
        #pose_a = (pose_a**2).sum(dim=-2) ** 0.5 ## calculating speed from velocity
        pose_list.append(pose_a)
      if feat == 'spatial':
        mean = self.variable_dict[self.output_modality][0][:,:,8:]
        pose_se = torch.zeros_like(x)
        pose_se = x - mean # just the elbow joints?
        pose_list.append(pose_se)
    return torch.cat(pose_list, dim=-1)

  def get_kmeans(self):
    model = sklearn.cluster.MiniBatchKMeans(n_clusters=self.num_clusters)
    collate_fn = self.data.collate_fn
    ## sample thorugh the complete dataset irrespective of the sampler
    dataloader = torch.utils.data.DataLoader(self.data.dataset, batch_size=32, collate_fn=collate_fn)
    for batch in tqdm(dataloader):
      pose = batch[self.variable]
      pose = self.remove_joints(pose)
      pose = self.get_feats(pose)
      pose = pose.view(-1, pose.shape[-1])
      model.partial_fit(pose)
    centers = model.cluster_centers_
    return centers

  def predict(self, x, **kwargs):
    x = x.double()
    x = self.get_feats(x)
    x_shape = list(x.shape)
    x = x.view(-1, 1, x_shape[-1])
    centers_shape = [1] + list(self.centers.shape)

    mse = ((self.centers.view(*centers_shape).to(x.device) - x)**2).sum(dim=-1)
    if kwargs:
      if kwargs['soft_labels']:
        labels = torch.nn.functional.softmax(-mse/mse.mean(-1).unsqueeze(-1), dim=-1).view(x_shape[:-1] + [centers_shape[1]])
      else:
        labels = mse.min(dim=-1)[1].view(x_shape[:-1])
    else:
      labels = mse.min(dim=-1)[1].view(x_shape[:-1])
    return labels

  def inv_predict(self, y, **kwargs):
    y_shape = list(y.shape) + [self.centers.shape[-1]]
    y = y.view(-1)
    return self.centers.to(y.device)[y].view(*y_shape)

  def update(self, batch):
    pass

  def __call__(self, batch, inv=False, **kwargs):
    if not inv:
      return self.predict(batch, **kwargs)
    else:
      return self.inv_predict(batch, **kwargs)

  def __repr__(self):
    return self.__class__.__name__ + '(variable_list={}, key={})'.format(self.variable, self.key)

class Relative2Parent():
  def __init__(self, parents=None):
    if parents is None:
      self.parents = Skeleton2D().parents
    else:
      self.parents = parents

  def inv(self, pose):
    for i, parent in enumerate(self.parents[1:]):
      pose[..., i+1] += pose[..., parent]
    return pose

  def __call__(self, batch, inv=False):
    batch_new = {}
    for key in batch:
      if 'pose' in key:
        pose = batch[key].clone()
        pose = pose.view(pose.shape[0], pose.shape[1], 2, -1)
        pose[..., 0] = 0
        if inv:
          pose = self.inv(pose)
        else:
          pose[..., 1:] = pose[..., 1:] - pose[..., self.parents[1:]]
        pose[..., 0] = batch[key].view(pose.shape[0], pose.shape[1], 2, -1)[..., 0]
        pose = pose.view(pose.shape[0], pose.shape[1], -1)
        batch_new[key] = pose
      else:
        batch_new[key] = batch[key]

    return batch_new

  def __repr__(self):
    return self.__class__.__name__ + '()'

class RemoveJoints():
  def __init__(self, mask, parents=None):
    self.mask = mask
    self.parents = parents
    #self.children = self.get_children()
    self.insert = None

  def get_children(self):
    if self.parents is None:
      return None
    children = {}
    for i, parent in enumerate(self.parents):
      if parent in children:
        children[parent].append(i)
      else:
        children[parent] = [i]
    return children

  def __call__(self, batch, inv=False, **kwargs):
    if inv:
      assert self.insert is not None, 'Call Remove Joints first before calling the inverse version'
      batch_cap = torchUtils.add_slices(batch.view(batch.shape[0], batch.shape[1], 2, -1),
                                        insert=self.insert,
                                        mask=self.mask,
                                        dim=-1)
      if self.parents is not None and 'batch_gt' in kwargs:
        ## Bring masked children close enough to the predicted parents for a better visualization.
        batch_gt = kwargs['batch_gt']
        batch_gt = batch_gt.view(batch_gt.shape[0], batch_gt.shape[1], 2, -1)
        for i in self.mask: ## must be in topological order
          if i != 0: ## ignore first joint
            j = self.parents[i]
            batch_cap[..., i] = (batch_gt[..., i] - batch_gt[..., j]) + batch_cap[..., j]

    else:
      batch = batch.view(batch.shape[0], batch.shape[1], 2, -1)
      batch, insert = torchUtils.remove_slices(batch, mask=self.mask, dim=-1)
      ## use save_insert=False to not save insert to self.insert
      if kwargs.get('save_insert') is None or kwargs.get('save_insert') is True:
        self.insert = insert
        self.insert = self.insert.to('cpu')

      batch_cap = batch

    return batch_cap.view(batch_cap.shape[0], batch_cap.shape[1], -1)

  def __repr__(self):
    return self.__class__.__name__ + '(mask={})'.format(self.mask)


if __name__ == '__main__':
  variable_list = ['pose', 'audio']
  data = DataLoader(DummyData(variable_list=variable_list), batch_size=1000)
  pre = ZNorm(['pose'], savepath='./preprocessing_temp', data=data)
  for batch in data:
    break
  mean = pre(batch)['pose'].mean()
  std = pre(batch)['pose'].std()
  print('Mean: {}, std: {} after Ztransform'.format(mean, std))
