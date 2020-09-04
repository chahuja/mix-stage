'''
Preprocess pose features
python data/skeleton.py -path2data ../dataset/groot/data/speech2gesture_data -path2outdata ../dataset/groot/data -speaker "['all']" -preprocess_methods "'data'"

python data/skeleton.py -path2data ../dataset/groot/data -path2outdata ../dataset/groot/data -speaker "['all']" -preprocess_methods "'normalize'"
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argsUtils import *
from datetime import datetime

from common import Modality, MissingData

from pathlib import Path
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
from joblib import Parallel, delayed
import yaml
import warnings

from pycasper.pathUtils import replace_Nth_parent

## Load pose (with confidence intervals) from yml files (to work with data from https://github.com/amirbar/speech2gesture)
def loadYAML(filename):
  with open(filename) as f:
    lines = f.readlines()
  lines = lines[3:]
  data = yaml.load(''.join(lines), Loader=yaml.FullLoader)
  return np.array(data['data']).reshape(-1, 3)

def loadPose(filename):
  filebase = '_'.join(filename.split('_')[:-1])
  filepose = filebase + '_pose.yml'
  fileleft = filebase + '_hand_left.yml'
  fileright = filebase + '_hand_right.yml'

  #body = [1, 2, 3, 4, 5, 6, 7, 0, 15, 16]
  body = [0, 1, 2, 3, 4, 5, 6, 18, 19, 21]
  pose = loadYAML(filepose)[body]
  left = loadYAML(fileleft)[:21]
  right = loadYAML(fileright)[:21]

  return np.concatenate([pose, left, right])

class Skeleton2D(Modality):
  def __init__(self, path2data='../dataset/groot/data/speech2gesture_data',
               path2outdata='../dataset/groot/data',
               speaker='all',
               preprocess_methods=['data']):
    super(Skeleton2D, self).__init__(path2data=path2data)
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.speaker = speaker
    self.preprocess_methods = preprocess_methods
    
    self.missing = MissingData(self.path2outdata)

  def preprocess(self):
    if self.speaker[0] != 'all':
      speakers = self.speaker
    else:
      speakers = self.speakers

    for speaker in tqdm(speakers, desc='speakers', leave=False):
      tqdm.write('Speaker: {}'.format(speaker))
      df_speaker = self.get_df_subset('speaker', speaker)
      interval_ids = df_speaker['interval_id'].unique()
      interval_ids = np.array(list(set(interval_ids) - self.missing.load_intervals()))

      # for interval_id in tqdm(interval_ids, desc='intervals'):
      #   self.save_intervals(interval_id, speaker)
      # pdb.set_trace()

      missing_data_list = Parallel(n_jobs=-1)(delayed(self.save_intervals)(interval_id, speaker)
                                          for interval_id in tqdm(interval_ids))
      self.missing.save_intervals(missing_data_list)
      
  def save_intervals(self, interval_id, speaker):
    ## process keypoints for each interval
    if self.preprocess_methods == 'data':
      process_interval = self.process_interval
    elif self.preprocess_methods == 'normalize':
      process_interval = self.normalize
    elif self.preprocess_methods == 'confidence':
      process_interval = self.confidence
    else:
      raise 'preprocess_methods = {} not found'.format(self.preprocess_methods)
    
    keypoints = process_interval(interval_id)
    if keypoints is None:
      return interval_id

    ## save keypoints
    filename = Path(self.path2outdata)/'processed'/speaker/'{}.h5'.format(interval_id)
    key = self.add_key(self.h5_key, self.preprocess_methods)
    try:
      self.append(filename, key, keypoints)
    except:
#      pdb.set_trace()
      return interval_id
    return None

  def normalize(self, interval_id):
    ## get filename from interval_id
    speaker = self.get_df_subset('interval_id', interval_id).iloc[0].speaker
    filename = Path(self.path2outdata)/'processed'/speaker/'{}.h5'.format(interval_id)

    ## Reference shoulder length
    ref_len = 167
    
    ## load keypoints
    try:
      data, h5 = self.load(filename, 'pose/data')
      data = data[()]
      h5.close()
    except:
      warnings.warn('pose/data not found in filename {}'.format(filename))
      return None

    ## exception
    if len(data.shape) == 3:
      return None
    ## normalize
    ratio = ref_len/((data.reshape(data.shape[0], 2, -1)[..., 1]**2).sum(1)**0.5)
    keypoints = ratio.reshape(-1, 1) * data
    keypoints[:, [0, 52]] = data[:, [0, 52]]
    
    return keypoints

  def berk_confidence(self, interval_id):
    file_list = self.get_filelist(interval_id)
    if file_list is None:
      return None

    augment_filename = lambda x: replace_Nth_parent(x[:-4] + '_pose.yml', by='keypoints_all', N=2)
    file_list = [augment_filename(filename) for filename in file_list]
    keypoints_list = [loadPose(filename) for filename in file_list]

    try:
      keypoints = np.stack(keypoints_list, axis=0)
    except:
      warnings.warn('[BERK_CONFIDENCE] interval_id: {}'.format(interval_id))
      pdb.set_trace()
      return None
    keypoints = keypoints[..., -1]
    
    return np.concatenate([keypoints]*2, axis=1) ## (Time, Joints)

  def get_speaker(self, interval_id):
    return self.df[self.df['interval_id'] == interval_id].speaker.iloc[0]
  
  def cmu_confidence(self, interval_id):
    filename = Path(self.path2outdata)/'raw_keypoints'/self.get_speaker(interval_id)/'{}.h5'.format(interval_id)
    try:
      data, h5 = self.load(filename.as_posix(), 'pose/data')
      data = data[()]
      h5.close()
    except:
      warnings.warn('interval {} not found'.format(interval_id))
      h5.close()
          
    keypoints = data[:, -1, :]
    return np.concatenate([keypoints]*2, axis=1) ## (Time, Joints)

  def confidence(self, interval_id):
    if interval_id[0] == 'c':
      return self.cmu_confidence(interval_id)
    else:
      return self.berk_confidence(interval_id)
    
  def process_interval(self, interval_id):
    file_list = self.get_filelist(interval_id)
    if file_list is None:
      return None

    keypoints_list = [np.loadtxt(filename) for filename in file_list]

    keypoints = np.stack(keypoints_list, axis=0)
    keypoints = self.process_keypoints(keypoints)

    return keypoints

  def process_keypoints(self, keypoints, inv=False):
    if not inv:
      keypoints_new = keypoints - keypoints[..., self.root:self.root+1]
      keypoints_new[..., self.root] = keypoints[..., self.root]
      keypoints_new = keypoints_new.reshape(keypoints_new.shape[0], -1)
    else:
      keypoints = keypoints.reshape(keypoints.shape[0], 2, -1)
      keypoints_new = keypoints + keypoints[..., self.root:self.root+1]
      keypoints_new[..., self.root] = keypoints[..., self.root]
    return keypoints_new

  def get_filelist(self, interval_id):
    df = self.df[self.df['interval_id'] == interval_id]
    start_time = df['start_time'].values[0].split(' ')[-1][1:]
    end_time = df['end_time'].values[0].split(' ')[-1][1:]
    speaker = df['speaker'].values[0]
    video_fn = df['video_fn'].values[0].split('.')[0] ## the folder names end at the first period of the video_fn
    video_fn = Path('_'.join(video_fn.split(' '))) ## the folder names have `_` instead of ` `
    path2keypoints = '{}/{}/keypoints_simple/{}/'.format(self.path2data, speaker, video_fn)
    file_df = pd.DataFrame(data=os.listdir(path2keypoints), columns=['files_temp'])
    file_df['files'] = file_df['files_temp'].apply(lambda x: (Path(path2keypoints)/x).as_posix())
    file_df['start_time'] = file_df['files_temp'].apply(self.get_time_from_file)
    file_df = file_df.sort_values(by='start_time').reset_index()

    try:
      start_id = file_df[file_df['start_time'] == start_time].index[0]
      end_id = file_df[file_df['start_time'] == end_time].index[0]
    except:
      return None
    if not (self.are_keypoints_complete(file_df, start_id, end_id)):
      #self.missing.append_interval(interval_id)
      warnings.warn('interval_id: {} not found.'.format(interval_id))
      return None
    return file_df.iloc[start_id:end_id+1]['files'].values

  def are_keypoints_complete(self, file_df, start_id, end_id):
    # frames = (end_id + 1) - start_id
    # diff = (datetime.strptime(end_time, '%H:%M:%S.%f') - datetime.strptime(start_time, '%H:%M:%S.%f')).total_seconds()
    # diff_frames = (self.fs * diff) - frames
    flag = (((file_df.iloc[start_id+1:end_id+1].start_time.apply(pd.to_timedelta).reset_index() - file_df.iloc[start_id:end_id].start_time.apply(pd.to_timedelta).reset_index())['start_time'].apply(lambda x: x.total_seconds()) - 1/self.fs('pose/data')).apply(abs) > 0.00008).any()
    if flag:
      return False
    # if abs(diff_frames) >= 2:
    #   return False

    return True

  def get_time_from_file(self, x):
    x_cap = ':'.join('.'.join(x.split('.')[:-1]).split('_')[-3:]).split('.')
    if len(x_cap) == 1: ## sometimes the filnames do not have miliseconds as it is all zeros
      x_cap = '.'.join(x_cap + ['000000'])
    else:
      x_cap = '.'.join(x_cap)
    return x_cap

  @property
  def parents(self):
    return [-1,
            0, 1, 2,
            0, 4, 5,
            0, 7, 7,
            6,
            10, 11, 12, 13,
            10, 15, 16, 17,
            10, 19, 20, 21,
            10, 23, 24, 25,
            10, 27, 28, 29,
            3,
            31, 32, 33, 34,
            31, 36, 37, 38,
            31, 40, 41, 42,
            31, 44, 45, 46,
            31, 48, 49, 50]

  @property
  def joint_subset(self):
    ## choose only the relevant skeleton key-points (removed nose and eyes)
    return np.r_[range(7), range(10, len(self.parents))]

  @property
  def root(self):
    return 0

  @property
  def joint_names(self):
    return ['Neck',
            'RShoulder', 'RElbow', 'RWrist',
            'LShoulder', 'LElbow', 'LWrist',
            'Nose', 'REye', 'LEye',
            'LHandRoot',
            'LHandThumb1', 'LHandThumb2', 'LHandThumb3', 'LHandThumb4',
            'LHandIndex1', 'LHandIndex2', 'LHandIndex3', 'LHandIndex4',
            'LHandMiddle1', 'LHandMiddle2', 'LHandMiddle3', 'LHandMiddle4',
            'LHandRing1', 'LHandRing2', 'LHandRing3', 'LHandRing4',
            'LHandLittle1', 'LHandLittle2', 'LHandLittle3', 'LHandLittle4',
            'RHandRoot',
            'RHandThumb1', 'RHandThumb2', 'RHandThumb3', 'RHandThumb4',
            'RHandIndex1', 'RHandIndex2', 'RHandIndex3', 'RHandIndex4',
            'RHandMiddle1', 'RHandMiddle2', 'RHandMiddle3', 'RHandMiddle4',
            'RHandRing1', 'RHandRing2', 'RHandRing3', 'RHandRing4',
            'RHandLittle1', 'RHandLittle2', 'RHandLittle3', 'RHandLittle4'
    ]

  def fs(self, modality):
    return 15

  @property
  def h5_key(self):
    return 'pose'

def preprocess(args, exp_num):
  path2data = args.path2data #'../dataset/groot/speech2gesture_data/'
  path2outdata = args.path2outdata #'../dataset/groot/data'
  speaker = args.speaker
  preprocess_methods = args.preprocess_methods
  skel = Skeleton2D(path2data, path2outdata, speaker, preprocess_methods)
  skel.preprocess()

if __name__ == '__main__':
  argparseNloop(preprocess)
