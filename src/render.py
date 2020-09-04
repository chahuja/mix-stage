from model import *
from data import Data, ZNorm
from animation import save_animation
from parallel import parallel

from argsUtils import argparseNloop
from pycasper.BookKeeper import BookKeeper
from pycasper.argsUtils import *
from htmlUtils.toHTML import makeHTMLfile

import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import os
import pdb

def render(args, exp_num):
  assert args.load, 'Load file must be provided'
  assert os.path.exists(args.load), 'Load file must exist'
  
  args_dict_update={'render':args.render}
  args_dict_update.update(get_args_update_dict(args)) ## update all the input args

  args_subset = ['exp', 'cpk', 'speaker', 'model']
  book = BookKeeper(args, args_subset, args_dict_update=args_dict_update,
                    tensorboard=args.tb)
  args = book.args

  dir_name = book.name.dir(args.save_dir)
  
  ## Training parameters
  path2data = args.path2data
  speaker = args.speaker
  modalities = args.modalities
  split = args.split
  batch_size = args.batch_size
  shuffle = True if args.shuffle else False
  time = args.time
  fs_new = args.fs_new if isinstance(args.fs_new, list) else [args.fs_new] * len(modalities)
  mask = args.mask
  mask = list(np.concatenate([np.r_[i] if isinstance(i, int) else np.r_[eval(i)] for i in mask])) ## convert ranges to list of numbers
  
  # define input and output modalities TODO hadcoded
  input_modalities = modalities[1:] if args.input_modalities is None else args.input_modalities
  output_modalities = modalities[:1] if args.output_modalities is None else args.output_modalities
  input_modality = input_modalities[0]
  output_modality = output_modalities[0]
  
  ## Load data iterables
  data = Data(path2data, speaker, modalities, fs_new,
              time=time, split=split, batch_size=batch_size,
              shuffle=shuffle, load_data=False)

  #feats_shape = int(data.shape[output_modality][-1]/2)

  if isinstance(speaker, str):
    speaker = [speaker]

  keypoints_dirnames = []
  for filename in os.listdir(Path(dir_name)):
    if 'keypoints' in filename:
      keypoints_dirnames.append(filename)
      
  for desc in ['test', 'train']:
    for spk in speaker:
      for k_dirname in keypoints_dirnames:
        render_path = Path(dir_name)/k_dirname/desc/spk
        if not os.path.exists(render_path):
          continue
        gt_path = Path(args.path2data)/'processed'/spk
        y_animates = []
        y_animates_eval = [] ## for human evaluation
        intervals = []
        start = []
        texts = []

        filenames = sorted(os.listdir(render_path))
        np.random.seed(0) ## always choose the same random samples
        #sample_idxs = np.array(list(set(np.random.randint(0, len(filenames), size=(args.render,)))))
        sample_idxs = np.random.permutation(np.arange(len(filenames)))[:args.render]
        sample_from_list = lambda x, idxs: [x[idx] for idx in idxs]
        filenames = sample_from_list(filenames, sample_idxs)

        for filename in tqdm(filenames, desc=desc):
          y, h5 = data.load((render_path/filename).as_posix(), output_modality)
          y = y[()]
          feats_shape = y.shape[-1]
          #pdb.set_trace()
          #y = y.reshape(-1, 2, feats_shape)
          y[..., 0] = 0 
          h5.close()

          y_gt, h5 = data.load((gt_path/filename).as_posix(), output_modality)
          y_gt = y_gt[()]
          y_gt = y_gt.reshape(-1, 2, feats_shape)
          y_gt[..., 0] = 0 
          h5.close()

          if args.render_text:
            try:
              text = pd.read_hdf((gt_path/filename).as_posix(), key='text/meta')
            except:
              text = None
          else:
            text = None

          y_animates.append([y, y_gt])
          y_animates_eval.append([y])
          intervals.append(Path(filename).stem)
          start.append(0)
          texts.append(text)

        #sample_idxs = np.random.randint(0, len(y_animates), size=(args.render,))
        #sample_from_list = lambda x, idxs: [x[idx] for idx in idxs]
        #y_animates = sample_from_list(y_animates, sample_idxs)
        #intervals = sample_from_list(intervals, sample_idxs)
        #start = sample_from_list(start, sample_idxs)

        ## remove the masked values from the renders
        subname1 = None if len(k_dirname.split('_')) == 1 else '_'.join(k_dirname.split('_')[1:])
        subname2 = 'eval' if len(k_dirname.split('_')) == 1 else 'eval_{}'.format('_'.join(k_dirname.split('_')[1:]))

        save_animation(y_animates, intervals, dir_name, desc, data, start, subname=subname1, text=texts, output_modalities=output_modality, mask=mask)
        save_animation(y_animates_eval, intervals, dir_name, desc, data, start, subname=subname2, text=texts, output_modalities=output_modality, mask=mask)


  ## render html
  if set(keypoints_dirnames) - {'keypoints', 'keypoints_style'}: 
    makeHTMLfile(dir_name, args.render, 'videos')
    makeHTMLfile(dir_name, 4, 'videos_subset')
        
if __name__ == '__main__':
  argparseNloop(render)
