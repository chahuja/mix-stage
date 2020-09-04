import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import subprocess
import pdb
import numpy as np
import pandas as pd
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.collections import LineCollection

from argsUtils import *
from data import Data
from parallel import *

import torch
from torch.utils.data import DataLoader

def split_text(text, max_length=27):
  text_subs = []
  start = 0
  cum_length = 0
  for end, row in text.iterrows():
    if len(row['Word']) + 1 + cum_length > max_length:
      text_subs.append(text.iloc[start:end].reset_index(drop=True))
      start = end
      cum_length = len(row['Word'])
    else:
      cum_length += len(row['Word']) + 1

  if cum_length > 0:
    text_subs.append(text.iloc[start:])
  return text_subs

def remove_labels(labels):
  for label in labels:
    label.remove()
  for i in range(len(labels)):
    labels.pop()

def update_labels(strings, labels, new, new_labels=False, x0=-500, fontsize=10):
  if new_labels:
    for i, (label, string) in enumerate(zip(labels, strings)):
      label.set_text(string)
      label.set_c('k')
      label.set_fontfamily('monospace')
      label.set_fontsize(fontsize)
      if i > 0:
        xpos = (len(''.join(strings[:i])) + i)/27 + x0
      else:
        xpos = x0
      label.set_position([xpos, -0.08])

    update_current_word(labels, 0, 0)
  else:
    if new == 0:
      old = 0
    else:
      old = new - 1
    try:
      update_current_word(labels, old, new)
    except:
      print('OLD:{}/{}'.format(old, new))

def update_current_word(labels, old, new):
  labels[old].set_c('k')
  labels[old].set_fontweight('normal')
  labels[new].set_c('tab:red')
  labels[new].set_fontweight('bold')

def get_line_segments(x, y, num_segments=20):
  x0, x1 = x
  y0, y1 = y

  def get_ranges(a, b):
    eps = (b-a)/num_segments
    if eps == 0:
      return np.repeat(a, repeats=num_segments)
    else:
      return np.arange(a, b+eps, eps)
  X = get_ranges(x0, x1)
  Y = get_ranges(y0, y1)

  min_len = min(X.shape[0], Y.shape[0])
  try:
    segments = np.stack([X[:min_len], Y[:min_len]], axis=-1)
  except:
    pdb.set_trace()
  segments = np.stack([segments[:-1,], segments[1:]], axis=1)
  return segments
  
def animate(ys, interval_id, parents, dir_name, desc, data, start, end, idx, subname, text=None, fps=15, bitrate=1000):
  #pdb.set_trace()
  if not isinstance(ys, list):
    ys = [ys]
  render_dir_name = 'render_{}'.format(subname) if subname is not None else 'render'
  if idx is None:
    filename = (Path(dir_name)/'{}/{}/{}/{}.mp4'.format(render_dir_name,
                                                        desc,
                                                        data.getSpeaker(interval_id),
                                                        interval_id))
    filename_temp = (Path(dir_name)/'{}/{}/{}/{}_temp.mp4'.format(render_dir_name,
                                                                  desc,
                                                                  data.getSpeaker(interval_id),
                                                                  interval_id))
  else:
    filename = (Path(dir_name)/'{}/{}/{}/{}_{:03d}.mp4'.format(render_dir_name,
                                                               desc,
                                                               data.getSpeaker(interval_id),
                                                               interval_id, idx))
    filename_temp = (Path(dir_name)/'{}/{}/{}/{}_{:03d}_temp.mp4'.format(render_dir_name,
                                                                         desc,
                                                                         data.getSpeaker(interval_id),
                                                                         interval_id, idx))
  os.makedirs(filename.parent, exist_ok=True)
  filename = filename.as_posix()

  ## init line collections
  plt.ioff()
  fig = plt.figure()
  fig.patch.set_alpha(0.) ## set backgroud transparent
  fig.patch.set_visible(False)

  C = len(ys)
  axs = [fig.add_subplot(1, C, c+1) for c in range(C)]
  # for ax in axs:
  #   ax.patch.set_visible(False)
  #   ax.xaxis.set_visible(False)
  #   ax.yaxis.set_visible(False)
  #   ax.set_frame_on(False)
  #lns = [[ax.plot([], [])[0] for _ in parents[1:]] for ax in axs]

  ## calculate the line widths for each segment
  num_segments, min_width, max_width, max_width2 = 100, 1, 3, 1
  def widths(x, y, segs, eps_flag=1):
    eps = (y-x)/segs
    if eps == 0:
      return np.repeat(x, repeats=segs)
    return np.arange(x, y+eps*int(eps_flag), eps)
  lwidths1 = np.concatenate([widths(min_width, max_width, num_segments/2, eps_flag=0), widths(max_width, min_width, num_segments/2, eps_flag=1)])
  lwidths2 = np.concatenate([widths(min_width, max_width2, num_segments/2, eps_flag=0), widths(max_width2, min_width, num_segments/2, eps_flag=1)])

  ## get color list
  colors = [axs[0].plot([], [])[0].get_color() for _ in parents[1:]]
  def get_line_collections():
    lns_list = []
    for i, (_,color) in enumerate(zip(parents[1:], colors)):
      if i <= 8:
        lwidths = lwidths1
      else:
        lwidths = lwidths2
        
      lns_list.append(LineCollection([], linewidths=lwidths, alpha=1, color=color))
    return lns_list
  lns = [get_line_collections() for ax in axs]
  
  ## add line collections to the correct subplot
  for ax, ln in zip(axs, lns):
    for l in ln:
      ax.add_collection(l)
      
  labels = []
  end_frame = [0]
  if C == 1:
    fontsize = 13
  else:
    fontsize = 10
  
  ## text inits
  if text is not None:
    text_subs = split_text(text)
    idx = [0]
    x0 = 0
    end_frame = [text_subs[idx[-1]].iloc[-1].end_frame]
    strings = text_subs[idx[-1]].Word.values
    labels = [axs[0].text(x0, 0, '', transform=axs[0].transAxes) for i in range(len(strings))] ## create new labels

  def init():
    for ax in axs:
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlim(-500, 500)
      ax.set_ylim(-500, 500)
      ax.set_aspect(aspect=1)

  def update(frame):
    #global labels, end_frame, idx
    for i, y in enumerate(ys):
      #segments_list = []
      for joint, parent in enumerate(parents):
        if joint != 0:
          segments = get_line_segments(x=[y[frame, 0, parent], y[frame, 0, joint]],
                                       y=[-y[frame, 1, parent], -y[frame, 1, joint]],
                                       num_segments=num_segments)
          lns[i][joint-1].set_segments(segments)
            # lns[i][joint-1].set_data([y[frame, 0, parent], y[frame, 0, joint]],
            #                          [-y[frame, 1, parent], -y[frame, 1, joint]])

            #print('NOT SAVED {}'.format(filename))

      #lns[i].set_segments(np.concatenate(segments_list, axis=0))

    ## update text
    if text is not None:
      if frame >= end_frame[-1]:
        #pdb.set_trace()
        idx.append(idx[-1]+1)
        end_frame.append(text_subs[idx[-1]].iloc[-1].end_frame)
        new_labels = True
        remove_labels(labels) ## remove old labels
        strings = text_subs[idx[-1]].Word.values
        for i in range(len(strings)):
          labels.append(axs[0].text(x0, 0, '', transform=axs[0].transAxes))
        #labels = [axs[0].text(x0, -580, '') for i in range(len(strings))] ## create new labels
      else:
        new_labels = False
        strings = text_subs[idx[-1]].Word.values
      if frame == 0:
        new_labels = True ## new labels for frame = 0
      current_word = text_subs[idx[-1]][frame < text_subs[idx[-1]]['end_frame']].iloc[0].name
      #pdb.set_trace()
      update_labels(strings, labels, new=current_word, new_labels=new_labels, x0=x0, fontsize=fontsize)

  anim = FuncAnimation(fig, update, frames=range(min([y.shape[0] for y in ys])),
                       init_func=init)
  Writer = writers['ffmpeg']
  writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
  
  if start is not None:
    anim.save(filename_temp, writer=writer, dpi=400)
    #anim.save(filename_temp, writer=writer, savefig_kwargs={'transparent': True, 'facecolor':'none'})
    audio_file = get_audio_file(data, interval_id)
    if audio_file is not None:
      add_audio(start, end, audio_file, filename_temp, filename)
    else:
      shutil.move(filename_temp, filename) ## if audio file not available, use the audio less file
  else:
    anim.save(filename, writer=writer)
  print('{} saved'.format(filename))
  plt.close('all')

def add_audio(start, end, audio_file, filename_temp, filename):
  ## add audio to the output
  command_inputs = []
  command_inputs.append('-ss')
  command_inputs.append('{}'.format(start))
  command_inputs.append('-i')
  command_inputs.append('{}'.format(audio_file))

  command = ['ffmpeg', '-y']
  command += command_inputs
  command.append('-i')
  command.append('"{}"'.format(filename_temp))
  command.append('-shortest')
  command.append('"{}"'.format(filename))

  #print(' '.join(command))
  FNULL = open(os.devnull, 'w')
  outs = subprocess.call(' '.join(command), shell=True, stdout=FNULL, stderr=FNULL)

  delete_command = ['rm', '{}'.format(filename_temp)]
  subprocess.call(delete_command, stderr=FNULL, stdout=FNULL)

  
def get_audio_file(data, interval_id):
  row = data.df[data.df['interval_id'] == interval_id]
  video_id = row['video_link'].values[0].split('=')[-1]
  speaker = row['speaker'].values[0]
  if '|' in speaker:
    speaker = speaker.split('|')[0]
    interval_id = interval_id.split('|')[0]
  if speaker == 'jon':
    audio_file = '{}_cropped/{}.mp3'.format(speaker, interval_id)
  else:
    audio_file = '{}_cropped/{}_{}.mp3'.format(speaker, video_id, interval_id)
  audio_file = (Path(data.path2data)/'raw'/audio_file).as_posix()
  
  if not os.path.exists(audio_file):
    audio_file = None
  return audio_file
  

'''
Save animations with corresposing audio.

  Arguments:
    y (torch.Tensor or np.array): Tensor of poses of size B x T x 2 x num_joints
    interval_ids (torch.Tensor, list): list of interval ids corresposing samples in ``y``
    desc (str): kind of loop- ``train``, ``dev`` or ``test``
    data (data.dataUtils.Data): dataloader instantiated to get information about the speaker
    start (float, optional): start time of audio of the interval in seconds (default: ``None``)
    start (float, optional): end time of audio of the interval in seconds (default: ``None``)
    idx: (int, optional): idx of the minidataloader for filenaming purposes (default: ``None``)
'''
def save_animation(y, interval_ids, dir_name, desc, data, start=None, end=None, idx=None, subname=None, text=None, output_modalities='pose/data', mask=[]):
  parents = data.modality_classes[output_modalities].parents

  ## negative masks to remove the non-predicted joints
  # mask = set(mask) - {0, 7, 8, 9} ## hardcoded to always render the head
  # negative_mask = sorted(set(range(len(parents))) - mask)
  # parents = np.array(parents)[negative_mask]
  # y = [[y__[..., negative_mask] for y__ in y_] for y_ in y]

  ## convert all args to lists for parallelization
  total_processes = len(interval_ids)
  interval_ids = get_tensor_items(interval_ids)
  parents = get_parallel_list(parents, total_processes, force=True)
  dir_name = get_parallel_list(dir_name, total_processes)
  desc = get_parallel_list(desc, total_processes)
  data = get_parallel_list(data, total_processes)
  start = get_tensor_items(get_parallel_list(start, total_processes))
  end = get_tensor_items(get_parallel_list(end, total_processes))
  idx = get_tensor_items(get_parallel_list(idx, total_processes))
  subname = get_parallel_list(subname, total_processes)
  text = get_parallel_list(text, total_processes)

#  animate(y[0], interval_ids[0], parents[0], dir_name[0], desc[0], data[0], start[0], end[0], idx[0], subname[0], text[0])
#  pdb.set_trace()
  # pdb.set_trace()
  # for i in range(total_processes):
  #   print(interval_ids[i])
  #   animate(y[i], interval_ids[i], parents[i], dir_name[i], desc[i], data[i], start[i], end[i], idx[i], subname[i], text[i])
  parallel(animate, -1,
           y, interval_ids, parents, dir_name, desc, data, start, end, idx, subname, text)


def in_modalities(modality, input_modalities):
  '''
  check if a modality like `text` is in input_modalities like ['pose/data', 'text/bert']
  '''
  for key in input_modalities:
    if modality in key:
      return True
  return False

def renderGroundTruth(args, exp_num):
  path2data = args.path2data
  speaker = args.speaker
  modalities = args.modalities
  fs_new = args.fs_new
  time = args.time
  split = args.split
  batch_size = args.batch_size
  shuffle = args.shuffle

  data = Data(path2data, speaker, modalities, fs_new,
              time=time, split=split, batch_size=batch_size,
              shuffle=shuffle)

  data_shape = data.shape
  
  output_modality = args.output_modalities[0]
  feats_shape = int(data_shape[output_modality][-1]/2)
  dir_name = (Path(args.path2outdata)/'{}'.format(args.speaker)).as_posix()
  os.makedirs(dir_name, exist_ok=True)

  data_train = data.train.dataset.datasets
  data_dev = data.dev.dataset.datasets
  data_test = data.test.dataset.datasets
  
  def loop(data, data_parent, desc='train'):
    Tqdm = tqdm(data, desc=desc+' {:.4f}'.format(0), leave=False, ncols=20)
    intervals = []
    start = []
    y_outs = []
    texts = []
    for count, loader in enumerate(Tqdm):
      if in_modalities('text', args.input_modalities):
        try:
          text = pd.read_hdf(loader.path2h5, 'text/meta')
        except:
          text = None
      else:
        text = None
      
      loader = DataLoader(loader, batch_size, shuffle=False)
      Y_cap = []
      for batch in loader:
        y = batch[output_modality]
         
        y[:, :, 0], y[:, :, feats_shape] = 0, 0
        
        y = y.view(y.shape[0], y.shape[1], 2, -1) ## (B, T, 2, feats)
        y = y.view(-1, 2, y.shape[-1]) ## (BxT, 2, feats)
        Y_cap.append(y)

      if Y_cap:
        intervals.append(batch['meta']['interval_id'][0])
        #start.append(torch.Tensor([0]).to(torch.float))
        start.append(0)
        y_outs.append([torch.cat(Y_cap, dim=0)])
        texts.append(text)
        
    if output_modality == 'pose/data':
      subname = 'eval'
    if output_modality == 'pose/normalize':
      subname = 'normalize'
    if in_modalities('text', args.input_modalities):
      subname += '_subs'

    save_animation(y_outs, intervals, dir_name, desc, data_parent, start, subname=subname, text=texts, output_modalities=output_modality)

  ## Training Loop
  loop(data_test, data, 'test')
  #loop(data_train, data, 'train')
  #loop(data_dev, data, 'dev')


def frames(ys, clusters, interval_id, parents, dir_name, data, subname, filename=None):
  if not isinstance(ys, list):
    ys = [ys]
  render_dir_name = 'render_{}'.format(subname) if subname is not None else 'render'
  if filename is None:
    filename = (Path(dir_name)/'{}/{}/{}/frame_{}.png'.format(render_dir_name,
                                                              data.getSpeaker(interval_id),
                                                              '{}', '{}'))

  def plt_figure(ys, count, filename):
    plt.ioff()
    fig = plt.figure()
    C = len(ys)
    axs = [fig.add_subplot(1, C, c+1) for c in range(C)]
    lns = [[ax.plot([], [])[0] for _ in parents[1:]] for ax in axs]

    for ax in axs:
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlim(-500, 500)
      ax.set_ylim(-500, 500)
      ax.set_aspect(aspect=1)

    for i, y in enumerate(ys):
      for joint, parent in enumerate(parents):
        if joint != 0:
          try:
            lns[i][joint-1].set_data([y[0, parent], y[0, joint]],
                                     [-y[1, parent], -y[1, joint]])
          except:
            pdb.set_trace()
    try:
      filename = filename.as_posix().format(clusters[count], count)
    except:
      pdb.set_trace()
    os.makedirs(Path(filename).parent, exist_ok=True)
    plt.savefig(filename)
    plt.close()

  for i in range(ys[0].shape[0]):
    plt_figure([y_[i] for y_ in ys], i, filename)

  return Path(filename).parent.parent
  
if __name__ == '__main__':
  argparseNloop(renderGroundTruth)
