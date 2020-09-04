import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ast import literal_eval
from pathlib import Path

from jinja2 import Environment, PackageLoader
import tempfile
import shutil
import pdb

from argsUtils import argparseNloop
from pycasper.BookKeeper import *

def get_list_files(path2videos, idx):
  speakers = literal_eval('[' + path2videos.split('[')[-1].split(']')[0] + ']')
  list_files = [['0'] + speakers]
  for sp1 in speakers:
    list_files_temp = []
    for i, sp2 in enumerate(speakers):
      if i == 0:
        list_files_temp.append(sp1)
      if sp1 == sp2:
        directory = 'render' 
      else:
        directory = '_'.join(['render', sp1, sp2])
      vid_parent = os.path.join(path2videos, directory, 'test', sp1)
      try:
          files = sorted(os.listdir(vid_parent))
      except:
          files = ['None']
      if idx < len(files):
        file = files[idx]
      else:
        file = files[0]
      filepath = os.path.join(vid_parent, file)
      filepath_ = os.path.join(directory, 'test', sp1, file)
      #new_path = '/'.join(modify_name(path2outvideos, filepath).split('/')[2:])
      list_files_temp.append(filepath_)
    list_files.append(list_files_temp)
  return list_files

def get_html_snippet(template_file, output_file, kwargs_dict, loader='app'):  
  env = Environment(loader=PackageLoader('htmlUtils.{}'.format(loader)))
  template = env.get_template(template_file)

  root = 'htmlUtils'
  filename = os.path.join(root, 'app/', output_file)
    
  with open(filename, 'w') as fh:
    fh.write(template.render(**kwargs_dict))
    

def makeHTMLfile(path2videos, idxs=20, outfile='videos'):
  file_list = []
  for idx in range(idxs):
    list_files = get_list_files(path2videos, idx)
    kwargs_dict = {'h2':'{}'.format(idx), 
                   'names': list_files, 
                   'columns': list(range(len(list_files[0])))}
    if idx == 0:
      kwargs_dict.update({'h1':'{}'.format(path2videos)})
    temp_filename = next(tempfile._get_candidate_names())
    get_html_snippet('grid.html', 'templates/{}.html'.format(temp_filename), kwargs_dict)
    file_list.append(temp_filename)
  kwargs_dict = {'filenames':['{}.html'.format(file) for file in file_list]}
  get_html_snippet('index.html', '{}.html'.format(temp_filename), kwargs_dict)

  temp_srcs = ['htmlUtils/app/templates/{}.html'.format(file) for file in file_list]
  src = 'htmlUtils/app/{}.html'.format(temp_filename)
  dest = os.path.join(path2videos, '{}.html'.format(outfile))
  shutil.move(src, dest)
  for temp_src in temp_srcs:
    os.remove(temp_src)

def makeHTMLfile_loop(args, exp_num):
  assert args.load, 'Load file must be provided'
  assert os.path.exists(args.load), 'Load file must exist'
  
  args_subset = ['exp', 'cpk', 'speaker', 'model']
  book = BookKeeper(args, args_subset, args_dict_update={'render':args.render},
                    tensorboard=args.tb)
  args = book.args

  dir_name = book.name.dir(args.save_dir)

  makeHTMLfile(dir_name, idxs=args.render, outfile='videos')
  makeHTMLfile(dir_name, idxs=4, outfile='videos_subset')

if __name__ == '__main__':
  argparseNloop(makeHTMLfile_loop)
