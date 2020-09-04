'''
Preprocess text files
Run it for all speakers after running audio preprocessing
```sh
python data/text.py -path2data ../dataset/groot/data -path2outdata ../dataset/groot/data -speaker all -preprocess_methods "['w2v']"
```
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
import librosa
import warnings
from joblib import Parallel, delayed
import h5py

from argsUtils import *
from pycasper.pathUtils import replace_Nth_parent
from common import Modality, MissingData, HDF5

import gensim
import nltk
import warnings

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.utils.data._utils.collate import default_collate
from functools import partial

def pad(datasets, key, dim):
  sizes = []
  for data in datasets:
    data = data[key]
    sizes.append(data.shape[dim])
  max_length = max(sizes)
  new_datasets = []
  lengths = []
  for data in datasets:
    data = data[key]
    length = data.shape[dim]
    zero_shape = list(data.shape)
    zero_shape[dim] = max_length-length
    new_datasets.append(np.concatenate([data, np.zeros(zero_shape)], axis=dim))
    lengths.append(length)
  return default_collate(new_datasets), default_collate(lengths)

def collate_fn_pad(batch, pad_key='text/meta', dim=0):
  if isinstance(batch[0], dict):
    data_dict = {}
    for key in batch[0]:
      if key in pad_key:
        padded_outs = pad(batch, key, dim=dim)
        if key == pad_key[-1]: ## TODO hardcoded to use the last key which is text/token_duration
          data_dict[key], data_dict['text/token_count'] = padded_outs[0], padded_outs[1]
        else:
          data_dict[key] = padded_outs[0]
      else:
        data_dict[key] = default_collate([d[key] for d in batch])
    return data_dict
  else:
    return default_collate(batch)

class Text(Modality):
  def __init__(self, path2data='../dataset/groot/data',
               path2outdata='../dataset/groot/data',
               speaker='all',
               preprocess_methods=['w2v'],
               text_aligned=0):
    super(Text, self).__init__(path2data=path2data)
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.speaker = speaker
    self.preprocess_methods = preprocess_methods

    self.missing = MissingData(self.path2data)

    ## list of word2-vec models
    self.w2v_models = []
    self.text_aligned = text_aligned
    
  def preprocess(self):
    ## load Glove/Word2Vec
    for pre_meth in self.preprocess_methods:
      if pre_meth == 'w2v':
        self.w2v_models.append(Word2Vec())
      elif pre_meth == 'bert':
        self.w2v_models.append(BertForSequenceEmbedding(hidden_size=512))
      elif pre_meth == 'tokens':
        self.w2v_models.append(BertSentenceBatching())        
      elif pre_meth == 'pos':
        self.w2v_models.append(POStagging())
      else:    
        raise 'preprocess_method not found'
    print('Embedding models loaded')
    
    if self.speaker[0] != 'all':
      speakers = self.speaker
    else:
      speakers = self.speakers

    if self.text_aligned:
      self.text_aligned_preprocessing(speakers)
    else:
      self.text_notAligned_preprocessing(speakers)

  def text_aligned_preprocessing(self, speakers):
    for speaker in tqdm(speakers, desc='speakers', leave=False):
      tqdm.write('Speaker: {}'.format(speaker))
      df_speaker = self.get_df_subset('speaker', speaker)
      filename_dict = {}
      interval_id_list = []
      for interval_id in tqdm(df_speaker.interval_id.unique(), desc='load'):
        path2interval = Path(self.path2data)/'processed'/speaker/'{}.h5'.format(interval_id)
        try:
          text = pd.read_hdf(path2interval, 'text/meta', 'r')
        except:
          warnings.warn('text/meta not found for {}'.format(interval_id))
          continue
        filename_dict[interval_id] = text
        interval_id_list.append(interval_id)
      missing_data_list = [] 
      for interval_id in tqdm(interval_id_list, desc='save'):
        inter = self.save_intervals(interval_id, speaker, filename_dict, None)
        missing_data_list.append(inter)
      self.missing.save_intervals(set(missing_data_list))

        
  def text_notAligned_preprocessing(self, speakers):
    for speaker in tqdm(speakers, desc='speakers', leave=False):
      tqdm.write('Speaker: {}'.format(speaker))
      df_speaker = self.get_df_subset('speaker', speaker)
      df_speaker.loc[:, 'video_id'] = df_speaker['video_link'].apply(lambda x: x.split('=')[-1])
      df_speaker.loc[:, 'Start'] = pd.to_timedelta(df_speaker['start_time'].str.split().str[1]).dt.total_seconds()
      df_speaker.loc[:, 'End'] = pd.to_timedelta(df_speaker['end_time'].str.split().str[1]).dt.total_seconds()
      interval_ids = df_speaker['interval_id'].unique()
      ## find path to processed files
      parent = Path(self.path2data)/'raw'/'{}'.format(speaker)
      filenames = os.listdir(parent)
      filenames = [filename for filename in filenames if filename.split('_')[-1] == 'transcripts']
      filenames = ['{}/{}.csv'.format(filename, '_'.join(filename.split('_')[:-1])) for filename in filenames]
      is_path = lambda x: os.path.exists(Path(parent)/x)
      # for filename in filenames:
      #   if not is_path(filename):
      #     pdb.set_trace()
      filenames = filter(is_path, filenames) ## remove paths that don't exist
      filename_dict = {Path(filename).stem: filename for filename in filenames}

      interval_lists = []
      for key in tqdm(filename_dict):
        interval_list = self.get_intervals_from_videos(key, df_speaker,
                                                       filename_dict, parent,
                                                       speaker)
        interval_lists += interval_list
      missing_data_list = set(interval_ids) - set(interval_lists)
      self.missing.save_intervals(missing_data_list)

  def get_intervals_from_videos(self, key, df, filename_dict, basepath, speaker):
    #interval_dict = {}
    ## Read the transcript
    path2text = Path(basepath) / filename_dict[key]
    text = pd.read_csv(path2text)

    ## get all intervals from the video id in a sorted table
    if key[:2] == '_-':
      key = key[2:]
    df_video = df[df['video_id'] == key].sort_values(by='start_time')
    if df_video.empty: ## non youtube videos
      new_key = '-'.join(key.split('-')[-5:])
      df_video = df[df['video_id'].apply(lambda x: new_key in x)].sort_values(by='start_time')
    text.loc[:, 'interval_id'] = text['End'].apply(self.find_interval_for_words, args=(df_video,))

    interval_ids = filter(None, text['interval_id'].unique())
    interval_ids = [idx for idx in interval_ids]
    texts = []
    for interval_id in interval_ids:
      try:
        ## get max_len of the pose data
        interval_path = replace_Nth_parent(basepath, 'processed')/'{}.h5'.format(interval_id)
        data, h5 = self.load(interval_path, 'pose/data')
        max_len = data.shape[0]
        h5.close()
      except: ## sometimes the interval is missing
        continue

      start_offset = pd.to_timedelta(self.df[self.df['interval_id'] == interval_id]['start_time'].str.split().str[1]).dt.total_seconds().iloc[0]

      start_frames, end_frames = [], []
      for i, row in text[text['interval_id'] == interval_id].reset_index().iterrows():
        start = row['Start']
        if i == 0:
          start_frames.append(0)
        else:
          start_frames.append(int(min(int((start - start_offset)*self.fs('text')), max_len)))
          end_frames.append(start_frames[-1])
      end_frames.append(max_len)
      text.loc[text['interval_id'] == interval_id, 'start_frame'] = start_frames
      text.loc[text['interval_id'] == interval_id, 'end_frame'] = end_frames
      #interval_dict[interval_id] = text[text['interval_id'] == interval_id].reset_index()
      subtext = text[text['interval_id'] == interval_id].reset_index()
      #texts.append(subtext)
      self.save_intervals(interval_id, speaker, {interval_id:subtext}, basepath)
    return interval_ids
  
  ## Find intervals corresponding to each word
  def find_interval_for_words(self, end_time, df):
    interval_ids = df[(df['End'] >= end_time) & (df['Start'] < end_time)]['interval_id']
    if interval_ids.shape[0] > 1:
      warnings.warn('More than one interval for one word')
    if interval_ids.shape[0] == 0:
      return None
    return str(interval_ids.iloc[0])
    
  def save_intervals(self, interval_id, speaker, filename_dict, parent):
    if interval_id in filename_dict:
      ## Store Meta
      text = filename_dict[interval_id][['Word', 'start_frame', 'end_frame']]
      #dt = h5py.special_dtype(vlen=str)
      #text = np.asarray(text, dtype=dt)
      filename = Path(self.path2outdata)/'processed'/speaker/'{}.h5'.format(interval_id)
      key = self.add_key(self.h5_key, ['meta'])

      if not HDF5.isDatasetInFile(filename, key):
        text.to_hdf(filename, key, mode='a')
      #self.append(filename, key, text)
      
      ## process data for each preprocess_method
      processed_datas = self.process_interval(interval_id, parent, filename_dict)

      ## save processed_data
      for preprocess_method, processed_data in zip(self.preprocess_methods, processed_datas):
        filename = Path(self.path2outdata)/'processed'/speaker/'{}.h5'.format(interval_id)
        key = self.add_key(self.h5_key, [preprocess_method])
        try:
          self.append(filename, key, processed_data)
        except:
          warnings.warn('interval_id: {} busy.'.format(interval_id))
          return interval_id
      return None
    else:
      warnings.warn('interval_id: {} not found.'.format(interval_id))
      return interval_id
  
  def process_interval(self, interval_id, parent, filename_dict):
    ## get filename
    text = filename_dict[interval_id]
    words_repeated = []
    for i, row in text.reset_index().iterrows():
      words_repeated += [row['Word']] * int((row['end_frame'] - row['start_frame']))

    processed_datas = []
    ## process file
    for preprocess_method, model in zip(self.preprocess_methods, self.w2v_models):
      if preprocess_method in ['w2v']:
        processed_datas.append(self.preprocess_map[preprocess_method](words_repeated, model))
      elif preprocess_method in ['bert']:
        processed_datas.append(self.preprocess_map[preprocess_method](text, model))
      elif preprocess_method in ['tokens']:
        processed_datas.append(self.preprocess_map[preprocess_method](text, model))        
      elif preprocess_method in ['pos']:
        processed_datas.append(self.preprocess_map[preprocess_method](text, model, words_repeated))

    ## return processed output
    return processed_datas

  '''
  PreProcess Methods
  '''
  @property
  def preprocess_map(self):
    return {
      'w2v':self.w2v,
      'bert':self.bert,
      'tokens':self.bert_tokens,
      'pos':self.pos
      }
  
  def w2v(self, words, model):
    return model(words)[0].squeeze(1)

  def bert(self, text, model):
    text['delta_frames'] = (text['end_frame'] - text['start_frame']).apply(int)
    text_delta_frames = text.delta_frames
    words = text['Word'].values
    words = [word.lower() for word in words]
    sentence = [' '.join(words)]
    outs, pool, words_cap, mask = model(sentence)
    count = 0
    text_cap = pd.DataFrame(columns=text.columns)
    temp_words = []
    temp_word = []
    delta_frames = []
    delta_frames_cap = []
    for word in words_cap[0][1:-1]:
      if '##' == word[:2]:
        temp_word.append(word[2:])
      else:
        temp_word.append(word)
      if ''.join(temp_word) == words[count]:
        temp_words.append((''.join(temp_word), len(temp_word)))
        delta_frames.append(len(temp_word))
        delta_frames_cap += [int(text_delta_frames[count]/delta_frames[-1])]*delta_frames[-1]
        if delta_frames[-1] > 1:
          delta_frames_cap[-1] = text_delta_frames.iloc[count] - sum(delta_frames_cap[-delta_frames[-1]+1:])
        temp_word = []
        count+=1

    feats = []
    for i, frames in enumerate(delta_frames_cap):
      feats += [outs[0, i+1:i+2]]*frames
    try:
      feats = torch.cat(feats, dim=0)
    except:
      pdb.set_trace()
    if not feats.shape[0] == sum(text_delta_frames):
      pdb.set_trace()
    return feats

  def bert_tokens(self, text, model):
    text['delta_frames'] = (text['end_frame'] - text['start_frame']).apply(int)
    text_delta_frames = text.delta_frames
    words = text['Word'].values
    words = [word.lower() for word in words]
    sentence = [' '.join(words)]
    outs, mask, words_cap = model(sentence)

    words_cap_ = []
    outs_list = []
    for wc, mk, ot in zip(words_cap, mask, outs):
      words_cap_ += wc[1:sum(mk).item() - 1]
      outs_list.append(ot[1:sum(mk).item() - 1])
    words_cap = words_cap_
    outs = torch.cat(outs_list)

    count = 0
    text_cap = pd.DataFrame(columns=text.columns)
    temp_words = []
    temp_word = []
    delta_frames = []
    delta_frames_cap = []
    for word in words_cap:
      if '##' == word[:2]:
        temp_word.append(word[2:])
      else:
        temp_word.append(word)
      if ''.join(temp_word) == words[count]:
        temp_words.append((''.join(temp_word), len(temp_word)))
        delta_frames.append(len(temp_word))
        delta_frames_cap += [int(text_delta_frames[count]/delta_frames[-1])]*delta_frames[-1]
        if delta_frames[-1] > 1:
          delta_frames_cap[-1] = text_delta_frames.iloc[count] - sum(delta_frames_cap[-delta_frames[-1]+1:])
        temp_word = []
        count+=1

    feats = []
    for i, frames in enumerate(delta_frames_cap):
      feats += [outs[i:i+1]]*frames
    try:
      feats = torch.cat(feats, dim=0)
    except:
      pdb.set_trace()
    if not feats.shape[0] == sum(text_delta_frames):
      pdb.set_trace()
    return feats

  
  def pos(self, text, model, words_repeated):
    return model(text, words_repeated)

  def fs(self, modality):
    return 15
  
  @property
  def h5_key(self):
    return 'text'

class BaseTokenizer():
  def __init__(self, vocab):
    self.vocab = vocab
    self.hidden_size = 300
    self._UNK = '_UNK'
    self._SEP = '_SEP'
    self.random_vec = np.random.rand(self.hidden_size)
    self.zero_vec = np.zeros(self.hidden_size)
    
  def tokenize(self, sentence):
    words_ = sentence.split(' ')

    ''' Lowercase all words '''
    words_ = [w.lower() for w in words_]

    ''' Add _UNK for unknown words '''
    words = []
    for word in words_:
      if word in self.vocab:
        words.append(word)
      else:
        words.append('_UNK')
    return words

class Word2Vec():
  '''
  Take a bunch of words and convert it to vectors.
  * Tokenize
  * Add _UNK for words that do not exist
  * Create a mask which denotes the batches
  '''
  def __init__(self, path2file='./data/nlp/GoogleNews-vectors-negative300.bin.gz'):
    self.model = gensim.models.KeyedVectors.load_word2vec_format(path2file, binary=True)
    print('Loaded Word2Vec model')
    
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BaseTokenizer(self.model.vocab)
    
    # Tokenized input
    text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    tokenized_text = self.tokenizer.tokenize(text)
    print('Tokenization example')
    print('{}  --->  {}'.format(text, tokenized_text))
    
  def __call__(self, x):
    x = [self.tokenizer.tokenize(x_) for x_ in x]
    max_len = max([len(x_) for x_ in x])

    mask = np.array([[1]*len(x_) + [0]*(max_len-len(x_)) for x_ in x])
    x = [x_ + ['_SEP']*(max_len-len(x_)) for x_ in x]
    vectors = []
    for sentence in x:
      vector = []
      for word in sentence:
        if word == self.tokenizer._UNK:
          vector.append(self.tokenizer.random_vec)
        elif word == self.tokenizer._SEP:
          vector.append(self.tokenizer.zero_vec)
        else:
          vector.append(self.model.word_vec(word))
      vector = np.stack(vector, axis=0)
      vectors.append(vector)
    vectors = np.stack(vectors, axis=0)
    return vectors, mask, x
  

def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off
  
class BertForSequenceEmbedding(nn.Module):
  def __init__(self, hidden_size):
    #config = BertConfig(32000) ## Dummy config file
    #super(BertForSequenceEmbedding, self).__init__(config)
    super(BertForSequenceEmbedding, self).__init__()
    self.hidden_size = hidden_size
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    # self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, hidden_size),
    #                                 nn.Dropout(self.bert.config.hidden_dropout_prob),
    #                                 nn.ReLU(),
    #                                 nn.Linear(hidden_size, hidden_size))
    ''' Fix Bert Embeddings and encoder '''
    toggle_grad(self.bert.embeddings, False)
    toggle_grad(self.bert.encoder, False)
    self.bert.eval()
    
    self.pre = BertSentenceBatching()
  
  def forward(self, sentences):
    input_ids, attention_mask, x = self.pre(sentences)
    token_type_ids = None
    outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    #outputs = self.classifier(outputs[:, 0])
    if len(sentences) == 1: ## batch size == 1
      len_data = sum(attention_mask.view(-1)) - (attention_mask.shape[0])*2
      outputs = outputs[:, 1:-1,:].reshape(-1, outputs.shape[-1])[:len_data]
      outputs = torch.cat([outputs.new(1, outputs.shape[-1]), outputs, outputs.new(1, outputs.shape[-1])])
      outputs = outputs.view(1, outputs.shape[0], outputs.shape[1])
      #x_ = ['[CLS]']
      x_ = []
      for X in x:
        x_ += X[1:-1]
      x_ = x_[:len_data]
      x = [['[CLS]'] + x_ + ['[SEP]'] ]
      #x = [['[CLS]'] + sum([x_[1:-1] for x_ in x]) +['[SEP]']]
    return outputs, pooled_output, x, attention_mask

  def train_params(self, idx=0):
    params = [self.classifier, self.bert.pooler]
    return params[idx].parameters()

  def train(self, mode=True):
    self.training = mode
    for module in self.children():
      module.train(mode)
    self.bert.eval() ## bert needs to be in eval mode for both modes
    return self

class BertSentenceBatching(nn.Module):
  '''
  Take a bunch of sentences and convert it to a format that Bert can process
  * Tokenize
  * Add [CLS] and [SEP] tokens
  * Create a mask which denotes the batches
  '''
  def __init__(self):
    super(BertSentenceBatching, self).__init__()
    self.dummy_param = nn.Parameter(torch.Tensor([1]))
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = self.tokenizer.tokenize(text)
    print('Tokenization example')
    print('{}  --->  {}'.format(text, tokenized_text))
    
  def __call__(self, x):
    self.device = self.dummy_param.device
    x = [self.tokenizer.tokenize(x_) for x_ in x]
    if len(x) == 1: ## when batch_size == 1 break it up into chunks of less than 510 words
      x = [x[0][i:i+510] for i in range(0, len(x[0]), 510)]
    x = [['[CLS]'] + x_ + ['[SEP]'] for x_ in x]
    max_len = max([len(x_) for x_ in x])

    mask = torch.Tensor([[1]*len(x_) + [0]*(max_len-len(x_)) for x_ in x]).long().to(self.device)
    x = [x_ + ['[SEP]']*(max_len-len(x_)) for x_ in x]
    indexed_tokens = torch.Tensor([self.tokenizer.convert_tokens_to_ids(x_) for x_ in x]).long().to(self.device)
    return indexed_tokens, mask, x

class POStagging():
  def __init__(self):
    self.tagset_full = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    count = 0
    self.tagset_rev_full = {}
    self.tagset_rev = {}
    self.tagset = []
    for tag in self.tagset_full:
      if tag[0] not in self.tagset_rev:
        self.tagset_rev[tag[0]] = count
        self.tagset.append(tag[0])
        self.tagset_rev_full[tag] = count
        count += 1
      else:
        self.tagset_rev_full[tag] = self.tagset_rev[tag[0]]

    #self.tagset_rev = {tag:i for i,tag in enumerate(self.tagset)}

  def __call__(self, x, words_repeated):
    words = list(x.Word.values)
    pos_tags = nltk.pos_tag(words)
    pos = []
    pos_labels = []
    for tag in pos_tags:
      try:
        pos.append(tag[1])
        pos_labels.append(self.tagset_rev_full.get(tag[1], self.tagset_rev_full['NN']))
      except:
        pdb.set_trace()

    pos_repeated = []
    for i, row in x.reset_index().iterrows():
      pos_repeated += [pos_labels[i]] * int((row['end_frame'] - row['start_frame']))
    return np.array(pos_repeated, dtype=np.int)
      
def preprocess(args, exp_num):
  path2data = args.path2data #'../dataset/groot/speech2gesture_data/'
  path2outdata = args.path2outdata #'../dataset/groot/data/processed/'
  speaker = args.speaker
  preprocess_methods = args.preprocess_methods
  text_aligned = args.text_aligned
  text = Text(path2data, path2outdata, speaker, preprocess_methods, text_aligned)
  text.preprocess()

  
if __name__ == '__main__':
  argparseNloop(preprocess)
