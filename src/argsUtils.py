import argparse
import itertools
from ast import literal_eval

def get_args_perm():
  parser = argparse.ArgumentParser()

  ## Dataset Parameters
  parser.add_argument('-path2data', nargs='+', type=str, default=['../dataset/groot/data'],
                      help='path to data')
  parser.add_argument('-path2outdata', nargs='+', type=str, default=['../dataset/groot/data'],
                      help='path to output data (used for pre-processing)')
  parser.add_argument('-speaker', nargs='+', type=literal_eval, default=['oliver'],
                      help='choose speaker or `all` to use all the speakers available')  
  parser.add_argument('-modalities', nargs='+', type=literal_eval, default=[['pose/data', 'audio/log_mel_512']],
                      help='choose a set of modalities to be loaded by the dataloader')  
  parser.add_argument('-input_modalities', nargs='+', type=literal_eval, default=[None],
                      help='choose the input modalities')  
  parser.add_argument('-output_modalities', nargs='+', type=literal_eval, default=[None],
                      help='choose the output_modalities')
  parser.add_argument('-mask', nargs='+', type=literal_eval, default=[[0, 7, 8, 9]],
                      help='mask out the features of pose using a list')
  parser.add_argument('-split', nargs='+', type=literal_eval, default=[None],
                      help='(train,dev) split of data. default=None')
  parser.add_argument('-batch_size', nargs='+', type=int, default=[32],
                      help='minibatch size. Use batch_size=1 when using time=0')
  parser.add_argument('-shuffle', nargs='+', type=int, default=[1],
                      help='shuffle the data after each epoch. default=True')
  parser.add_argument('-time', nargs='+', type=int, default=[4.3],
                      help='time (in seconds) for each sample')
  parser.add_argument('-fs_new', nargs='+', type=literal_eval, default=[[15, 15]],
                      help='subsample to the new frequency')  
  parser.add_argument('-num_workers', nargs='+', type=int, default=[1],
                      help='number of workers to load the dataset')
  parser.add_argument('-window_hop', nargs='+', type=int, default=[0],
                      help='overlap of windows sampled from the interval. default:0 which means no overlap')
  parser.add_argument('-num_clusters', nargs='+', type=int, default=[None],
                      help='number of clusters for pose data')
  parser.add_argument('-pos', nargs='+', type=int, default=[0],
                      help='if true, us POS tags as cluster labels')
  parser.add_argument('-feats', nargs='+', type=literal_eval, default=[['pose', 'velocity']],
                      help='the kind of feats to use for clustering')
  parser.add_argument('-style_dim', nargs='+', type=int, default=[10],
                      help='Dimension of the style embedding')
  parser.add_argument('-style_losses', nargs='+', type=literal_eval, default=[{'id_a':1, 'id_p':1, 'cluster_a':1, 'cluster_p':1, 'style_a':1, 'style_p':1, 'content_+':1, 'content_-':1, 'rec_a':1, 'rec_p':1}],
                      help='Dimension of the style embedding')
  parser.add_argument('-style_iters', nargs='+', type=int, default=[0],
                      help='number of iterations for style based models where p+ and p- are required')
  parser.add_argument('-load_data', nargs='+', type=int, default=[1],
                      help='if load_data=1, load data for training, if 0, do not load data saves time and memory for using pretrained models')
  parser.add_argument('-repeat_text', nargs='+', type=int, default=[1],
                      help='if True repeats text to match the the frequency of the pose')
  parser.add_argument('-filler', nargs='+', type=int, default=[0],
                      help='if True return filler mask with text modality. Works with only one text modality')
  parser.add_argument('-relative2parent', nargs='+', type=int, default=[0],
                      help='if True the joints are calculated wrt to parents, if False calculated wrt the root=0')
  parser.add_argument('-quantile_sample', nargs='+', type=literal_eval, default=[None],
                      help='choose the top q percentile of the poses (with high velocity) for training. [q1, q2] for poses above q2 and below q2 i.e. tail enders. q> 1 rebalances the dataset - Use with quantile_num_training_sample > 0. use None to use the complete dataset')
  parser.add_argument('-quantile_num_training_sample', nargs='+', type=int, default=[3000],
                      help='number of samples to train after rebalncing dataset, default: 3000')
  parser.add_argument('-finetune_quantile_sample', nargs='+', type=float, default=[None],
                      help='finetune using a quantile of the data after training. It is hardcoded to run for 20 epochs')
  
  parser.add_argument('-pretrained_model', nargs='+', type=int, default=[0],
                      help='if pretrained_model=1')
  parser.add_argument('-pretrained_model_weights', nargs='+', type=str, default=[None],
                      help='path to pretrained weights')

  parser.add_argument('-noise', nargs='+', type=float, default=[0],
                      help='0 or 0.1 or 0.01 specify the std of the noise to be added to the ground truth to improve robusteness of the model')
  
  parser.add_argument('-view', nargs='+', type=str, default=['sentences.txt'],
                      help='list of sentences to sample from')  

  ## BookKeeper Args
  parser.add_argument('-exp', nargs='+', type=int, default=[None],
                      help='experiment number')
  parser.add_argument('-debug', nargs='+', type=int, default=[0],
                      help='debug mode')
  parser.add_argument('-save_dir', nargs='+', type=str, default=['save/model'],
                      help='directory to store checkpointed models')
  parser.add_argument('-cpk', nargs='+', type=str, default=['m'],
                      help='checkpointed model name')
  parser.add_argument('-dev_key', nargs='+', type=str, default=['dev'],
                      help='Dev Key. Metric used to decide early stopping')
  parser.add_argument('-dev_sign', nargs='+', type=int, default=[1],
                      help='if lesser loss is better choose 1, else choose -1')
  parser.add_argument('-tb', nargs='+', type=int, default=[0],
                      help='Tensorboard Flag')
  parser.add_argument('-seed', nargs='+', type=int, default=[11212],
                      help='manual seed')
  parser.add_argument('-load', nargs='+', type=str, default=[None],
                      help='Load weights from this file')
  parser.add_argument('-cuda', nargs='+', type=int, default=[0],
                      help='choice of gpu device, -1 for cpu')
  parser.add_argument('-overfit', nargs='+', type=int, default=[0],
                      help='disables early stopping and saves models even if the dev loss increases. useful for performing an overfitting check')
  parser.add_argument('-note', nargs='+', type=str, default=[None],
                      help='Notes about the model')

  
  ## model hyperparameters
  parser.add_argument('-model', nargs='+', type=str, default=['SnL2PoseLate_G'],
                      help='choice of model')  
  parser.add_argument('-modelKwargs', nargs='+', type=literal_eval, default=[{}],
                      help='choice of model arguments')

  ## GAN params
  parser.add_argument('-gan', nargs='+', type=int, default=[0],
                      help='if True, train with a GAN at the end')
  parser.add_argument('-dg_iter_ratio', nargs='+', type=float, default=[1],
                      help='Discriminator Generator Iteration Ratio')
  parser.add_argument('-lambda_gan', nargs='+', type=float, default=[1],
                      help='lambda for G_gan_loss (generator loss)')
  parser.add_argument('-lambda_D', nargs='+', type=float, default=[1],
                      help='lamdda for fake_D_loss (discrimiator loss)')
  parser.add_argument('-joint', nargs='+', type=int, default=[0],
                      help='to feed in X to the discriminator along with the fake/real pose set it to 1')
  parser.add_argument('-update_D_prob_flag', nargs='+', type=int, default=[0],
                      help='Update D_prob in a GAN. True by default') 
  parser.add_argument('-no_grad', nargs='+', type=int, default=[0],
                      help='Use no_grad while training for generator in a GAN. True by default')

  parser.add_argument('-discriminator', nargs='+', type=str, default=[None],
                      help='name of the discriminator. if None, it will infer from the name of the model')
  parser.add_argument('-weighted', nargs='+', type=int, default=[0],
                      help='if 0, choose GAN, else GANWeighted')

  ## Noise Params
  parser.add_argument('-noise_only', nargs='+', type=int, default=[0],
                      help='if True, train with inputs as Noise')

  ## Loss hyperparameters
  parser.add_argument('-loss', nargs='+', type=str, default=['MSELoss'],
                      help='choice of losses MSELoss, SmoothL1loss etc.')
  parser.add_argument('-lossKwargs', nargs='+', type=literal_eval, default=[{}],
                      help='kwargs corresposing to the losses')

  ## Pre-processing args
  parser.add_argument('-preprocess_methods', nargs='+', type=literal_eval, default=[['log_mel_512']],
                      help='preprocess methods used by functions like skeleton.py, audio.py in the folder data/ etc.')  
  parser.add_argument('-preprocess_only', nargs='+', type=int, default=[0],
                      help='preprocess methods used by functions like skeleton.py, audio.py in the folder data/ etc.')  
  parser.add_argument('-text_aligned', nargs='+', type=int, default=[1],
                      help='preprocess methods after text is aligned and stored at "meta"')  

  
  ## training parameters
  parser.add_argument('-num_epochs', nargs='+', type=int, default=[50],
                      help='number of epochs for training')
  parser.add_argument('-early_stopping', nargs='+', type=int, default=[1],
                      help='Use 1 for early stopping')
  parser.add_argument('-greedy_save', nargs='+', type=int, default=[1],
                      help='save weights after each epoch if 1')
  parser.add_argument('-save_model', nargs='+', type=int, default=[1],
                      help='flag to save model at every step')
  parser.add_argument('-stop_thresh', nargs='+', type=int, default=[3],
                      help='number of consequetive validation loss increses before stopping')
  parser.add_argument('-min_epochs', nargs='+', type=int, default=[0],
                      help='minimum number of epochs after which early stoppping can be invoked')

  parser.add_argument('-eps', nargs='+', type=float, default=[0],
                      help='if the decrease in validation is less than eps, it counts for one step in stop_thresh ')
  parser.add_argument('-num_iters', nargs='+', type=int, default=[0],
                      help='breaks the loop after number of iteration, =0 implies complete dataset')
  parser.add_argument('-num_training_iters', nargs='+', type=int, default=[None],
                      help='number of training iterations; different from num_training_sample')
  parser.add_argument('-num_training_sample', nargs='+', type=int, default=[None],
                      help='[Few shot learning] Chooses a fixed set of training samples from the dataset')  
  parser.add_argument('-metrics', nargs='+', type=int, default=[1],
                      help='if true update all the metrics')

  ## Training Parameters
  parser.add_argument('-curriculum', nargs='+', type=int, default=[0],
                      help='learn generating time steps by starting with 2 timesteps upto time, increasing by a power of 2')
  parser.add_argument('-kl_anneal', nargs='+', type=int, default=[0],
                      help='anneal kl loss till the number of epochs')
  
  ## optimization paramters
  parser.add_argument('-optim', nargs='+', type=str, default=['Adam'],
                      help='optimizer')
  parser.add_argument('-lr', nargs='+', type=float, default=[0.0001],
                      help='learning rate')
  parser.add_argument('-optimKwargs', nargs='+', type=literal_eval, default=[{}],
                      help='kwargs corresposing to optims')
  parser.add_argument('-optim_separate', nargs='+', type=float, default=[None],
                      help='separate starting LR for bert')

  ## lr_scheduler parameter
  parser.add_argument('-scheduler', nargs='+', type=str, default=[None],
                      help='kind of scheduler; choices: linear_decay')
  parser.add_argument('-scheduler_warmup_steps', nargs='+', type=int, default=[0],
                      help='number of warmup steps for linear decay')
  parser.add_argument('-gamma', nargs='+', type=float, default=[0.99],
                      help='learning rate decay gamma')

  ## dataProcessing/augmentDataset.py parameters
  parser.add_argument('-angles', nargs='+', type=literal_eval, default=[[90]],
                      help='set of angles to augment data. Example: [-90, 90]')  
  
  ## slurm_generator Parameters
  parser.add_argument('-config', nargs='+', type=str, default=[None],
                      help='Config file to generate slurm job files')
  parser.add_argument('-script', nargs='+', type=str, default=[None],
                      help='script to use for the job files')
  parser.add_argument('-prequel', nargs='+', type=str, default=['source activate torch\\n'],
                      help='prequel to the script in slurm_generator')

  ## sampling methods
  parser.add_argument('-sample_all_styles', nargs='+', type=int, default=[0],
                      help='if >0, all styles sampled with number of samples == sample_all_styles, if -1, all samples and styles will be sampled')  
  parser.add_argument('-mix', nargs='+', type=int, default=[0],
                      help='if True, generate animations as a mixture of all styles')  
  
  ## render.py Params
  parser.add_argument('-clean_render', nargs='+', type=int, default=[1],
                      help='render all videos from scratch if True')  
  parser.add_argument('-render_list', nargs='+', type=str, default=[None],
                      help='render videos only from the render list')

  ## render_in sample.py Params
  parser.add_argument('-render', nargs='+', type=int, default=[0],
                      help='render animation or not')  
  parser.add_argument('-render_text', nargs='+', type=int, default=[1],
                      help='render text in animation or not')  
  parser.add_argument('-render_transparent', nargs='+', type=int, default=[0],
                      help='render background transparent if true')  

  ## evil twins
  parser.add_argument('-transforms', nargs='+', type=literal_eval, default=[['mirror']],
                      help='transform to create a new speaker')  


  ## render after training
  parser.add_argument('-cpu', nargs='+', type=int, default=[10],
                      help='number of cpus for training')  
  parser.add_argument('-mem', nargs='+', type=int, default=[16000],
                      help='number of cpus for training')  
  
  
  args, unknown = parser.parse_known_args()
  print(args)
  print(unknown)

  ## Create a permutation of all the values in argparse
  args_dict = args.__dict__
  args_keys = sorted(args_dict)
  args_perm = [dict(zip(args_keys, prod)) for prod in itertools.product(*(args_dict[names] for names in args_keys))]
  
  return args, args_perm

def argparseNloop(loop):
  args, args_perm = get_args_perm()

  for i, perm in enumerate(args_perm):
    args.__dict__.update(perm)
    print(args)    
    loop(args, i)
