import os 
import sys
import time
from datetime import datetime
import random
from multiprocessing import Queue

import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from trainers.pretrain import pretrain
from trainers.evaluate import evaluate
from trainers.test import test
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
hydra.HYDRA_FULL_ERROR = 1


#@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  now = datetime.now()
  start = time.time()
  pl.seed_everything(args.seed)

  args = prepend_paths(args)
  time.sleep(random.randint(1,5)) # Prevents multiple runs getting the same version when launching many jobs at once

  if args.resume_training:
    if args.wandb_id:
      wandb_id = args.wandb_id
    tmp_data_base = args.data_base
    checkpoint = args.checkpoint
    ckpt = torch.load(args.checkpoint)
    args = ckpt['hyper_parameters']
    args = OmegaConf.create(args)
    #with open_dict(args):
    args.checkpoint = checkpoint
    args.resume_training = True
    if not 'wandb_id' in args or not args.wandb_id:
      args.wandb_id = wandb_id
    # Run prepend again in case we move to another server and need to redo the paths
    args.data_base = tmp_data_base
    args = re_prepend_paths(args)
  
  if args.generate_embeddings:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'dataset')
    generate_embeddings(args)
    return args
  
  base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'result')
  
  exp_name = f'{args.exp_name}_{args.target}_{now.strftime("%m%d_%H%M")}'
  if args.use_wandb:
    if args.resume_training and args.wandb_id:
      wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline, id=args.wandb_id, resume='allow')
    else:
      wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline)
  else:
    wandb_logger = WandbLogger(name=exp_name, project='Test', entity=args.wandb_entity, save_dir=base_dir, offline=args.offline)
  args.wandb_id = wandb_logger.version

  if args.checkpoint and not args.resume_training:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'datatype')
  
  print('Comment: ', args.comment)
  print(f'Pretrain LR: {args.lr}, Decay: {args.weight_decay}')
  print(f'Finetune LR: {args.lr_eval}, Decay: {args.weight_decay_eval}')
  print(f'Corruption rate: {args.corruption_rate}, temperature: {args.temperature}')
  if args.algorithm_name == 'TIP':
    print('Special Replace Ratio: ', args.replace_special_rate)
    
  if args.pretrain:
    print('=================================================================================\n')
    print('Start pretraining\n')  
    print(f'Target is {args.target}')
    print('=================================================================================')
    torch.cuda.empty_cache()
    pretrain(args, wandb_logger)
    args.checkpoint = os.path.join(base_dir, 'runs', args.datatype, wandb_logger.experiment.name, f'checkpoint_last_epoch_{args.max_epochs-1:02}.ckpt')
  
  if args.test:
    test(args, wandb_logger)
  elif args.evaluate:
    print('=================================================================================\n')
    print('Start Finetuning')  
    print(f'Target is {args.target}')
    print('=================================================================================\n')
    torch.cuda.empty_cache()
    evaluate(args, wandb_logger)

  wandb.finish()
  del wandb_logger

  end = time.time()
  time_elapsed = end-start
  print('Total running time: {:.0f}h {:.0f}m'.
      format(time_elapsed // 3600, (time_elapsed % 3600)//60))

@property
def exception(self):
  if self._pconn.poll():
    self._exception = self._pconn.recv()
  return self._exception

@hydra.main(config_path='./configs', config_name='config_biomedia', version_base=None)
def control(args: DictConfig):
  run(args)

if __name__ == "__main__":
  control()

