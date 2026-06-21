from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from datasets.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
from omegaconf import OmegaConf
import pandas as pd
from os.path import join
from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, grab_wids, create_logdir


def test(hparams, wandb_logger=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  
  if hparams.eval_datatype == 'imaging':
      test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
                                  dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
      hparams.transform_test = test_dataset.transform_val.__repr__()
  elif hparams.eval_datatype == 'multimodal':
    assert hparams.strategy == 'tip'
    test_dataset = ImagingAndTabularDataset(
    hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
    hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
    hparams.labels_test_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=0,
    data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
    augmentation_speedup=hparams.augmentation_speedup
  )
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.eval_datatype == 'tabular':
    test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, 0, 0, train=False, 
                                eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                data_base=hparams.data_base, 
                                strategy=hparams.strategy, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate)
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.eval_datatype == 'imaging_and_tabular':
    test_dataset = ImagingAndTabularDataset(
      hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
      hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target,
      corruption_rate=0.0, data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
      augmentation_speedup=hparams.augmentation_speedup)
    hparams.input_size = test_dataset.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  
  drop = ((len(test_dataset)%hparams.batch_size)==1)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  logdir = create_logdir('test', hparams.resume_training, wandb_logger)
  hparams.dataset_length = len(test_loader)

  tmp_hparams = OmegaConf.create(OmegaConf.to_container(hparams, resolve=True))
  tmp_hparams.checkpoint = None
  model = Evaluator(tmp_hparams)
  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
  test_results = trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)
  df = pd.DataFrame(test_results)
  df.to_csv(join(logdir, 'test_results.csv'), index=False)