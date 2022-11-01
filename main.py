import random
import os
import config
import data_loader
import nmt_utils
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from train import train, test
#from data_loader import MTDataset_zh2en as MTDataset
from model import make_model, LabelSmoothing


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('固定随机种子:', seed)


#set_seed(77)


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    nmt_utils.set_logger(config.log_path)
    MTDataset = None
    if config.TRAN_ORI == 'zh2en':
        MTDataset = data_loader.MTDataset_zh2en
    if config.TRAN_ORI == 'en2ms':
        MTDataset = data_loader.MTDataset_en2ms
    if config.TRAN_ORI == 'zh2ms':
        MTDataset = data_loader.MTDataset_zh2ms

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info('【翻译模式:】' + str(config.TRAN_ORI))
    logging.info('【是否翻转翻译:】' + str(config.TRAN_REVERSE))
    logging.info('训练集:' + str(config.train_data_path))
    logging.info('开发集:' + str(config.dev_data_path))
    logging.info('batch_size:' + str(config.batch_size))
    logging.info('epoch_num:' + str(config.epoch_num))
    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    if not os.path.exists(config.model_path):
        logging.info("[!!!] LOAD_LAST_MODEL为True，但是SAVE_FILE文件不存在!")
    else:
        model.load_state_dict(torch.load(config.model_path))
        logging.info("[!] 从文件中加载模型参数完成!")

    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        logging.info('【指定学习率】:' + str(config.lr))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    #test(test_dataloader, model, criterion)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings

    warnings.filterwarnings('ignore')
    run()
