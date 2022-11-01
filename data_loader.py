import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nmt_utils import malaysian_tokenizer_load
from nmt_utils import XLNET_chinese_tokenizer_load
import config

DEVICE = config.device


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0, src_is_zh=False):
        self.src_text = src_text
        self.trg_text = trg_text
        src_token_type_ids = None

        if src_is_zh:
            src_tokens = src['input_ids'].to(DEVICE)
            src_mask = torch.tensor(src['attention_mask'], dtype=torch.bool).to(DEVICE)
            src_token_type_ids = src['token_type_ids'].to(DEVICE)
        else:
            src_mask = (src != pad).unsqueeze(-2)
            src_tokens = src.to(DEVICE)
        self.src = src_tokens
        self.src_mask = src_mask
        self.src_token_type_ids = src_token_type_ids

        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵

        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset_zh2ms(Dataset):
    def __init__(self, data_path):
        print('中2马数据集加载器123456...')
        self.out_src_sent, self.out_tgt_sent = self.get_dataset(data_path, sort=True)
        self.sp_src = XLNET_chinese_tokenizer_load()
        self.sp_tgt = malaysian_tokenizer_load()
        self.PAD = self.sp_tgt.pad_id()  # 0
        self.BOS = self.sp_tgt.bos_id()  # 2
        self.EOS = self.sp_tgt.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        out_src_sent = []
        out_tgt_sent = []
        with open(data_path, "r", encoding="utf-8") as file:
            corpus = file.readlines()
        for item in corpus:
            item = item.replace('\n', '').strip()
            arr = item.split('\t')
            if len(arr) != 2:
                continue
            out_src_sent.append(arr[0])
            out_tgt_sent.append(arr[1])
        if sort:
            sorted_index = self.len_argsort(out_tgt_sent)
            out_src_sent = [out_src_sent[i] for i in sorted_index]
            out_tgt_sent = [out_tgt_sent[i] for i in sorted_index]
        return out_src_sent, out_tgt_sent

    def __getitem__(self, idx):
        src_text = self.out_src_sent[idx]
        tgt_text = self.out_tgt_sent[idx]
        return [src_text, tgt_text]

    def __len__(self):
        return len(self.out_src_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_len = [len(x) for x in src_text]
        max_src_len = max(src_len) + 2

        # src_tokens = [[self.BOS] + self.sp_src.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        batch_input = self.sp_src(src_text,
                                  padding='max_length',  # Pad to max_length
                                  truncation=True,  # Truncate to max_length
                                  max_length=max_src_len,
                                  return_tensors='pt')  # Return torch.Tensor objects

        tgt_tokens = [[self.BOS] + self.sp_tgt.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
        #                            batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        if config.TRAN_REVERSE:
            return Batch(tgt_text, src_text, batch_target, batch_input, self.PAD)
        else:
            # 中翻译马
            return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD, src_is_zh=True)