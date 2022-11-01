import torch
import time

d_model = 768
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 5
bos_idx = 3
eos_idx = 4
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 64
epoch_num = 60
early_stop = 10
lr = 8e-5

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 4
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = False

TRAN_ORI = 'zh2ms'  # zh2en、en2ms、zh2ms
TRAN_REVERSE = False  # 是否翻转翻译方向
train_start = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
data_dir = './minidata_' + TRAN_ORI
train_data_path = data_dir + '/mini_train.txt'
dev_data_path = data_dir + '/mini_dev.txt'
test_data_path = data_dir + '/mini_dev.txt'

tempStr = ''
if TRAN_REVERSE:
    tempStr = ''
model_path = './experimentT/' + TRAN_ORI + tempStr + '/model.pth'
log_path = './experimentT/' + TRAN_ORI + tempStr + '/train' + train_start + '.log'
output_path = './experimentT/' + TRAN_ORI + tempStr + '/output' + train_start + '.txt'

PRE_TRAIN_MODEL_NAME = 'bert-base-chinese'
#PRE_TRAIN_MODEL_NAME = '../xlnet_pretrain_base'
# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')