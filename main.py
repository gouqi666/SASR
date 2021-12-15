from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import numpy as np

from LM.dataset import LMDataSet
from LM.model import Transformer
from LM.trainer import LMTrainer
from utils import lm_token_path, am_token_path, TextFeaturizer
from pypinyin import pinyin

#
LMTrainer().train(False)


# print(LMTrainer().predict([item[0] for item in pinyin("郭俊杰")]))
