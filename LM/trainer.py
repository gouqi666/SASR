import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel
from transformers.models.marian.convert_marian_to_pytorch import remove_prefix
from .loss import BertFeatureLoss, LMCrossEntropyLoss, LMAccuracy
from .bert import bert_model, bert_tokenizer
from .dataset import LMDataSet
from utils import lm_token_path, am_token_path, TextFeaturizer
from .config import training_config
from torch.nn.utils.rnn import pad_sequence
from .model import Transformer
import numpy as np
from torch.nn import CrossEntropyLoss

am_features = TextFeaturizer(am_token_path)
lm_features = TextFeaturizer(lm_token_path)


def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data[0]), reverse=True)
    x = [torch.Tensor(item[0]) for item in train_data]
    y = [torch.Tensor(item[1]) for item in train_data]
    bert_token = [torch.Tensor(item[2]) for item in train_data]
    x = pad_sequence(x, batch_first=True, padding_value=0).int()
    y = pad_sequence(y, batch_first=True, padding_value=0).int()
    bert_token = pad_sequence(bert_token, batch_first=True, padding_value=0).int()
    mask = torch.eq(x, 0)
    bert_feature = bert_model.forward(bert_token.cuda(), attention_mask=torch.logical_not(mask).int().cuda())
    train_data = (x, y, bert_feature[0])
    return train_data, mask


class LMTrainer:
    def __init__(self):
        self.amf = TextFeaturizer(am_token_path)
        self.lmf = TextFeaturizer(lm_token_path)
        self.train_data = DataLoader(LMDataSet(self.amf, self.lmf, True),
                                     batch_size=training_config["batch_size"],
                                     shuffle=True, collate_fn=collate_fn)
        self.valid_data = DataLoader(LMDataSet(self.amf, self.lmf, False),
                                     batch_size=training_config["batch_size"],
                                     shuffle=True, collate_fn=collate_fn)
        self.model = Transformer(self.amf, self.lmf)
        self.save_path = training_config["save_path"]
        self.epoch = training_config["epoch"]

    def train(self, resume=True):
        # print(torch.cuda.current_device())
        with torch.cuda.device("cuda:0"):
            self.model = self.model.cuda()

            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)

            scheduler = StepLR(optimizer, step_size=25, gamma=0.8)

            bert_feature_loss = BertFeatureLoss()
            cross_entropy_loss = LMCrossEntropyLoss()
            metrics = LMAccuracy()

            if resume:  # restore from checkpoint
                self.model, optimizer, epoch = self.restore_from(self.model, optimizer, self.save_path)

            for epoch in range(self.epoch):

                train_loss = []
                self.model.train()
                print("epoch:{}".format(epoch))
                with tqdm(total=len(self.train_data)) as bar:
                    for i, batch in enumerate(self.train_data):
                        (x, y, bert_feature), mask = batch
                        x, y, bert_feature, mask = x.cuda(), y.cuda(), bert_feature.cuda(), mask.cuda()
                        optimizer.zero_grad()
                        output_classes, features = self.model(x, mask)
                        class_loss = cross_entropy_loss(output_classes, y, mask)
                        acc = metrics(output_classes, y, mask)
                        feature_loss = bert_feature_loss(features, bert_feature, mask)
                        loss = class_loss + feature_loss
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss)
                        bar.set_postfix(train_loss=loss, acc=acc)
                        bar.update(1)
                    train_loss = np.mean(train_loss)

                valid_loss = []
                self.model.eval()  # 注意model的模式从train()变成了eval()
                for i, batch in enumerate(tqdm(self.valid_data)):
                    (x, y, bert_feature), mask = batch
                    x, y, bert_feature, mask = x.cuda(), y.cuda(), bert_feature.cuda(), mask.cuda()
                    # output_classes, features = self.model(x.cuda(), mask.cuda())
                    output_classes, features = self.model(x, mask)
                    class_loss = cross_entropy_loss(output_classes, y, mask)
                    feature_loss = bert_feature_loss(features, bert_feature, mask)
                    loss = class_loss + feature_loss
                    valid_loss.append(loss)
                valid_loss = np.mean(valid_loss)

                scheduler.step()

                print("valid loss: {}".format(valid_loss))

                if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epoch:  # 保存模型
                    torch.save(
                        {'epoch': epoch,
                         'state_dict': self.model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        self.save_path)

    @staticmethod
    def restore_from(model, optimizer, ckpt_path):
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
        epoch = ckpt['epoch']
        ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
        model.load_state_dict(ckpt_model_dict, strict=False)  # load model
        optimizer.load_state_dict(ckpt['optimizer'])  # load optimizer

        return model, optimizer, epoch
