import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import pandas as pd
from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext import data
from torchtext.vocab import Vectors, GloVe

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

CLASS_NUM = len(ag_news_label)

class FastText(nn.Module):
    def __init__(self, vocab, vec_dim, label_size, hidden_size):
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(len(vocab), vec_dim)
        # 若使用预训练的词向量，需在此处指定预训练的权重
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, label_size)
        )

    def forward(self, x):
        x = self.embed(x)
        out = self.fc(torch.mean(x, dim=1))
        return out

def model_test(net, test_iter, batch_size):
    net.eval()  # 必备，将模型设置为训练模式
    correct = torch.zeros(CLASS_NUM)
    total = torch.zeros(CLASS_NUM)
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
            data, label = batch.text, batch.label
            data, label = data.to(device), label.to(device)
            logging.info("test batch_id=" + str(i))
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            predicted += 1

            for i in range(1,5):
                filter = (torch.ones(label.size(0))*i).to(device)
                filtered_label = ((label == filter).int()).to(device)
                total[i-1] += filtered_label.sum().item()
                correct[i-1] += ((filtered_label*i) == predicted).int().sum().item()

        totoal_accuracy = round(correct.sum().item()/total.sum().item(),3)
        print('Accuracy of the network on test set:()',totoal_accuracy)
        accuray = (correct.float() / total.float()).numpy()
        accuray = np.round(accuray,3)
        accuray = dict(zip(ag_news_label.values(), accuray.tolist()))
        print(accuray)

        return totoal_accuracy

def train_model(net, train_iter, test_iter, epoch, lr, batch_size):
    print("begin training")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for i in range(epoch):  # 多批次循环
        net.to(device)
        net.train()  # 必备，将模型设置为训练模式
        for batch_idx, batch in enumerate(train_iter):
            # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
            data, target = batch.text, batch.label - 1
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清除所有优化的梯度
            output = net(data)  # 传入数据并前向传播获取输出
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            logging.info("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / batch_size))

        acc = model_test(net, test_iter, batch_size)
        if acc > best_acc:
            torch.save({'state_dict': net.cpu().state_dict()}, "saved_model/ag_fasttext_model.pth.tar")
            best_acc = acc

    print('Finished Training')

def get_data_iter(train_csv, test_csv, fix_length):
    TEXT = data.Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=batch_size, device=-1, sort_key=lambda x: len(x.text),
                                sort_within_batch=False, repeat=False)
    test_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    test = TabularDataset(path=test_csv, format="csv", fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)

    # vectors = Vectors(name=word2vec_dir)
    TEXT.build_vocab(train, vectors=GloVe(name='6B',dim=300))
    vocab = TEXT.vocab
    return train_iter, test_iter, vocab

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

def predict(text, model, vocab, sentence_max_size, ngrams=1):
    model.to(device)
    model.eval()

    tokenizer = get_tokenizer("basic_english")
    tokenized_text = tokenizer(text)
    # pad = ['[PAD]' for i in range(sentence_max_size-len(tokenized_text))]
    # tokenized_text.extend(pad)

    with torch.no_grad():
        text_tensor = torch.tensor([vocab[token]
                             for token in ngrams_iterator(tokenized_text, ngrams)])
        text_tensor = torch.stack([text_tensor],0).to(device)
        output = model(text_tensor)
        label = output.argmax(1).item() + 1

        print(output)
        print("Classification:"+ag_news_label[label])

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    train_csv = "/home/dengerqiang/Documents/WORK/DataSets/ag_news_csv/train.csv"
    test_csv = "/home/dengerqiang/Documents/WORK/DataSets/ag_news_csv/test.csv"
    word2vec_dir = "glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
    sentence_max_size = 100  # 每篇文章的最大词数量
    net_dir = "saved_model/ag_fasttext_model.pth.tar"
    batch_size = 64
    epoch = 10  # 迭代次数
    emb_dim = 300  # 词向量维度
    lr = 0.001
    hidden_size = 200
    label_size = 4
    device = torch.device('cuda:1')

    train_iter, test_iter, vocab = get_data_iter(train_csv, test_csv, sentence_max_size)
    # 定义模型
    net = FastText(vocab=vocab, vec_dim=emb_dim, label_size=label_size, hidden_size=hidden_size)

    if False:
        # 训练
        logging.info("开始训练模型")
        train_model(net, train_iter, test_iter, epoch, lr, batch_size)
        # 保存模型

        # torch.save({'state_dict': net.cpu().state_dict()}, net_dir)
        # logging.info("开始测试模型")
        # model_test(net, test_iter, batch_size)
    else:
        model_dict = torch.load(net_dir)
        net.load_state_dict(model_dict['state_dict'])

        net.to(device)
        # model_test(net, test_iter, batch_size)

        ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
            enduring the season’s worst weather conditions on Sunday at The \
            Open on his way to a closing 75 at Royal Portrush, which \
            considering the wind and the rain was a respectable showing."

        # text_tensor = tokenize(ex_text_str, sentence_max_size, vocab)

        predict(ex_text_str, net, vocab, sentence_max_size)
