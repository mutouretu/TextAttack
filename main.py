import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

import pandas as pd
from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext import data
from torchtext.vocab import Vectors, GloVe

from FastText import FastText, get_model

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

CLASS_NUM = len(ag_news_label)
device = torch.device("cuda:1")

def model_test(net, test_iter):
    net.eval()  # 必备，将模型设置为训练模式
    correct = torch.zeros(CLASS_NUM)
    total = torch.zeros(CLASS_NUM)
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
            data, label = batch.text, batch.label
            data, label = data.to(device), label.to(device)
            logging.info("test batch_id=" + str(i))
            data = net.embed(data)
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



def predict(text, model, vocab, ngrams=1):
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

        text_tensor = model.embed(text_tensor)

        output = model(text_tensor)
        label = output.argmax(1).item() + 1

        print(output)
        print("Classification:"+ag_news_label[label])

        return label

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    if False:
        # 训练
        logging.info("开始训练模型")
        train_model(net, train_iter, test_iter, epoch, lr, batch_size)
        # 保存模型

        # torch.save({'state_dict': net.cpu().state_dict()}, net_dir)
        # logging.info("开始测试模型")
        # model_test(net, test_iter, batch_size)
    else:
        model,_,test_iter = get_model(True)


        model.to(device)
        model_test(model, test_iter)

        ex_text_str = "WASHINGTON, Aug. 19 (Xinhuanet) -- Andre Agassi cruised into quarter-finals in Washington Open tennis with a 6-4, 6-2 victory over Kristian Pless of Denmark here on Thursday night."

        # text_tensor = tokenize(ex_text_str, sentence_max_size, vocab)

        predict(ex_text_str, model, model.vocab)
