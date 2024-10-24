import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from tqdm import tqdm


def train_network(net=None, train_set=None, val_set=None, device=None,
                  epochs=10, bs=20, optimizer=None, criterion=None, file_prefix = None):  # outdir = None, file_prefix = None):

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)

    batch_avg_time=[]
    tr_losses = []
    val_losses = []
    tr_accs = []
    val_accs = []
    # 定义一个变量来保存最佳验证集上的准确率
    best_val_acc = float(-0.1)

    for epoch in range(1, epochs + 1):
        t1 = time.time()
        net.train()
        tr_loss = 0

        y_trues = []
        y_preds = []

        for i, sampled_batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False)):
            t2 = time.time()

            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)

            data = data.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(data)
            output = output.to(device)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            # print(net.ql1.weights.grad)
            tr_loss += loss.data.cpu().numpy()

            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()
            print('batch({})-time:{:.2f} seconds'.format(i, time.time() - t2))
            batch_avg_time.append(time.time() - t2)

        tr_acc = accuracy_score(y_trues, y_preds)
        tr_accs.append(tr_acc)
        tr_loss = tr_loss / (i + 1)
        tr_losses.append(tr_loss)

        cnf = confusion_matrix(y_trues, y_preds)
        print(cnf)

        print('Epoch:{}, TR_Loss: {:.4f}, TR_Acc: {:.4f}'.format(epoch, tr_loss, tr_acc))

        net.eval()
        val_loss = 0

        y_trues = []
        y_preds = []

        for i, sampled_batch in enumerate(val_loader):
            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)

            data = data.to(device)
            y = y.to(device)

            with torch.no_grad():
                output = net(data)

            loss = criterion(output, y)
            val_loss = val_loss + loss.data.cpu().numpy()

            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()

        val_acc = accuracy_score(y_trues, y_preds)
        val_accs.append(val_acc)
        val_loss = val_loss / (i + 1)
        val_losses.append(val_loss)

        cnf = confusion_matrix(y_trues, y_preds)
        print(cnf)
        # 保存最佳模型
        print('judge best_val_acc')
        if best_val_acc < val_acc:
            print('save model')
            best_val_acc = val_acc
            torch.save(net.state_dict(), '{}-best_model.pth'.format(file_prefix))
        print('Epoch: {} VAL_Loss: {:.4f}, VAL_Acc: {:.4f}'.format(epoch, val_loss, val_acc))
        print('Time for Epoch ({}): {:.2f}seconds'.format(epoch, time.time() - t1))

    # save model and results
    # os.makedirs(outdiroutdir, exist_ok = True)
    # torch.save(net.state_dict(), outdir + '/' + file_prefix + '_model')
    # np.save(outdir + '/' + file_prefix + '_training_loss.npy', tr_losses)
    # np.save(outdir + '/' + file_prefix + '_validation_loss.npy', val_losses)
    # np.save(outdir + '/' + file_prefix + '_training_accuracy.npy', tr_accs)
    # np.save(outdir + '/' + file_prefix + '_validation_accuracy.npy', val_accs)
    # 打开文件，如果文件不存在会创建一个新文件
    with open('{}.txt'.format(file_prefix), 'w') as f:
        # 将内容写入文件
        f.write('_batch_avg_time{}\n'.format(batch_avg_time))
        f.write('_training_loss{}\n'.format(tr_losses))
        f.write('_validation_loss{}\n'.format(val_losses))
        f.write('_training_accuracy{}\n'.format(tr_accs))
        f.write('_validation_accuracy{}\n'.format(val_accs))
