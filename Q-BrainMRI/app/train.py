import torch
from torch.utils.data import DataLoader
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from tqdm import tqdm
import logging
from datetime import datetime


def train_network(net=None, train_set=None, val_set=None, device=None,
                  epochs=10, bs=64, optimizer=None, criterion=None, file_prefix=None):
    # 获取当前日期和时间
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # 配置日志
    logging.basicConfig(
        filename=f"{file_prefix}_training_{start_time}.log",  # 日志文件名
        level=logging.INFO,  # 日志级别
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)

    batch_avg_time = []
    tr_losses = []
    val_losses = []
    tr_accs = []
    val_accs = []
    best_val_acc = float(-0.1)

    for epoch in range(1, epochs + 1):
        t1 = time.time()
        net.train()
        tr_loss = 0

        y_trues = []
        y_preds = []

        for i, (inputs, labels) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit="batch", leave=False)):
            t2 = time.time()
            data = inputs.type(torch.FloatTensor).to(device)
            y = labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.data.cpu().numpy()
            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()
            batch_avg_time.append(time.time() - t2)

        tr_acc = accuracy_score(y_trues, y_preds)
        tr_loss = tr_loss / (i + 1)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)

        cnf = confusion_matrix(y_trues, y_preds)
        logging.info(f"Epoch:{epoch}, TR_Loss: {tr_loss:.4f}, TR_Acc: {tr_acc:.4f}, Confusion Matrix:\n{cnf}")

        net.eval()
        val_loss = 0
        y_trues = []
        y_preds = []

        for i, (inputs, labels) in enumerate(val_loader):
            data = inputs.type(torch.FloatTensor).to(device)
            y = labels.type(torch.LongTensor).to(device)

            with torch.no_grad():
                output = net(data)

            loss = criterion(output, y)
            val_loss += loss.data.cpu().numpy()
            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()

        val_acc = accuracy_score(y_trues, y_preds)
        val_loss = val_loss / (i + 1)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        cnf = confusion_matrix(y_trues, y_preds)
        logging.info(f"Epoch:{epoch}, VAL_Loss: {val_loss:.4f}, VAL_Acc: {val_acc:.4f}, Confusion Matrix:\n{cnf}")

        # 保存最佳模型
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), f"{file_prefix}-best_model.pth")
            logging.info(f"Epoch:{epoch} - New Best Model Saved with VAL_Acc: {val_acc:.4f}")

        logging.info(f"Time for Epoch {epoch}: {time.time() - t1:.2f} seconds")
    # 获取当前日期和时间
    end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # save model and results
    # os.makedirs(outdiroutdir, exist_ok = True)
    # torch.save(net.state_dict(), outdir + '/' + file_prefix + '_model')
    # np.save(outdir + '/' + file_prefix + '_training_loss.npy', tr_losses)
    # np.save(outdir + '/' + file_prefix + '_validation_loss.npy', val_losses)
    # np.save(outdir + '/' + file_prefix + '_training_accuracy.npy', tr_accs)
    # np.save(outdir + '/' + file_prefix + '_validation_accuracy.npy', val_accs)
    # 打开文件，如果文件不存在会创建一个新文件
    with open('{}_{}.txt'.format(file_prefix, end_time), 'w') as f:
        # 将内容写入文件
        f.write('_batch_avg_time{}\n'.format(batch_avg_time))
        f.write('_training_loss{}\n'.format(tr_losses))
        f.write('_validation_loss{}\n'.format(val_losses))
        f.write('_training_accuracy{}\n'.format(tr_accs))
        f.write('_validation_accuracy{}\n'.format(val_accs))
