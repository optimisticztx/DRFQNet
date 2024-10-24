import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from app.load_data import MyCSVDatasetReader as CSVDataset
from sklearn.metrics import  accuracy_score

from models.inception_copy import Net as Net


# load the dataset
dataset = CSVDataset('./datasets/mnist_179_1200.csv')
# output location/file names
# outdir = 'results_255_tr_mnist358'
# file_prefix = 'mnist_358'

# load the device
# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
net = Net()
net.load_state_dict(torch.load('inception--default.qubit.torch--best-best_model.pth'))
net.to(device)

criterion = nn.CrossEntropyLoss()  # loss function

_, val_id = train_test_split(list(range(len(dataset))), test_size=0.9, random_state=0)
val_set = Subset(dataset, val_id)
bs = 32


def test_network(net=None, test_loader=None, criterion=None, device=None, bs = None):
    net.eval()
    val_loss = 0
    test_loader = DataLoader(test_loader, batch_size=bs, shuffle=True)

    y_trues = []
    y_preds = []
    start_time = time.time()

    with torch.no_grad(), tqdm(total=len(test_loader)) as progress_bar:
        for i, sampled_batch in enumerate(test_loader):
            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)

            data = data.to(device)
            y = y.to(device)

            output = net(data)

            y_trues += y.cpu().numpy().tolist()
            y_preds += output.data.cpu().numpy().argmax(axis=1).tolist()

            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f'Testing: {i+1}/{len(test_loader)}')

    val_acc = accuracy_score(y_trues, y_preds)
    print('Test Accuracy: {:.2f}%'.format(val_acc))
    print('Total test time: {:.2f} seconds'.format(time.time()-start_time))


test_network(net=net, test_loader=val_set,
             criterion=criterion, device=device, bs=32)  # outdir = outdir, file_prefix = file_prefix)

