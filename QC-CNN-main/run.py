import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from app.load_data import MyCSVDatasetReader as CSVDataset
from models.classical import Net as Net2
from models.hybrid_layer import Net as Net3
# from models.inception import Net as Net1
from models.inception_copy import Net as Net1
from models.single_encoding import Net as Net4
# from models.multi_encoding import Net
# from models.multi_noisy import Net
from app.train import train_network

# load the dataset
dataset = CSVDataset('./datasets/mnist_179_1200.csv')
# output location/file names
# outdir = 'results_255_tr_mnist358'
# file_prefix = 'mnist_358'

# load the device
# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
net1 = Net1()
# net2 = Net2()
# net3 = Net3()
# net4 = Net4()
net1.to(device)
# prefix2 = 'classical--'
# prefix3 = 'hybrid_layer--'
prefix1 = 'inception--default.qubit.torch--best(delete)'
# prefix1 = 'inception--default.qubit.torch--best(50 epochs Adagrad lr 0.5 without quantum layer)'
# prefix4 = 'single_encoding--'

epochs = 200
bs = 32

criterion = nn.CrossEntropyLoss()  # loss function
optimizer1 = torch.optim.Adagrad(net1.parameters(), lr=0.02)  # optimizer
# optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.02)  # optimizer
# optimizer2 = torch.optim.Adagrad(net2.parameters(), lr=0.5)  # optimizer
# optimizer3 = torch.optim.Adagrad(net3.parameters(), lr=0.5)  # optimizer
# optimizer4 = torch.optim.Adagrad(net4.parameters(), lr=0.5)  # optimizer
train_id, val_id = train_test_split(list(range(len(dataset))), test_size=0.3, random_state=0)
train_set = Subset(dataset, train_id)
val_set = Subset(dataset, val_id)

train_network(net=net1, train_set=train_set, val_set=val_set, device=device,
              epochs=epochs, bs=bs, optimizer=optimizer1,
              criterion=criterion, file_prefix=prefix1)  # outdir = outdir, file_prefix = file_prefix)
# train_network(net=net2, train_set=train_set, val_set=val_set, device=device,
#               epochs=epochs, bs=bs, optimizer=optimizer2,
#               criterion=criterion, file_prefix=prefix2)  # outdir = outdir, file_prefix = file_prefix)
# train_network(net=net3, train_set=train_set, val_set=val_set, device=device,
#               epochs=epochs, bs=bs, optimizer=optimizer3,
#               criterion=criterion, file_prefix=prefix3)  # outdir = outdir, file_prefix = file_prefix)
# train_network(net=net4, train_set=train_set, val_set=val_set, device=device,
#               epochs=epochs, bs=bs, optimizer=optimizer4,
#               criterion=criterion, file_prefix=prefix4)  # outdir = outdir, file_prefix = file_prefix)




