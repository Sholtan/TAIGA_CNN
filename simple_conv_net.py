print("simple_conv_net module imported\n")


import torch
import torch.nn as nn
import torch.nn.functional as tfunctional

class SimpleConvNet(nn.Module):
    def __init__(self):
        # вызов конструктора предка
        super(SimpleConvNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("CNN will run on device:", self.device)

        # необходмо заранее знать, сколько каналов у картинки (сейчас = 1),
        # которую будем подавать в сеть, больше ничего
        # про входящие картинки знать не нужно
        # 27*27
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        # 23*23
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 12*12
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=15, kernel_size=5)
        # 12
        self.fc1 = nn.Linear(9 * 9 * 15, 500)  # !!!
        self.fc2 = nn.Linear(500, 120)
        self.fc3 = nn.Linear(120, 1)

        self.to(self.device)
        self.my_loss_fn = nn.BCELoss()
        self.my_learning_rate = 1e-4
        self.my_optimizer = torch.optim.Adam(self.parameters(), lr=self.my_learning_rate)


    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        # 27*27
        x = tfunctional.relu(self.conv1(x))
        # 23*23
        x = self.pool(tfunctional.relu(self.conv2(x)))
        # 19*19 -> 10*10
        #print("x.shape", x.shape)
        #input("wait...")
        x = x.view(-1, 9 * 9 * 15)  # !!!
        x = tfunctional.relu(self.fc1(x))
        x = tfunctional.relu(self.fc2(x))
        x = tfunctional.sigmoid(self.fc3(x))
        #x = F.sigmoid(self.fc4(x))
        return x



