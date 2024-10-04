import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class STN3d(nn.Layer):
    def __init__(self):
        super(STN3d, self).__init__()

        self.conv1 = nn.Conv1D(3, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

    def forward(self, x):
        batchsize = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x))) # [32, 64, 2500]
        x = F.relu(self.bn2(self.conv2(x))) # [32, 128, 2500]
        x = F.relu(self.bn3(self.conv3(x))) # [32, 1024, 2500]

        x = paddle.max(x, 2, keepdim=True) # [32, 1024, 1]
        x = x.reshape([-1, 1024]) # [32, 1024]

        x = F.relu(self.bn4(self.fc1(x))) # [32, 512]
        x = F.relu(self.bn5(self.fc2(x))) # [32, 256]
        x = self.fc3(x) # [32, 9]

        iden = paddle.to_tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='float32').reshape([1, 9]).tile([batchsize, 1])
        
        x += iden
        x = x.reshape([-1, 3, 3])

        return x

class STNkd(nn.Layer):
    def __init__(self, k=64):
        super(STNkd, self).__init__()

        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape([-1, 1024])

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = paddle.eye(self.k, dtype='float32').flatten().reshape([1, self.k * self.k]).tile([batchsize, 1])

        x += iden
        x = x.reshape([-1, self.k, self.k])

        return x

if __name__ == '__main__':
	sim_data = paddle.uniform([32, 3, 2500])
	trans = STN3d()
	out = trans(sim_data)
	print('stn', out.shape)
	
	sim_data_64d = paddle.uniform([32, 64, 2500])
	trans = STNkd(k=64)
	out = trans(sim_data_64d)
	print('stn64d', out.shape)
	
