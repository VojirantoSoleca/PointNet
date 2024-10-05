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

class PointNetfeat(nn.Layer):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()

        self.stn = STN3d()
        if feature_transform:
            self.fstn = STNkd(k=64)

        self.conv1 = nn.Conv1D(3, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
    
    def forward(self, x):
        n_pts = x.shape[2]
        trans = self.stn(x)

        x = x.transpose([0, 2, 1])
        x = paddle.bmm(x, trans)
        x = x.transpose([0, 2, 1])
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose([0, 2, 1])
            x = paddle.bmm(x, trans_feat)
            x = x.transpose([0, 2, 1])
        else:
            trans_feat = None
        
        pointfeat = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape([-1, 1024])

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.reshape([-1, 1024, 1]).tile([1, 1, n_pts])
            return paddle.concat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn. Layer):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()

        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.dropout = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, axis=1), trans, trans_feat

class PointNetDenseCls(nn.Layer):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()

        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)

        self.conv1 = nn.Conv1D(1088, 512, 1)
        self.conv2 = nn.Conv1D(512, 256, 1)
        self.conv3 = nn.Conv1D(256, 128, 1)
        self.conv4 = nn.Conv1D(128, self.k, 1)

        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.bn3 = nn.BatchNorm1D(128)
    
    def forward(self, x):
        batchsize = x.shape[0]
        n_pts = x.shape[2]
        x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose([0, 2, 1])
        x = F.log_softmax(x.reshape([-1, self.k]), axis = -1)
        x = x.reshape([batchsize, n_pts, self.k])

        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.shape[1]
    batchsize = trans.shape[0]
    I = paddle.unsqueeze(paddle.eye(d), 0)

    loss = paddle.mean(paddle.norm(paddle.bmm(trans, trans.transpose([0, 2, 1])) - I, axis=[1, 2]))

    return loss

if __name__ == '__main__':
	sim_data = paddle.uniform([32, 3, 2500])
	trans = STN3d()
	out = trans(sim_data)
	print('stn', out.shape)
	print('loss', feature_transform_regularizer(out))
	
	sim_data_64d = paddle.uniform([32, 64, 2500])
	trans = STNkd(k=64)
	out = trans(sim_data_64d)
	print('stn64d', out.shape)
	print('loss', feature_transform_regularizer(out))
	
	pointfeat = PointNetfeat(global_feat=True)
	out, _, _ = pointfeat(sim_data)
	print('global feat', out.shape)
	
	pointfeat = PointNetfeat(global_feat=False)
	out, _, _ = pointfeat(sim_data)
	print('point feat', out.shape)
	
	cls = PointNetCls(k=5)
	out, _, _ = cls(sim_data)
	print('class', out.shape)
	
	seg = PointNetDenseCls(k=3)
	out, _, _ = seg(sim_data)
	print('seg', out.shape)
