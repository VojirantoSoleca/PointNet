import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import argparse
import os
import random
import paddle.optimizer as optim
import paddle.io
from paddle.io import DataLoader
from tqdm import tqdm

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

class ModelNetDataset(io.Dataset):
    def __init__(self, root, npoints=2500, split='modelnet40_train', data_augmentatiion=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentatiion

        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())
        
        self.cat = {}
        with open('/home/aistudio/work/modelnet.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])
        print(self.cat)
        self.classes = list(self.cat.keys())
        print(self.classes)
    
    def __getitem__(self, index):
        fn = self.fns[index]
        cls = fn[:-5]

        plyData = [[], [], []]
        with open(os.path.join(self.root, cls + '/' + fn + '.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                temp = line.split(',')
                plyData[0].append(float(temp[0]))
                plyData[0].append(float(temp[3]))
                plyData[1].append(float(temp[1]))
                plyData[1].append(float(temp[4]))
                plyData[2].append(float(temp[2]))
                plyData[2].append(float(temp[5]))

        pts = np.vstack([plyData[0], plyData[1], plyData[2]]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)
        
        point_set = paddle.to_tensor(point_set.astype(np.float32))
        cls = paddle.to_tensor(self.cat[cls]).astype(np.int64)

        return point_set, cls
    
    def __len__(self):
        return len(self.fns)

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'modelnet40_train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip()[:-5])
    classes = np.unique(classes)
    with open('/home/aistudio/work/modelnet.txt', 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

if __name__ == '__main__':
    args = {
    'batchSize': 32,
    'num_points': 2500,
    'workers': 4,
    'nepoch': 250,
    'outf': 'cls',
    'model': '',
    'dataset': '/home/aistudio/data/data72849/modelnet40_normal_resampled',
    'dataset_type': 'shapenet',
    'feature_transform': False
}

    opt = argparse.Namespace(**args)

    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    paddle.seed(opt.manualSeed)

    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='modelnet40_train')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='modelnet40_test',
        npoints=opt.num_points)

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.set_state_dict(paddle.load(opt.model))
        classifier.eval()

    optimizer = optim.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, parameters=classifier.parameters())
    scheduler = optim.lr.StepDecay(learning_rate=0.001, step_size=20, gamma=0.5)
    # classifier = paddle.to_tensor(classifier)


    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose([0, 2, 1])
            points, target = paddle.to_tensor(points), paddle.to_tensor(target)
            optimizer.clear_grad()
            classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.cross_entropy(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = paddle.argmax(pred, axis=1)
            correct = paddle.equal(pred_choice, target).sum().numpy().item()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.numpy().item(), correct / float(opt.batchSize)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose([0, 2, 1])
                points, target = paddle.to_tensor(points), paddle.to_tensor(target)
                classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.cross_entropy(pred, target)
                pred_choice = paddle.argmax(pred, axis=1)
                correct = paddle.equal(pred_choice, target).sum().numpy().item()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'test', loss.numpy().item(), correct / float(opt.batchSize)))

        paddle.save(classifier.state_dict(), '%s/cls_model_%d.pdparams' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose([0, 2, 1])
        points, target = paddle.to_tensor(points), paddle.to_tensor(target)
        classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = paddle.argmax(pred, axis=1)
        correct = paddle.equal(pred_choice, target).sum().numpy().item()
        total_correct += correct
        total_testset += points.shape[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
    
