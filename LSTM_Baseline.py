import os
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Train_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_pth, label=None, test=False, transform=None, padding_size=100):
        class_num = 1
        self.audio_feature_pth = os.path.join(root_pth, 'MFCC_Feature/feature_mfcc_train')
        self.train_label_pth = os.path.join(root_pth, 'marker_distance/label_train')
        self.video_feature_pth = os.path.join(root_pth, 'marker_distance/feature_train')
        self.label = label  # gt['filling_level'].to_numpy()
        self.is_test = test
        self.each_class_size = []
        self.each_class_sum = [0] * class_num
        for i in range(class_num):
            self.each_class_size.append(np.count_nonzero(self.label == i))
        mx = 0
        mn = 1000
        len_mx = 0

        if label is None:
            label = []
            annotation_paths = [os.path.join(self.train_label_pth, path) for path in
                                sorted(os.listdir(self.train_label_pth))]
            for i, path in enumerate(annotation_paths):
                tmp_audio_annotation = np.load(path)  # (4,)
                tmp_audio_annotation = [x / 100 for x in tmp_audio_annotation]
                label.append(tmp_audio_annotation)
            self.label = label  # (sample_index, 4)

        # for idx in range(len(os.listdir(self.mid_pth))):
        #     data = np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx) + '.npy'), allow_pickle=True)
        #     # self.each_class_sum[self.label[idx]]+=data.shape[0]
        #     if data.shape[0] > len_mx:
        #         len_mx = data.shape[0]
        #     tmp_max = np.max(data)
        #     tmp_min = np.min(data)
        #     if mx < tmp_max:
        #         mx = tmp_max
        #     if mn > tmp_min:
        #         mn = tmp_min
        # self.mn = mn
        # self.mx = mx
        # self.pad = Padding(padding_size)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
            lbl = np.array(lbl)
            lbl = torch.from_numpy(lbl.astype(np.float32))

        # audio_data = np.load(os.path.join(self.audio_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
                             allow_pickle=True)
        # video_data = torch.from_numpy(video_data)  # (48, 256)
        # video_data = video_data.unsqueeze(0)
        # conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 5), stride=(2, 2))
        # video_data = conv1(video_data)
        # video_data = video_data.view(4, -1)
        # linear1 = torch.nn.Linear(video_data.shape[1], 128)
        # video_data = linear1(video_data)
        # video_data = video_data.detach().numpy()
        # data_combine = np.concatenate((audio_data, video_data), axis=1)  # (4, 256)
        # data_combine = torch.from_numpy(data_combine.astype(np.float32))
        video_data = torch.from_numpy(video_data.astype(np.float32))
        # audio_data = torch.from_numpy(audio_data.astype(np.float32))
        return video_data, lbl

    def get_each_class_size(self):
        return np.array(self.each_class_size)

    def get_each_class_avg_len(self):
        each_class_avg_len = np.array(self.each_class_sum) / np.array(self.each_class_size)
        all_class_avg_len = np.sum(np.array(self.each_class_sum)) / np.sum(np.array(self.each_class_size))
        return each_class_avg_len, all_class_avg_len


class Test_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_pth, label=None, test=False, transform=None, padding_size=100):
        class_num = 1
        self.audio_feature_pth = os.path.join(root_pth, 'MFCC_Feature/feature_mfcc_test')
        self.test_label_pth = os.path.join(root_pth, 'marker_distance/label_test')
        self.video_feature_pth = os.path.join(root_pth, 'marker_distance/feature_test')
        self.label = label  # gt['filling_level'].to_numpy()
        self.is_test = test

        if label is None:
            label = []
            annotation_paths = [os.path.join(self.test_label_pth, path) for path in
                                sorted(os.listdir(self.test_label_pth))]
            for i, path in enumerate(annotation_paths):
                tmp_audio_annotation = np.load(path)  # (4,)
                tmp_audio_annotation = [x / 100 for x in tmp_audio_annotation]
                label.append(tmp_audio_annotation)
            self.label = label

    def __len__(self):
        return len(self.label)  # 103

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
            lbl = np.array(lbl)
            lbl = torch.from_numpy(lbl.astype(np.float32))

        # audio_data = np.load(os.path.join(self.audio_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
                             allow_pickle=True)
        # video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        # video_data = torch.from_numpy(video_data)
        # video_data = video_data.unsqueeze(0)
        # conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 5), stride=(2, 2))
        # video_data = conv1(video_data)
        # video_data = video_data.view(4, -1)
        # linear1 = torch.nn.Linear(video_data.shape[1], 128)
        # video_data = linear1(video_data)
        # video_data = video_data.detach().numpy()
        # data_combine = np.concatenate((audio_data, video_data), axis=1)  # (4, 256)
        # data_combine = torch.from_numpy(data_combine.astype(np.float32))
        video_data = torch.from_numpy(video_data.astype(np.float32))
        # audio_data = torch.from_numpy(audio_data.astype(np.float32))
        return video_data, lbl


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.n_directions = 2 if bidirectional else 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(11088, 1024),
            # nn.Dropout(0.2),
            # nn.Linear(4096, 1024)
        )

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # self.linear = nn.Linear(hidden_layer_size * self.n_directions, output_size)
        # self.relu = nn.ReLU()
        self.MLP3 = nn.Sequential(
            nn.Linear(hidden_layer_size * self.n_directions, output_size),
            nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(1024, output_size),
            # nn.ReLU()
        )

        self.hidden_cell = (
            torch.zeros(num_layers * self.n_directions, 4, self.hidden_layer_size).to(torch.device("cuda:0")),
            torch.zeros(num_layers * self.n_directions, 4, self.hidden_layer_size).to(torch.device("cuda:0")))

    def forward(self, input_seq):
        input_seq = input_seq.unsqueeze(dim=1)  # [4, 1, 48, 256]
        input_seq = self.conv1(input_seq)
        input_seq = input_seq.view(4, -1)
        # print(input_seq.shape)
        input_seq = self.linear(input_seq)
        input_seq = input_seq.view(4, 4, -1)
        lstm_out, hidden_state = self.lstm(input_seq, self.hidden_cell)
        # predictions = self.linear(lstm_out)
        predictions = self.MLP3(lstm_out)
        return predictions, hidden_state


batch_size = 4
base_path = ''
train_set = Train_DataSet(base_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_set = Test_DataSet(base_path)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

model = LSTM(input_size=256, hidden_layer_size=512, output_size=1, num_layers=2, bidirectional=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 100
all_test_accuracy_max = 0
all_test_accuracy_max_epoch = 0

all_losses = []
all_errors = []
all_test_accuracy = []
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_error = 0.0
    all_pred = []
    all_target = []
    model.train()
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        target = target.float()
        # target = target.unsqueeze(dim=2)
        inputs, target = inputs.to(device), target.to(device)
        # print('input', inputs.shape)
        #         # print(type(inputs))
        # print('555', target.shape)
        #         # print(type(target))
        #         # print(target)
        optimizer.zero_grad()
        #         # print(target.view([-1, 1]))
        y_pred, hidden = model.forward(inputs)  # y_pred = (4, 4, 1)
        # print('pred', y_pred.shape)

        for j in range(4):
            for i in range(4):
                epoch_error += (abs(y_pred[j][i][0].item() - target[j][i][0].item()))
                all_target.append(target[j][i][0].item())
                all_pred.append(y_pred[j][i][0].item())
        single_loss = loss_function(y_pred, target)
        # single_loss = single_loss.float()
        single_loss.backward()
        optimizer.step()
        epoch_loss += single_loss.item()
    all_errors.append(abs(epoch_error / 4 / batch_idx / 4))
    all_losses.append(epoch_loss / batch_idx)
    print('epoch_error:', epoch_error / 4 / batch_idx / 4)
    print('[epochs: %d] MSE_loss: %.3f' % (epoch + 1, epoch_loss / batch_idx))

    all_pred = np.array(all_pred)
    all_target = np.array(all_target)
    print('average_Mae_loss:', abs((all_pred - all_target)).sum() / len(all_pred))
    print('5% accuracy: ', ((abs((all_pred - all_target))) <= 0.05).sum() / len(all_pred) * 100, '%')
    print('10% accuracy: ', ((abs((all_pred - all_target))) <= 0.1).sum() / len(all_pred) * 100, '%')
    print('15% accuracy: ', ((abs((all_pred - all_target))) <= 0.15).sum() / len(all_pred) * 100, '%')

    model.eval()
    correct_5 = 0
    correct_10 = 0
    correct_15 = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            target = target.float()
            inputs, target = inputs.to(device), target.to(device)
            # target = target.unsqueeze(dim=2)
            outputs, hidden = model(inputs)
            for i in range(4):
                tmp_target = target[i].cpu().numpy()  # (4, 1)
                tmp_outputs = outputs[i].cpu().numpy()
                total += len(tmp_outputs)
                diff = abs(tmp_outputs - tmp_target)  # (4, 1)
                correct_5 += (diff <= 0.05).sum().item()
                correct_10 += (diff <= 0.1).sum().item()
                correct_15 += (diff <= 0.15).sum().item()

    all_test_accuracy.append(100 * correct_5 / total)
    if all_test_accuracy_max < 100 * correct_5 / total:
        all_test_accuracy_max_epoch = epoch + 1
        all_test_accuracy_max = 100 * correct_5 / total
    print('5_Accuracy on test set: %d %%' % (100 * correct_5 / total))
    print('10_Accuracy on test set: %d %%' % (100 * correct_10 / total))
    print('15_Accuracy on test set: %d %%' % (100 * correct_15 / total))

plt.figure()
plt.plot(all_losses)
plt.title('LSTM_Baseline')
plt.xlabel('Training Epoches')
plt.ylabel('Training MSE Loss')

plt.figure()
plt.plot(all_errors)
plt.title('LSTM_Baseline')
plt.xlabel('Training Epoches')
plt.ylabel('Training MAE Loss')
plt.show()

plt.figure()
plt.plot(all_test_accuracy)
plt.title('LSTM_Test_accuracy')
plt.xlabel('Training Epoches')
plt.ylabel('Training Test accuracy')
plt.show()
print("最大5%_test_auucuracy的训练轮数是：")
print(all_test_accuracy_max_epoch + 1)
print("对应最大值是：")
print(all_test_accuracy_max)
