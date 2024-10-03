import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


RAND_SEED = 0
LEARNING_RATE = 0.001
MODEL_PATH = 'MLP_LSTM_MLP.pth'
train_audio_feature_path = r'E:/Orignal_F/PyCharm/pythonProject3/VGGish_Audio_Feature/train_audio_features'
test_audio_feature_path = r'E:/Orignal_F/PyCharm/pythonProject3/VGGish_Audio_Feature/test_audio_features'
train_label_path = r'E:/Orignal_F/PyCharm/pythonProject3/labels/train_labels_4s_8_32000'
test_label_path = r'E:/Orignal_F/PyCharm/pythonProject3/labels/test_labels_4s_8_32000'
target_path = 'target_results'
prediction_path = 'prediction_results'
batch_size = 4
base_path = ''
epochs = 100
all_test_accuracy_max = 0
all_test_accuracy_max_epoch = 0

all_losses = []
all_errors = []
all_test_accuracy = []


def fix_random_seeds(specified_seed):
    random.seed(specified_seed)
    np.random.seed(seed=specified_seed)
    torch.manual_seed(seed=specified_seed)
    torch.cuda.manual_seed_all(seed=specified_seed)


fix_random_seeds(RAND_SEED)


class Train_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_pth, label=None, test=False, audio_feature_pth='', label_path=''):
        self.audio_feature_pth = os.path.join(root_pth, audio_feature_pth)
        self.train_label_pth = os.path.join(root_pth, label_path)
        # self.video_feature_pth = os.path.join(root_pth, label_path)
        self.label = label
        self.is_test = test

        if label is None:
            label = []
            annotation_paths = [os.path.join(self.train_label_pth, path) for path in
                                sorted(os.listdir(self.train_label_pth))]
            for i, path in enumerate(annotation_paths):
                tmp_audio_annotation = np.load(path)  # (8,)
                tmp_audio_annotation = [x / 100 for x in tmp_audio_annotation]
                label.append(tmp_audio_annotation)
            self.label = label  # (sample_index, 8)

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

        audio_data = np.load(os.path.join(self.audio_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
                             allow_pickle=True)
        # audio_data = audio_data.transpose(1, 0)
        # video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        # video_data = -1/np.log(video_data)
        MAX = np.max(audio_data)
        MIN = np.min(audio_data)
        audio_data = (audio_data - MIN) / (MAX - MIN)
        # audio_data = audio_data.reshape(8, -1)
        # video_data = np.expand_dims(video_data, axis=1)
        # data_combine = np.concatenate((audio_data, video_data), axis=1)  # (8, 6156)
        # data_combine = torch.from_numpy(data_combine.astype(np.float32))
        # video_data = torch.from_numpy(video_data.astype(np.float32))
        audio_data = torch.from_numpy(audio_data.astype(np.float32))
        return audio_data, lbl


class Test_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_pth, label=None, test=False, audio_feature_pth='', label_path=''):
        self.audio_feature_pth = os.path.join(root_pth, audio_feature_pth)
        self.test_label_pth = os.path.join(root_pth, label_path)
        # self.video_feature_pth = os.path.join(root_pth, label_path)
        self.label = label
        self.is_test = test

        if label is None:
            label = []
            annotation_paths = [os.path.join(self.test_label_pth, path) for path in
                                sorted(os.listdir(self.test_label_pth))]
            for i, path in enumerate(annotation_paths):
                tmp_audio_annotation = np.load(path)  # (8,)
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

        audio_data = np.load(os.path.join(self.audio_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
                             allow_pickle=True)
        # audio_data = audio_data.transpose(1, 0)
        # video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        # video_data = 1/np.log(video_data)
        # # print(video_data)
        MAX = np.max(audio_data)
        MIN = np.min(audio_data)
        audio_data = (audio_data - MIN) / (MAX - MIN)
        # audio_data = audio_data.reshape(8, -1)
        # video_data = np.expand_dims(video_data, axis=1)
        # data_combine = np.concatenate((audio_data, video_data), axis=1)  # (4, 256)
        # data_combine = torch.from_numpy(data_combine.astype(np.float32))
        # video_data = torch.from_numpy(video_data.astype(np.float32))
        audio_data = torch.from_numpy(audio_data.astype(np.float32))
        return audio_data, lbl


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.n_directions = 2 if bidirectional else 1
        self.n_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, dropout=0.2,
                            batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_layer_size * self.n_directions, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.hidden_dim = hidden_layer_size
        self.fc = nn.Sequential(
            nn.Linear(6144, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def init_hidden(self, batch_size):
        device = torch.device("cuda")
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * self.n_directions, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers * self.n_directions, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

    def forward(self, x, hidden):
        x = self.fc(x)
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.linear(lstm_out)
        predictions = self.relu(predictions)
        return predictions, hidden


train_set = Train_DataSet(base_path, audio_feature_pth=train_audio_feature_path, label_path=train_label_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_set = Test_DataSet(base_path, audio_feature_pth=test_audio_feature_path, label_path=test_label_path)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

model = LSTM(input_size=128, hidden_layer_size=1024, output_size=1, num_layers=2, bidirectional=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_error = 0.0
    all_pred = []
    all_target = []
    # train
    model.train()
    hidden = model.init_hidden(batch_size)
    for batch_idx, data in enumerate(train_loader, 0):
        hidden = tuple([e.data for e in hidden])
        inputs, target = data
        target = target.float()
        target = target.unsqueeze(dim=2)
        inputs, target = inputs.to(device), target.to(device)
        # print('input', inputs.shape)
        #         # print(type(inputs))
        # print('target', target.shape)
        #         # print(type(target))
        #         # print(target)
        optimizer.zero_grad()
        #         # print(target.view([-1, 1]))
        y_pred, hidden = model.forward(inputs, hidden)  # y_pred = (4, 8, 1)
        # y_pred, hidden = model.forward(inputs)  # y_pred = (4, 8, 1)
        # print('pred', y_pred.shape)

        # calculate training MAE
        for j in range(batch_size):
            for i in range(8):
                epoch_error += (abs(y_pred[j][i][0].item() - target[j][i][0].item()))
                all_target.append(target[j][i][0].item())
                all_pred.append(y_pred[j][i][0].item())
        single_loss = loss_function(y_pred, target)
        single_loss.backward()
        optimizer.step()
        epoch_loss += single_loss.item()
    all_errors.append(abs(epoch_error / batch_size / batch_idx / 8))
    all_losses.append(epoch_loss / batch_idx)
    print('epoch_error:', epoch_error / batch_size / batch_idx / 8)
    print('[epochs: %d] MSE_loss: %.3f' % (epoch + 1, epoch_loss / batch_idx))

    all_pred = np.array(all_pred)
    all_target = np.array(all_target)
    print('average_Mae_loss:', abs((all_pred - all_target)).sum() / len(all_pred))
    print('5% accuracy: ', ((abs((all_pred - all_target))) <= 0.05).sum() / len(all_pred) * 100, '%')
    print('10% accuracy: ', ((abs((all_pred - all_target))) <= 0.1).sum() / len(all_pred) * 100, '%')
    print('15% accuracy: ', ((abs((all_pred - all_target))) <= 0.15).sum() / len(all_pred) * 100, '%')

    # test
    model.eval()
    all_target = []
    all_prediction = []
    correct_5 = 0
    correct_10 = 0
    correct_15 = 0
    total = 0
    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        for data in test_loader:
            hidden = tuple([e.data for e in hidden])
            inputs, target = data
            target = target.float()
            inputs, target = inputs.to(device), target.to(device)
            target = target.unsqueeze(dim=2)
            outputs, hidden = model(inputs, hidden)
            all_target.append(target.squeeze().cpu().numpy().copy())
            all_prediction.append(outputs.squeeze().cpu().numpy().copy())
            # outputs, hidden = model(inputs)
            # print('preds\n', outputs)
            # print('target\n', target)
            for i in range(batch_size):
                tmp_target = target[i].cpu().numpy()  # (8, 1)
                tmp_outputs = outputs[i].cpu().numpy()
                total += len(tmp_outputs)
                diff = abs(tmp_outputs - tmp_target)  # (8, 1)
                correct_5 += (diff <= 0.05).sum().item()
                correct_10 += (diff <= 0.1).sum().item()
                correct_15 += (diff <= 0.15).sum().item()

    all_test_accuracy.append(100 * correct_5 / total)
    if all_test_accuracy_max < 100 * correct_5 / total:
        torch.save(model.state_dict(), MODEL_PATH)
        all_prediction = np.array(all_prediction)
        all_target = np.array(all_target)
        print(all_prediction.shape)
        print(all_target.shape)
        np.save(target_path, all_target)
        np.save(prediction_path, all_prediction)
        all_test_accuracy_max_epoch = epoch + 1
        all_test_accuracy_max = 100 * correct_5 / total
    print('5_Accuracy on test set: %.3f %%' % (100 * correct_5 / total))
    print('10_Accuracy on test set: %.3f %%' % (100 * correct_10 / total))
    print('15_Accuracy on test set: %.3f %%' % (100 * correct_15 / total))



plt.figure()
plt.plot(all_losses)
plt.title('VGGish_Baseline')
plt.xlabel('Training Epochs')
plt.ylabel('Training MSE Loss')

plt.figure()
plt.plot(all_errors)
plt.title('VGGish_Baseline')
plt.xlabel('Training Epochs')
plt.ylabel('Training MAE Loss')
plt.show()

plt.figure()
plt.plot(all_test_accuracy)
plt.title('VGGish_Test_accuracy')
plt.xlabel('Training Epochs')
plt.ylabel('Training Test accuracy')
plt.show()
print("max 5%_test_accuracy corresponding epoch: ")
print(all_test_accuracy_max_epoch)
print("corresponding accuracy: ")
print(all_test_accuracy_max)