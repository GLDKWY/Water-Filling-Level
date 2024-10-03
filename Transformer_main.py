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
        self.audio_feature_pth = os.path.join(root_pth, 'train_audio_features_4s_8_32000')
        self.train_label_pth = os.path.join(root_pth, 'train_labels_4s_8_32000')
        self.video_feature_pth = os.path.join(root_pth, 'marker_distance/feature_train')
        self.label = label  # gt['filling_level'].to_numpy()
        self.is_test = test

        if label is None:
            label = []
            annotation_paths = [os.path.join(self.train_label_pth, path) for path in
                                sorted(os.listdir(self.train_label_pth))]
            for i, path in enumerate(annotation_paths):
                tmp_audio_annotation = np.load(path)  # (4,)
                tmp_audio_annotation = [x / 100 for x in tmp_audio_annotation]
                label.append(tmp_audio_annotation)
            self.label = label  # (sample_index, 4)

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
        # video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        # video_data = -1/np.log(video_data)
        # MAX = np.max(video_data)
        # MIN = np.min(video_data)
        # video_data = (video_data - MIN) / (MAX - MIN)
        # audio_data = audio_data.reshape(8, -1)
        # video_data = np.expand_dims(video_data, axis=1)
        # data_combine = np.concatenate((audio_data, video_data), axis=1)  # (8, 6156)
        # data_combine = torch.from_numpy(data_combine.astype(np.float32))
        # video_data = torch.from_numpy(video_data.astype(np.float32))
        zero_pad = np.zeros(128)
        two_pad = 2 * np.ones(128)
        zero_pad = np.expand_dims(zero_pad, axis=0)
        two_pad = np.expand_dims(two_pad, axis=0)
        audio_data_pad_zero = np.concatenate((zero_pad, audio_data))
        audio_data_pad_two = np.concatenate((audio_data_pad_zero, two_pad))
        # MAX_ZERO = np.max(audio_data_pad_zero)
        # MIN_ZERO = np.min(audio_data_pad_zero)
        # MAX = np.max(audio_data)
        # MIN = np.min(audio_data)
        # audio_data_pad_zero = (audio_data_pad_zero - MIN_ZERO) / (MAX_ZERO - MIN_ZERO)
        # audio_data = (audio_data - MIN) / (MAX - MIN)
        audio_data_pad_zero = torch.from_numpy(audio_data_pad_zero.astype(np.float32))
        audio_data_pad_two = torch.from_numpy(audio_data_pad_two.astype(np.float32))
        audio_data = torch.from_numpy(audio_data.astype(np.float32))
        return audio_data_pad_zero, audio_data, lbl


class Test_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_pth, label=None, test=False, transform=None, padding_size=100):
        class_num = 1
        self.audio_feature_pth = os.path.join(root_pth, 'test_audio_features_4s_8_32000')
        self.test_label_pth = os.path.join(root_pth, 'test_labels_4s_8_32000')
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

        audio_data = np.load(os.path.join(self.audio_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
                             allow_pickle=True)
        # video_data = np.load(os.path.join(self.video_feature_pth, "{0:05d}".format(idx + 1) + '.npy'),
        #                      allow_pickle=True)
        # video_data = 1/np.log(video_data)
        # # print(video_data)
        # MAX = np.max(video_data)
        # MIN = np.min(video_data)
        # video_data = (video_data - MIN) / (MAX - MIN)
        # audio_data = audio_data.reshape(8, -1)
        # video_data = np.expand_dims(video_data, axis=1)
        # data_combine = np.concatenate((audio_data, video_data), axis=1)  # (4, 256)
        # data_combine = torch.from_numpy(data_combine.astype(np.float32))
        # video_data = torch.from_numpy(video_data.astype(np.float32))
        zero_pad = np.zeros(128)
        two_pad = 2 * np.ones(128)
        zero_pad = np.expand_dims(zero_pad, axis=0)
        two_pad = np.expand_dims(two_pad, axis=0)
        audio_data_pad_zero = np.concatenate((zero_pad, audio_data))
        audio_data_pad_two = np.concatenate((audio_data_pad_zero, two_pad))
        # MAX_ZERO = np.max(audio_data_pad_zero)
        # MIN_ZERO = np.min(audio_data_pad_zero)
        # MAX = np.max(audio_data)
        # MIN = np.min(audio_data)
        # audio_data_pad_zero = (audio_data_pad_zero - MIN_ZERO) / (MAX_ZERO - MIN_ZERO)
        # audio_data = (audio_data - MIN) / (MAX - MIN)
        audio_data_pad_zero = torch.from_numpy(audio_data_pad_zero.astype(np.float32))
        audio_data_pad_two = torch.from_numpy(audio_data_pad_two.astype(np.float32))
        audio_data = torch.from_numpy(audio_data.astype(np.float32))
        return audio_data_pad_zero, audio_data, lbl


max_length = 8


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512,
                                          batch_first=True, dropout=0.2)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        self.predictor = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.linear2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.MLP3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 1),
        )

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2])
        tgt_mask = tgt_mask.to(device)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.MLP3(out)
        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out


batch_size = 4
base_path = ''
train_set = Train_DataSet(base_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_set = Test_DataSet(base_path)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

model = Transformer()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

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
        src, tgt, target = data
        target = target.float()
        target = target.unsqueeze(dim=2)
        src, tgt, target = src.to(device), tgt.to(device), target.to(device)
        # print('input', inputs.shape)
        #         # print(type(inputs))
        # print('555', target.shape)
        #         # print(type(target))
        #         # print(target)
        optimizer.zero_grad()
        #         # print(target.view([-1, 1]))
        y_pred = model.forward(src, tgt)  # y_pred = (4, 4, 1)
        # print('pred', y_pred.shape)

        for j in range(batch_size):
            for i in range(8):
                epoch_error += (abs(y_pred[j][i][0].item() - target[j][i][0].item()))
                all_target.append(target[j][i][0].item())
                all_pred.append(y_pred[j][i][0].item())
        single_loss = loss_function(y_pred, target)
        # single_loss = single_loss.float()
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

    model.eval()
    correct_5 = 0
    correct_10 = 0
    correct_15 = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            src, tgt, target = data
            target = target.float()
            src, tgt, target = src.to(device), tgt.to(device), target.to(device)
            target = target.unsqueeze(dim=2)
            outputs = model(src, tgt)
            # print('preds\n', outputs)
            # print('target\n', target)
            for i in range(batch_size):
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
print(all_test_accuracy_max_epoch)
print("对应最大值是：")
print(all_test_accuracy_max)
