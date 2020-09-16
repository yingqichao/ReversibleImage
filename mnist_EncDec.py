import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 512

# 标准化
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def to_img(x):
    x = 0.5 * (x + 1.)  # 将-1~1转成0-1
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b,16,10,10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b,16,5,5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b,8,3,3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b,8,2,2
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1),  # b,8,3,3
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 16, 4, stride=4, padding=1),  # b,16,10,10
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 1, 3, stride=3, padding=1),  # b,1,28,28
            # nn.Tanh()
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b,16,5,5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b,8,15,15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b,1,28,28
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


model = AutoEncoder().to(device)
# 定义loss函数和优化方法
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
for t in range(40):
    for data in train_data:
        img, label = data
        img = img.to(device)
        label = label.to(device)
        _, output = model(img)
        loss = loss_fn(output, img) / img.shape[0]  # 平均损失
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (t + 1) % 5 == 0:  # 每 5 次，保存一下解码的图片和原图片
        print('epoch: {}, Loss: {:.4f}'.format(t + 1, loss.item()))
        pic = to_img(output.cpu().data)
        if not os.path.exists('./conv_autoencoder'):
            os.mkdir('./conv_autoencoder')
        save_image(pic, './conv_autoencoder/decode_image_{}.png'.format(t + 1))
        save_image(img, './conv_autoencoder/raw_image_{}.png'.format(t + 1))