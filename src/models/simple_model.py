import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self, features, in_channels=1):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.convs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.avgpool(x)

        return x


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        sequence_length,
        device,
        hidden_size=50,
        num_layers=2,
        num_classes=8,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(
            in_features=hidden_size * sequence_length, out_features=num_classes
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        x, _ = self.lstm(x, (h0, c0))

        x0 = x.shape[0]
        x = x.contiguous().view(x0, -1)
        x = self.fc(x)

        return x


class SimpleNetwork(nn.Module):
    def __init__(self, features, input_size, sequence_length, device):
        super(SimpleNetwork, self).__init__()
        self.CNN = CNN(features=features)
        self.LSTM = LSTM(input_size, sequence_length, device)

    def forward(self, i):
        # Shape : Batch_size * Seq_length * Channel * W * L
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
        x = self.CNN(x)
        x = x.view(i.shape[0], i.shape[1], -1)

        # Shape : Batch_size * Seq_length * Features
        x = self.LSTM(x)

        return x


if __name__ == "__main__":
    batch_size = 12

    features = [
        8,
        16,
        32,
        64,
    ]  # features=[64, 128, 256, 512] increasing sequences of channel sizes
    in_channels, width, height = 1, 160, 160

    input_size = features[-1] * 2
    sequence_length = 40
    device = torch.device("cpu")

    x = torch.rand((batch_size, sequence_length, in_channels, width, height))
    model = SimpleNetwork(features, input_size, sequence_length, device)
    out = model(x)

    print(x.shape)
    print(out.shape)
