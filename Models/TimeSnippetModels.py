import torch
from torch import nn


class DeepConvLSTM(nn.Module):
    def __init__(self, ts_per_is, feature_count, recursive_size, frequency_hz, step_seconds):
        super(DeepConvLSTM, self).__init__()

        samples_per_ts = frequency_hz * step_seconds

        #last sample of each time snippet in the prediction segment (recursive size)
        self.mask = [samples_per_ts*ts_per_is + (i+1) * samples_per_ts - 1 
                     for i in range(recursive_size)]

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, (5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, (5, 1), padding=(2, 0)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(64*feature_count, 128,
                            batch_first=True, num_layers=2, dropout=0.5)

        self.lin = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        l = self.lstm(torch.flatten(
            self.conv(x).transpose(2, 1), start_dim=2))[0]
        return self.lin(l[:, self.mask, :])
