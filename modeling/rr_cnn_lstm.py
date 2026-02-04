# modeling/rr_cnn_lstm.py

import torch
import torch.nn as nn

class RRCNNLSTM(nn.Module):
    def __init__(self, seq_len):
        super().__init__()

        # CNN over RR intervals
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )

        # Classifier
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        """
        x shape: (batch, seq_len)
        """

        # (batch, 1, seq_len)
        x = x.unsqueeze(1)

        # CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # (batch, channels, time) -> (batch, time, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        _, (h_n, _) = self.lstm(x)

        # Last hidden state
        h_last = h_n[-1]

        # Output
        out = self.fc(h_last).squeeze(1)

        return out
