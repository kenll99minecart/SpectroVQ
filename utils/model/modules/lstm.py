# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, biLSTM:bool = False):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers) if not biLSTM else nn.LSTM(dimension, dimension, num_layers, bidirectional=True)
        self.biLSTM = biLSTM
        # if biLSTM:
        #     self.linear = nn.Sequential(
        #         nn.Linear(dimension, dimension),
        #         nn.ELU(),
        #     )
    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            if self.biLSTM:
                y = y[:,:,:int(y.shape[2]/2)] + y[:,:,int(y.shape[2]/2):]
                # y = self.linear(y)
            y = y + x
        y = y.permute(1, 2, 0)
        return y

