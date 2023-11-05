import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ConvLSTM was first introduced for forecasting into this paper https://arxiv.org/pdf/1506.04214.pdf
Another really common network architecture used for cell segmentation is U-net : https://arxiv.org/pdf/1505.04597.pdf
In this paper https://arxiv.org/pdf/1805.11247v2.pdf, one tries to mixed both CLSTM and U-net to perform cell videos 
segmentation, the idea here is to reproduce their encoder to see if it can perform well. The only 
difference is that I use 2 times (ConvLayer + BatchNorm + LeakyRelu) for a Block2D instead of 1

There are no native implementation for ConvLSTM in Pytorch so the one below was essentially taken from this blog post : 
https://towardsdatascience.com/video-prediction-using-convlstm-with-pytorch-lightning-27b195fd21a2
"""


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class BlockConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConv2D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CellEncoder(nn.Module):
    def __init__(self, sequence_length, features, in_channels=1, num_classes=8):
        super(CellEncoder, self).__init__()
        self.L = len(features)
        self.convLSTM = nn.ModuleList()
        self.conv = nn.ModuleList()

        for feature in features:
            self.convLSTM.append(ConvLSTM(input_dim=in_channels, hidden_dim=feature))
            self.conv.append(BlockConv2D(feature, feature))
            in_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(
            in_features=features[-1] * sequence_length, out_features=num_classes
        )

    def forward(self, x):
        for i in range(self.L):
            out, _ = self.convLSTM[i](x)
            x = F.relu(out[0])

            orig_shape = x.shape

            y = torch.reshape(
                x,
                [
                    orig_shape[0] * orig_shape[1],
                    orig_shape[2],
                    orig_shape[3],
                    orig_shape[4],
                ],
            )
            y = self.conv[i](y)
            y = self.pool(y)

            if i < (self.L - 1):
                x = torch.reshape(
                    y,
                    [orig_shape[0], orig_shape[1], y.shape[1], y.shape[2], y.shape[3]],
                )
            else:
                y = self.avgpool(y)
                x = torch.reshape(
                    y,
                    [orig_shape[0], orig_shape[1], y.shape[1], y.shape[2], y.shape[3]],
                )
                x = torch.flatten(x, start_dim=1)
                x = self.fc(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand((15, 10, 8, 160, 160))
    convlstm = ConvLSTM(
        input_dim=8,
        hidden_dim=16,
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    )
    _, last_states = convlstm(x)
    h = last_states[0][0]  # 0 for layer index, 0 for h index
    assert h.size() == (15, 16, 160, 160)

    sequence_length = 10
    features = [8, 16, 32, 64]
    x = torch.rand((15, sequence_length, 1, 160, 160))
    model = CellEncoder(sequence_length=sequence_length, features=features)
    out = model(x)
    assert out.shape == (15, 8)
