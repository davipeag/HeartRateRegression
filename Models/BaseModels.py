import torch
from torch import nn
from typing import List, Tuple, Sequence
import copy

class HalveConvolution1D(nn.Module):
    def __init__(self, input_length, input_channels, output_channels,
                 approximate_down=True):
        super(HalveConvolution1D, self).__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.output_channels = output_channels
        self.approximate_down = approximate_down

        if approximate_down:
            padding = 0
        else:
            padding = 1

        if (input_length > 2):
            kernel_size = 3
            stride = 2
            if (input_length % 2) == 0:
                padding = 1
            else:
                padding = padding

        elif (input_length > 1):
            kernel_size = 2
            stride = 2
            padding = 0
        else:
            kernel_size = 1
            stride = 1
            padding = 0

        self.conv = nn.Conv1d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        return self.conv(x)


class ConstantHiddenSizeHalvingFullyConvolutionalEncoder1D(nn.Module):

    def __init__(self,
                 input_length,
                 input_channels,
                 output_channels,
                 hidden_size,
                 dropout_rate=0,
                 activation_function=nn.LeakyReLU()
                 ):
        super(ConstantHiddenSizeHalvingFullyConvolutionalEncoder1D,
              self).__init__()

        self.input_length = input_length
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        lengths = [input_length, *self.__fully_halve(input_length)]
        channels = [self.hidden_size for _ in lengths]
        channels[0] = self.input_channels
        channels[-1] = self.output_channels
        dropout_rates = [dropout_rate for _ in range(len(channels)-1)]
        dropout_rates[0] = 0

        cell_inps = zip(lengths[0:-1], channels[0:-1],
                        channels[1:], dropout_rates)

        self.cells = [self.make_cell(*cell_inp)
                      for cell_inp in cell_inps]

        self.encoder = nn.Sequential(*self.cells)

    def forward(self, x):
        return self.encoder(x)

    def make_cell(self, input_length, input_channels, output_channels, dropout_rate=0):
        activation = self.activation_function
        conv = HalveConvolution1D(input_length, input_channels,
                                  output_channels, approximate_down=True)
        dropout = nn.Dropout(dropout_rate)

        return nn.Sequential(conv, dropout, activation)

    def get_parameters(self):
        return {
            "input_length": self.input_length,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "hidden_size": self.hidden_size,
            "activation_function_name": self.activation_function_name,
            "activation_function_parameters": self.activation_function_parameters,
            "batch_norm": self.batch_norm
        }

    def input_shape(self):
        raise NotImplementedError()

    def output_shape(self, input_shape):
        raise NotImplementedError()

    def __fully_halve(self, length):
        if length > 1:
            half = length//2
            return [half, *self.__fully_halve(half)]
        else:
            return []


class LinearCell(nn.Module):

    def __init__(self, input_size, output_size, dropout_rate, activation_function=nn.LeakyReLU()):
        super(LinearCell, self).__init__()

        self.cell = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(dropout_rate),
            activation_function
        )

    def forward(self, x):
        return self.cell(x)
    


class SkipFullyConnected(nn.Module):
    def __init__(
        self, input_features: int, output_features:int, layer_sizes: List[int], skip_mapping: List[Tuple[int, int]],
        dropout_rate: float, activation_function = nn.LeakyReLU()):
        """
        skip_mapping List(Tuple(src, dst))
        """

        imp_layer_sizes_tmp = [input_features, *layer_sizes]
        self.check_skip_mapping(skip_mapping, imp_layer_sizes_tmp)
        imp_layer_sizes = copy.deepcopy(imp_layer_sizes_tmp)
        for src,dst in skip_mapping:
            imp_layer_sizes[dst] = imp_layer_sizes[dst] + imp_layer_sizes[src]
        out_layer_sizes = [*imp_layer_sizes_tmp[1:], output_features]

        imp_outs = list(zip(imp_layer_sizes, out_layer_sizes))
        self.layers = nn.ModuleList()
        for i,o in imp_outs[:-1]:
            self.layers.append(LinearCell(i,o, dropout_rate, activation_function))
        self.layers.append(LinearCell(imp_outs[-1][0], imp_outs[-1][1], 0, nn.Identity()))

        self.concat_indices = [[i] for i in range(imp_layer_sizes)]
        for src,dst in skip_mapping:
            self.concat_indices[dst].append(src)


    def forward(self, x):
        imps = [None for _ in self.concat_indices]
        for i,layer in enumerate(self.layers):
            imps[i] = x
            x = layer(torch.cat([imps[j] for j in self.concat_indices[i]], dim=-1))
        return x

    
    def normalize_index(self, idx, size):
        if (idx > 0) and (idx < size):
            return idx
        if (idx < 0) and (idx >= -size):
            return size - idx
        else:
            raise IndexError(f"Index ({idx}) out of Range({size})")

    def check_skip_mapping(self, skip_mapping, inp_layer_sizes):
        size = len(inp_layer_sizes)
        for src, dst in skip_mapping:
            dst = self.normalize_index(dst, size) #max(dst, len(inp_layer_sizes)-dst)
            src = self.normalize_index(src, size) #max(src, len(inp_layer_sizes)-src)
            if ((dst) < src + 1 ):
                raise ValueError(f"dest({dst}) < src({src}) + 2")            
            
           




