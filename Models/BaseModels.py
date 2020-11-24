import torch
from torch import nn

class HalveConvolution1D(nn.Module):
  def __init__(self, input_length, input_channels, output_channels,
               approximate_down = True):
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
      stride=2
      if (input_length % 2) == 0:
        padding = 1
      else:
        padding = padding

    elif (input_length>1):
      kernel_size=2
      stride=2
      padding=0
    else:
      kernel_size=1
      stride=1
      padding=0

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
                    activation_function =  nn.LeakyReLU()
               ):
    super(ConstantHiddenSizeHalvingFullyConvolutionalEncoder1D,
          self).__init__() 
    

    self.input_length = input_length
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.hidden_size = hidden_size
    self.activation_function = activation_function

    lengths = [input_length, *self.__fully_halve(input_length)]
    channels = [self.hidden_size for _ in lengths]
    channels[0] = self.input_channels
    channels[-1] = self.output_channels
    
    cell_inps = zip(lengths[0:-1], channels[0:-1], channels[1:])
    
    self.cells = [self.make_cell(*cell_inp)
                  for cell_inp in cell_inps]
    
    self.encoder = nn.Sequential(*self.cells)  

  def forward(self, x):
    return self.encoder(x)

  def make_cell(self, input_length, input_channels, output_channels):      
      activation = self.activation_function
      conv = HalveConvolution1D(input_length, input_channels,
                                output_channels, approximate_down=True)

      return nn.Sequential(conv, norm, activation)

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