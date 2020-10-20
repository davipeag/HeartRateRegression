from torch import nn
import torch


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            for attr_name in [a for a in dir(m) if "weight" in a]:
                attr = getattr(m, attr_name)
                try:
                    try:
                        nn.init.orthogonal_(attr)
                    except ValueError:
                        nn.init.normal_(attr)
                except (AttributeError, ValueError):
                    pass
            for attr_name in [a for a in dir(m) if "bias" in a]:
                attr = getattr(m, attr_name)
                try:
                    nn.init.constant_(attr, 0)
                except (AttributeError, ValueError):
                    pass
    return model


class SnippetConvolutionalTransformer(nn.Module):
    def __init__(
            self,
            nfeatures=4,
            conv_filters=64,
            nconv_layers = 4,
            conv_dropout=0.5, 
            nenc_layers=2,
            ndec_layers=2,
            nhead=4,
            feedforward_expansion=2,
            nlin_layers = 2,
            lin_size = 32,
            lin_dropout = 0
            ):

        super(SnippetConvolutionalTransformer, self).__init__()

        t_size = nfeatures*conv_filters  # transformer input size

        self.transformer = nn.Transformer(
            t_size, nhead=nhead, num_encoder_layers=nenc_layers,
            num_decoder_layers=ndec_layers,
            dim_feedforward=int(feedforward_expansion*t_size))

        self.conv_net = self.make_conv_net(1, conv_filters, conv_filters, nconv_layers, conv_dropout)

        self.regressor = self.make_lin_net(t_size, lin_size, 1, nlin_layers, lin_dropout)
    
    def make_conv_layer(self, input_channels, output_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (5,1), padding=(2,0)),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate)
        )
    
    def make_lin_layer(self, input_size, output_size, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def make_conv_net(self, input_channels, hidden_channels, output_channels, nlayers, dropout_rate):
        if nlayers == 0:
            return nn.Identity()
        elif nlayers == 1:
            return self.make_conv_layer(input_channels, output_channels, dropout_rate)
        return nn.Sequential(
            self.make_conv_layer(input_channels, hidden_channels, dropout_rate),
            *[
                self.make_conv_layer(hidden_channels, hidden_channels, dropout_rate)
                for _ in range(nlayers - 2)
            ],
            self.make_conv_layer(hidden_channels, output_channels, dropout_rate)
        )
    
    def make_lin_net(self, input_channels, hidden_channels, output_channels, nlayers, dropout_rate):
        if nlayers == 0:
            return nn.Identity()
        elif nlayers == 1:
            return self.make_lin_layer(input_channels, output_channels, dropout_rate)
        return nn.Sequential(
            self.make_lin_layer(input_channels, hidden_channels, dropout_rate),
            *[
                self.make_lin_layer(hidden_channels, hidden_channels, dropout_rate)
                for _ in range(nlayers - 2)
            ],
            self.make_lin_layer(hidden_channels, output_channels, dropout_rate)
        )
    

    def forward(self, x):
        xr = x[:, :, :, 1:] ## disconsider heart rate, therefore begins from 1
        c = torch.flatten(self.conv_net(xr).transpose(1, 2), 2).transpose(0, 1)
        cd = c[-1:]

        t_o = self.transformer(c, cd)
        return self.regressor(t_o)[-1]



