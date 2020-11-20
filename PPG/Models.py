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
        
        if nconv_layers == 0:
            t_size = nfeatures
        else:
            t_size = nfeatures*conv_filters  # transformer input size

        self.transformer = nn.Transformer(
            t_size, nhead=nhead, num_encoder_layers=nenc_layers,
            num_decoder_layers=ndec_layers,
            dim_feedforward=int(feedforward_expansion*t_size))

        self.conv_net = self.make_conv_net(1, conv_filters, conv_filters, nconv_layers, conv_dropout)

        self.regressor = self.make_lin_net(t_size, lin_size, 1, nlin_layers, lin_dropout)

        initialize_weights(self)
    
    def make_conv_layer(self, input_channels, output_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (5,1)),#, padding=(2,0)),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
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


class ConvTransfRNN(nn.Module):
  def __init__(self, input_channels, bvp_idx, nfilters=64, dropout_rate=0.1, embedding_size=128,
               bvp_embedding_size=12, predictor_hidden_size=32, feedforward_expansion=2,
               num_encoder_layers= 2, num_decoder_layers=2,
               nheads=4):
    super(ConvTransfRNN, self).__init__()
    if isinstance(bvp_idx, list):
      self.bvp_idx = bvp_idx
    else:
      self.bvp_idx = [bvp_idx]

    self.embedding_size = embedding_size
    self.dropout_rate = dropout_rate

    self.encoder_bvp =  nn.Sequential(
          nn.Conv1d(len(self.bvp_idx), nfilters, kernel_size=(3,), stride=(2,), padding=1),
          nn.LeakyReLU(negative_slope=0.01),
          nn.Dropout(dropout_rate),
          nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding = 1),
          nn.LeakyReLU(negative_slope=0.01),
          nn.Dropout(dropout_rate),
          nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding = 1),
          nn.LeakyReLU(negative_slope=0.01),
          nn.Dropout(dropout_rate),
          nn.Conv1d(nfilters, bvp_embedding_size, kernel_size=(3,), stride=(2,), padding = 1)
      )

    self.encoder =  nn.Sequential(
          nn.Conv1d(input_channels, nfilters, kernel_size=(3,), stride=(2,), padding=1),
          nn.LeakyReLU(negative_slope=0.01),
          nn.Dropout(dropout_rate),
          nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding = 1),
          nn.LeakyReLU(negative_slope=0.01),
          nn.Dropout(dropout_rate),
          nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding = 1),
          nn.LeakyReLU(negative_slope=0.01),
          nn.Dropout(dropout_rate),
          nn.Conv1d(nfilters, embedding_size - bvp_embedding_size, kernel_size=(3,), stride=(2,), padding = 1),
          nn.Dropout(dropout_rate),
          nn.LeakyReLU()
          )
    
    self.iencoder = nn.Sequential(
        self.encoder,
        nn.Conv1d(embedding_size - bvp_embedding_size, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(self.dropout_rate),
        nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(self.dropout_rate),
        nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(nfilters, embedding_size - bvp_embedding_size, kernel_size=(2,), stride=(2,)),
        nn.Dropout(self.dropout_rate),
        nn.LeakyReLU(negative_slope=0.01)
    )

    self.iencoder_bvp = nn.Sequential(
        self.encoder_bvp,
        nn.Conv1d(bvp_embedding_size, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(self.dropout_rate),
        nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(self.dropout_rate),
        nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(nfilters, bvp_embedding_size, kernel_size=(2,), stride=(2,)),
        nn.Dropout(self.dropout_rate),
        nn.LeakyReLU(negative_slope=0.01)
    )

    self.is_encoder = nn.Sequential(
        nn.Conv1d(embedding_size, embedding_size, kernel_size=(2,), stride=(2,)),
        nn.LeakyReLU(negative_slope=0.01),
    )

    self.predictor = nn.Sequential(nn.Linear(embedding_size, predictor_hidden_size),
                                  nn.LeakyReLU(), nn.Linear(predictor_hidden_size,1), nn.LeakyReLU())

    # self.initial_net = nn.Linear(embedding_size, embedding_size)

    self.transformer = nn.Transformer(
                embedding_size, nhead=nheads, num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=int(feedforward_expansion*embedding_size))


  def forward(self, xi, yi, xp):
    xi_bvp = xi[:, : , self.bvp_idx[0]: self.bvp_idx[-1]+1, :]
    xp_bvp = xp[:, : , self.bvp_idx[0]: self.bvp_idx[-1]+1, :]

    encoded_xp_imu = torch.stack([self.encoder(b)
              for b in xp])

    encoded_xp_bvp = torch.stack([self.encoder_bvp(b)
              for b in xp_bvp])

    encoded_xp = torch.cat([encoded_xp_imu, encoded_xp_bvp], dim=2).transpose(0,1).transpose(2,3).transpose(2,1)

    # encoded_xi_imu = torch.stack([self.encoder(b)
    #           for b in xi])

    # encoded_xi_bvp = torch.stack([self.encoder_bvp(b)
    #           for b in xi_bvp])

    # encoded_xi = torch.cat([encoded_xi_imu, encoded_xi_bvp], dim=2).transpose(0,1).transpose(2,3).transpose(2,1)

    xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)
    xin_bvp = xi_bvp.transpose(1,2).reshape(xi_bvp.shape[0], xi_bvp.shape[2],  -1)

     

    encoded_xi_imu = self.iencoder(xin) #self.ts_encoder(xin)
    encoded_xi_bvp = self.iencoder_bvp(xin_bvp)# self.ts_encoder_bvp(xin_bvp)

    encoded_xi = self.is_encoder(torch.cat([encoded_xi_imu, encoded_xi_bvp],dim=1))

    #print(encoded_xi_imu2.shape, xin.shape)
    #print(encoded_xi_bvp2.shape, xin_bvp.shape)

    #sv = self.initial_net(torch.ones([1, xi.shape[0],self.embedding_size]).to(next(self.parameters()).device))
    #torch.Size([50, 128, 1]) torch.Size([1, 50, 128])
    sv = encoded_xi.permute(2,0,1)

    #print(encoded_xi2.shape, sv.shape, sv2.shape)


    # for x in encoded_xi:
    #   sv = self.transformer(x, sv)
      

    ps = list()
    for x in encoded_xp:
      sv = self.transformer(x, sv)
      ps.append(self.predictor(sv[-1:]))

    p = torch.cat(ps,dim=0).transpose(0,1)
    return p
