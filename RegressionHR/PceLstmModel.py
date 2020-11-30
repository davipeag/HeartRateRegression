import torch
from torch import nn

import Models
class HiddenInitializationConvLSTMAssembler(torch.nn.Module):
    def __init__(self, ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
                 lstm, fc_net, predictor, bvp_idx = 4):
        """
        initial_points: number of datapoints in initial baseline (yi.shape[1])
        """
        super(HiddenInitializationConvLSTMAssembler, self).__init__()

        self.ts_encoder = ts_encoder
        self.is_encoder = is_encoder
        self.h0_fc_net = h0_fc_net
        self.c0_fc_net = c0_fc_net
        self.lstm = lstm
        self.fc_net = fc_net
        self.predictor = predictor
        
        self.bvp_idx = bvp_idx

    def ts_encode(self, x):
      xp = self.ts_cell_0(x)
      for c,m in zip(self.convs, self.mps):
        cx = c(xp)
        mx = m(xp)
        xp = cx + mx
      return self.ts_cell_7(xp)
     
    def encode_initial(self, encoded_xi, yi):
        yi = yi.reshape(*yi.shape[0:2],1)
        enc_initial = torch.cat([encoded_xi, yi],axis=2).transpose(2,1)
        return self.is_encoder(enc_initial).reshape(
            yi.shape[0],-1)
      
    def calc_h0c0(self, encoded_xi, yi):
        i_enc = self.encode_initial(encoded_xi, yi)
        h0,c0 = self.h0_fc_net(i_enc), self.c0_fc_net(i_enc)
        return h0, c0
      
    def forward(self,xi, yi, xp):
        encoded_xp= torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
          for b in xp], dim=0)
      
        xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)
        
        encoded_xi = self.ts_encoder(xin)
        ie = torch.cat([encoded_xi, yi.transpose(2,1)], axis=1)
        
        i_enc = self.is_encoder(ie).reshape(xi.shape[0], -1)
        h = self.h0_fc_net(i_enc)
        c = self.c0_fc_net(i_enc)
        hsf, _ = self.lstm(encoded_xp, (h.unsqueeze(0),c.unsqueeze(0)))
        ps = self.predictor(self.fc_net(hsf))
        
        return ps  

class HiddenInitializationConvLSTMAssembler2(torch.nn.Module):
    def __init__(self, ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
                 lstm, fc_net, predictor, bvp_idx = 4):
        """
        initial_points: number of datapoints in initial baseline (yi.shape[1])
        """
        super(HiddenInitializationConvLSTMAssembler2, self).__init__()

        self.ts_encoder = ts_encoder
        self.is_encoder = is_encoder
        self.h0_fc_net = h0_fc_net
        self.c0_fc_net = c0_fc_net
        self.lstm = lstm
        self.fc_net = fc_net
        self.predictor = predictor
        
        self.bvp_idx = bvp_idx

    def ts_encode(self, x):
      xp = self.ts_cell_0(x)
      for c,m in zip(self.convs, self.mps):
        cx = c(xp)
        mx = m(xp)
        xp = cx + mx
      return self.ts_cell_7(xp)
     
    def encode_initial(self, encoded_xi, yi):
        yi = yi.reshape(*yi.shape[0:2],1)
        enc_initial = torch.cat([encoded_xi, yi],axis=2).transpose(2,1)
        return self.is_encoder(enc_initial).reshape(
            yi.shape[0],-1)
      
    def calc_h0c0(self, encoded_xi, yi):
        i_enc = self.encode_initial(encoded_xi, yi)
        h0,c0 = self.h0_fc_net(i_enc), self.c0_fc_net(i_enc)
        return h0, c0
      
    def forward(self,xi, yi, xp):
        encoded_xp= torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
          for b in xp], dim=0)

        encoded_xi= torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
          for b in xi], dim=0).squeeze(-1)

        i_enc = self.is_encoder(torch.cat([encoded_xi, yi], dim=2).transpose(2,1)).squeeze(-1)
      
        # xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)
        
        # encoded_xi = self.ts_encoder(xin)
        # ie = torch.cat([encoded_xi, yi.transpose(2,1)], axis=1)
        
        # i_enc = self.is_encoder(ie).reshape(xi.shape[0], -1)
        h = self.h0_fc_net(i_enc)
        c = self.c0_fc_net(i_enc)
        hsf, _ = self.lstm(encoded_xp, (h.unsqueeze(0),c.unsqueeze(0)))
        ps = self.predictor(self.fc_net(hsf))
        
        return ps  


class MakeOurConvLSTM():
  def __init__(self, ts_h_size = 32, lstm_size=32, lstm_input = 128, dropout_rate = 0,
               nattrs = 5):
    self.ts_h_size = ts_h_size
    self.lstm_size = lstm_size
    self.lstm_input = lstm_input
    self.dropout_rate = dropout_rate
    self.nattrs = nattrs

  def make_ts_encoder(self, input_channels, output_channels, nfilters):
    return nn.Sequential(
      nn.Conv1d(input_channels, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,)),# padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,)),#, padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Conv1d(nfilters, output_channels, kernel_size=(2,), stride=(2,)),
      nn.Dropout(self.dropout_rate),
      nn.LeakyReLU(negative_slope=0.01),
    )
  
  def make_is_encoder(self):
    return nn.Sequential(
        nn.Conv1d(self.lstm_input+1, self.lstm_size, kernel_size=(2,), stride=(2,)),
        nn.LeakyReLU(negative_slope=0.01),
    )
  
  def make_h0_fc_net(self):
    return nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size, bias=True)

  def make_c0_fc_net(self):
    return nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size, bias=True)
  
  def make_fc_net(self):
    return nn.Sequential(
        nn.Linear(in_features=self.lstm_size, out_features=32, bias=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(in_features=32, out_features=self.lstm_size, bias=True),
        nn.Dropout(p=0, inplace=False),
    )
  
  
  def make_hr_predictor(self):
    return nn.Linear(in_features=self.lstm_size, out_features=1)
  
  def make_lstm(self):
    return nn.LSTM(self.lstm_input, self.lstm_size, batch_first=True)
  
  def __call__(self):
    net = HiddenInitializationConvLSTMAssembler(
    ts_encoder= self.make_ts_encoder(self.nattrs, self.lstm_input, self.ts_h_size),
    is_encoder = self.make_is_encoder(),
    h0_fc_net = self.make_h0_fc_net(),
    c0_fc_net = self.make_c0_fc_net(),
    lstm= self.make_lstm(),
    fc_net = self.make_fc_net(),
    predictor = self.make_hr_predictor()
    )
    return net


class ParametrizedEncoderMakeOurConvLSTM():
  def __init__(self, input_length=400, ts_per_is=2, ts_h_size = 32, is_h_size = 32, lstm_size=32, lstm_input = 128, dropout_rate = 0,
               nattrs = 5):
    self.input_length = input_length
    self.ts_per_is = ts_per_is
    self.ts_h_size = ts_h_size
    self.is_h_size = is_h_size
    self.lstm_size = lstm_size
    self.lstm_input = lstm_input
    self.dropout_rate = dropout_rate
    self.nattrs = nattrs

  def make_ts_encoder(self, input_channels, output_channels, nfilters):
    return Models.BaseModels.ConstantHiddenSizeHalvingFullyConvolutionalEncoder1D(
      self.input_length, input_channels, output_channels, self.ts_h_size, self.dropout_rate
    )
  
  def make_is_encoder(self):
    return Models.BaseModels.ConstantHiddenSizeHalvingFullyConvolutionalEncoder1D(
      self.ts_per_is, self.lstm_input + 1, self.lstm_size,  self.is_h_size, self.dropout_rate
    )
    
  def make_h0_fc_net(self):
    return nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size, bias=True)

  def make_c0_fc_net(self):
    return nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size, bias=True)
  
  def make_fc_net(self):
    return nn.Sequential(
        nn.Linear(in_features=self.lstm_size, out_features=32, bias=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(in_features=32, out_features=self.lstm_size, bias=True),
        nn.Dropout(p=0, inplace=False),
    )
  
  
  def make_hr_predictor(self):
    return nn.Linear(in_features=self.lstm_size, out_features=1)
  
  def make_lstm(self):
    return nn.LSTM(self.lstm_input, self.lstm_size, batch_first=True)
  
  def __call__(self):
    net = HiddenInitializationConvLSTMAssembler2(
    ts_encoder= self.make_ts_encoder(self.nattrs, self.lstm_input, self.ts_h_size),
    is_encoder = self.make_is_encoder(),
    h0_fc_net = self.make_h0_fc_net(),
    c0_fc_net = self.make_c0_fc_net(),
    lstm= self.make_lstm(),
    fc_net = self.make_fc_net(),
    predictor = self.make_hr_predictor()
    )
    return net


def make_pce_lstm(
    ts_h_size = 32,
    lstm_size=32,
    lstm_input = 128,
    dropout_rate = 0,
    nattrs=5
    ):
  return MakeOurConvLSTM(ts_h_size, lstm_size, lstm_input, dropout_rate,
                nattrs)()


def make_par_enc_pce_lstm(
    sample_per_ts = 400,
    ts_per_is = 2,
    ts_h_size = 32,
    is_h_size = 32,
    lstm_size=32,
    lstm_input = 128,
    dropout_rate = 0,
    nattrs=40
    ):
  return ParametrizedEncoderMakeOurConvLSTM(sample_per_ts, ts_per_is, ts_h_size, is_h_size, lstm_size, lstm_input, dropout_rate, nattrs )()
   #ts_h_size, lstm_size, lstm_input, dropout_rate, nattrs)()



class Discriminator(torch.nn.Module):
  def __init__(self, input_length, nlayers=3, layer_size=32, dropout_rate = 0, activation = torch.nn.LeakyReLU()):
    super(Discriminator, self).__init__()
    self.nlayers = nlayers
    self.input_length = input_length
    self.layer_size = layer_size
    self.activation = activation
    # last_activation = torch.nn.Sigmoid()
    last_activation = torch.nn.Identity()
    if nlayers > 0:
      first_layer = self.make_layer(input_length, layer_size, self.activation)
      last_layer = self.make_layer(layer_size, 1, last_activation)
    else:
      first_layer = torch.nn.Identity()
      last_layer = self.make_layer(input_length, 1, last_activation)

    self.discriminator =  torch.nn.Sequential(
        first_layer,
        *[self.make_layer(layer_size, layer_size, self.activation, dropout_rate) for _ in range(nlayers-1)],
        last_layer)
  
  def make_layer(self, input_size, output_size, activation, dropout_rate=0):
    return torch.nn.Sequential(torch.nn.Linear(input_size, output_size),
                               torch.nn.Dropout(dropout_rate),
                               activation)

  def forward(self, x):
    return self.discriminator(x)

class PceDiscriminatorAssembler(torch.nn.Module):
  def __init__(self, ts_encoder, is_encoder, discriminator):
    super(PceDiscriminatorAssembler, self).__init__()
    self.ts_encoder = ts_encoder
    self.is_encoder = is_encoder
    self.discriminator = discriminator
  
  def forward(self, x0, hr0, x1, hr1):
    x0enc = self.ts_encoder(x0.transpose(1,2).reshape(x0.shape[0], x0.shape[2],  -1))
    x1enc = self.ts_encoder(x1.transpose(1,2).reshape(x1.shape[0], x1.shape[2],  -1))

    pce0 = self.is_encoder(torch.cat([x0enc, hr0.transpose(2,1)], axis=1)).squeeze(2)
    pce1 = self.is_encoder(torch.cat([x1enc, hr1.transpose(2,1)], axis=1)).squeeze(2)
  
    return self.discriminator(torch.cat([pce0, pce1], axis=1))


class PceDiscriminatorAssembler2(torch.nn.Module):
  def __init__(self, ts_encoder, is_encoder, discriminator):
    super(PceDiscriminatorAssembler2, self).__init__()
    self.ts_encoder = ts_encoder
    self.is_encoder = is_encoder
    self.discriminator = discriminator
  
  def ts_encode(self, x):
    return torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
          for b in x], dim=0).squeeze(-1)
  
  def is_encode(self, encx, y):
        return self.is_encoder(torch.cat([encx, y], dim=2).transpose(2,1)).squeeze(-1)
  
  def compute_pce(self, x, y):
    enc = self.ts_encode(x)
    return self.is_encode(enc, y)
        
  def forward(self, x0, hr0, x1, hr1):
    
      
    # x0enc = self.ts_encoder(x0.transpose(1,2).reshape(x0.shape[0], x0.shape[2],  -1))
    # x1enc = self.ts_encoder(x1.transpose(1,2).reshape(x1.shape[0], x1.shape[2],  -1))

    # pce0 = self.is_encoder(torch.cat([x0enc, hr0.transpose(2,1)], axis=1)).squeeze(2)
    # pce1 = self.is_encoder(torch.cat([x1enc, hr1.transpose(2,1)], axis=1)).squeeze(2) 

    pce0 = self.compute_pce(x0, hr0)
    pce1 = self.compute_pce(x1, hr1)

    return self.discriminator(torch.cat([pce0, pce1], axis=1))



def make_pce_lstm_and_discriminator(
    ts_h_size = 32,
    lstm_size=32,
    lstm_input = 128,
    dropout_rate = 0,
    nattrs=5,
    disc_nlayers=3,
    disc_layer_size=32,
    disc_dropout_rate = 0
    ):
  pce_lstm = MakeOurConvLSTM(ts_h_size, lstm_size, lstm_input, dropout_rate,
                nattrs)()
  discriminator = Discriminator(lstm_size*2, disc_nlayers, disc_layer_size, disc_dropout_rate)

  pce_discriminator = PceDiscriminatorAssembler(pce_lstm.ts_encoder, pce_lstm.is_encoder, discriminator)
  return pce_lstm, pce_discriminator


def parametrized_encoder_make_pce_lstm_and_discriminator(
    sample_per_ts = 400,
    ts_per_is = 2,
    ts_h_size = 32,
    is_h_size = 32,
    lstm_size=32,
    lstm_input = 128,
    dropout_rate = 0,
    nattrs=40,
    disc_nlayers=3,
    disc_layer_size=32,
    disc_dropout_rate = 0
    ):
  pce_lstm = ParametrizedEncoderMakeOurConvLSTM(sample_per_ts, ts_per_is, ts_h_size, is_h_size, lstm_size, lstm_input, dropout_rate, nattrs)()
  discriminator = Discriminator(lstm_size*2, disc_nlayers, disc_layer_size, disc_dropout_rate)

  pce_discriminator = PceDiscriminatorAssembler2(pce_lstm.ts_encoder, pce_lstm.is_encoder, discriminator)
  return pce_lstm, pce_discriminator
