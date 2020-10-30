import torch
from torch import nn

class HiddenInitializationConvLSTMAssembler(torch.nn.Module):
    def __init__(self, ts_encoder, ts_encoder_bvp, is_encoder, h0_fc_net, c0_fc_net,
                 lstm, fc_net, predictor, bvp_idx = [4,5]):
        """
        initial_points: number of datapoints in initial baseline (yi.shape[1])
        """
        super(HiddenInitializationConvLSTMAssembler, self).__init__()

        self.ts_encoder = ts_encoder
        self.ts_encoder_bvp = ts_encoder_bvp
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

        xi_bvp = xi[:, : , self.bvp_idx[0]: self.bvp_idx[1]+1, :]
        xp_bvp = xp[:, : , self.bvp_idx[0]: self.bvp_idx[1]+1, :]

        encoded_xp_imu = torch.cat(
            [self.ts_encoder(b).transpose(0,2).transpose(1,2)
             for b in xp], dim=0)
      
        encoded_xp_bvp = torch.cat(
            [self.ts_encoder_bvp(b).transpose(0,2).transpose(1,2)
             for b in xp_bvp], dim=0)

        encoded_xp = torch.cat((encoded_xp_bvp, encoded_xp_imu), axis=-1)

        xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)
        xin_bvp = xi_bvp.transpose(1,2).reshape(xi_bvp.shape[0], xi_bvp.shape[2],  -1)

        encoded_xi = self.ts_encoder(xin)
        encoded_xi_bvp = self.ts_encoder_bvp(xin_bvp)

        ie = torch.cat([encoded_xi, encoded_xi_bvp], axis=1)
        
        i_enc = self.is_encoder(ie).reshape(xi.shape[0], -1)
        h = self.h0_fc_net(i_enc)
        c = self.c0_fc_net(i_enc)
        hsf, _ = self.lstm(encoded_xp, (h.unsqueeze(0),c.unsqueeze(0)))
        ps = self.predictor(self.fc_net(hsf))
        
        return ps  


class MakeOurConvLSTM():
  def __init__(self, ts_h_size = 32, lstm_size=32, lstm_input = 128, dropout_rate = 0,
               bvp_count=12, nattrs = 5):
    self.ts_h_size = ts_h_size
    self.lstm_size = lstm_size
    self.lstm_input = lstm_input
    self.dropout_rate = dropout_rate
    self.nattrs = nattrs
    self.bvp_count = bvp_count

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
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Dropout(self.dropout_rate),
      nn.Conv1d(nfilters, nfilters, kernel_size=(3,), stride=(2,), padding=(1,)),
      nn.LeakyReLU(negative_slope=0.01),
      nn.Conv1d(nfilters, output_channels, kernel_size=(2,), stride=(2,)),
      nn.Dropout(self.dropout_rate),
      nn.LeakyReLU(negative_slope=0.01),
    )
  
  def make_is_encoder(self):
    return nn.Sequential(
        nn.Conv1d(self.lstm_input, self.lstm_size, kernel_size=(2,), stride=(2,)),
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
    ts_encoder= self.make_ts_encoder(self.nattrs, self.lstm_input - self.bvp_count, self.ts_h_size),
    ts_encoder_bvp = self.make_ts_encoder(2, self.bvp_count, self.ts_h_size),
    is_encoder = self.make_is_encoder(),
    h0_fc_net = self.make_h0_fc_net(),
    c0_fc_net = self.make_c0_fc_net(),
    lstm= self.make_lstm(),
    fc_net = self.make_fc_net(),
    predictor = self.make_hr_predictor()
    )
    return net


def make_no_hr_pce_lstm(
    ts_h_size = 32,
    lstm_size=32,
    lstm_input = 128,
    dropout_rate = 0,
    bvp_count=12,
    nattrs=5
    ):
  return MakeOurConvLSTM(ts_h_size, lstm_size, lstm_input, dropout_rate,
                bvp_count, nattrs)()


