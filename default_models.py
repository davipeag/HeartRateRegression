import torch
from torch import nn



class HiddenInitializationConvLSTMAssembler(nn.Module):
    def __init__(self, ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
                lstm, fc_net, predictor):
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

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                #nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)        
        
    
    def forward(self,xi, yi, xp):
        
        encoded_xp = torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
            for b in xp], dim=0)

        xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)#.shape #reshape(*xis.shape[:2], -1).shape, xis.shape

        encoded_xi = self.ts_encoder(xin)

        ie = torch.cat([encoded_xi, yi.transpose(2,1)], axis=1)
        
        i_enc = self.is_encoder(ie).reshape(xi.shape[0], -1)
        h = self.h0_fc_net(i_enc)
        c = self.c0_fc_net(i_enc)
        
        hsf, _ = self.lstm(encoded_xp, (h.unsqueeze(0),c.unsqueeze(0)))
        ps = self.predictor(self.fc_net(hsf))
        
        return ps  


class NoISHiddenInitializationConvLSTMAssembler(HiddenInitializationConvLSTMAssembler):
    def __init__(self, ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
                lstm, fc_net, predictor):
        """
        initial_points: number of datapoints in initial baseline (yi.shape[1])
        """
        super(NoISHiddenInitializationConvLSTMAssembler, self).__init__(
            ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
            lstm, fc_net, predictor
        )

        
    def forward(self,xi, yi, xp):
        encoded_xp = torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
            for b in xp], dim=0)
        hsf, _ = self.lstm(encoded_xp)
        ps = self.predictor(self.fc_net(hsf))
        return ps


class LSTMISHiddenInitializationConvLSTMAssembler(HiddenInitializationConvLSTMAssembler):
    def __init__(self, ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
                lstm, fc_net, predictor):
        """
        initial_points: number of datapoints in initial baseline (yi.shape[1])
        """
        super(LSTMISHiddenInitializationConvLSTMAssembler, self).__init__(
            ts_encoder, is_encoder, h0_fc_net, c0_fc_net,
            lstm, fc_net, predictor
        )

        
    def forward(self,xi, yi, xp):
        encoded_xp = torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
            for b in xp], dim=0)
        
        encoded_xi = torch.cat(
        [self.ts_encoder(b).transpose(0,2).transpose(1,2)
            for b in xi], dim=0)

        _, hidden_vectors = self.lstm(encoded_xi)
        hsf, _ = self.lstm(encoded_xp, hidden_vectors)
        ps = self.predictor(self.fc_net(hsf))
        return ps  


