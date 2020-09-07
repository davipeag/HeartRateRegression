#%%

from torch import nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy
import math

from preprocessing_utils import (
            RecursiveHrMasker,
            LabelCumSum,
            LinearImputation,
            HZMeanSubstitute,
            DeltaHzToLabel,
            NormalizeDZ,
            LocalMeanReplacer,
            ZTransformer,
            ImputeZero,
            ActivityIdRelabeler,
            Downsampler,
            FeatureLabelSplit,
            TimeSnippetAggregator,
            RemoveLabels,
            SampleMaker,
            InitialStatePredictionSplit,
            TransformerPipeline,
            IdentityTransformer,
            OffsetLabel,
            FeatureMeanSubstitute,
            SlidingWindow,
            FakeNormalizeDZ,
            ShuffleIS,
            IsSplitNormalizeDZ
        )

import default_models
from default_models import (
    HiddenInitializationConvLSTMAssembler,
    NoISHiddenInitializationConvLSTMAssembler,
    LSTMISHiddenInitializationConvLSTMAssembler
)


def initialize_weights(model):
    for m in model.modules():
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
            nn.init.constant_(m.bias, 0)
    return model

def make_deep_conv_lstm(recursive_size = 160, total_size=162):
    class ConvLSTM(nn.Module):
        def __init__(self):
            super(ConvLSTM, self).__init__()

            self.mask = [99+i*100 for i in range(total_size)][-recursive_size:]
            
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, (5,1), padding=(2,0)),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(64, 64, (5,1), padding=(2,0)),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(64, 64, (5,1), padding=(2,0)),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(64, 64, (5,1), padding=(2,0)),
                nn.ReLU(),
            )

            self.lstm = nn.LSTM(64*40, 128, batch_first=True, num_layers=2, dropout=0.5)

            self.lin = nn.Sequential(
                nn.Linear(128, 1)
            )
            
            self.initialize_weights()

        def initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight)
                    #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)


        def forward(self, x):
            l = self.lstm(torch.flatten(self.conv(x).transpose(2,1),start_dim=2))[0]
            return self.lin(l[:, self.mask, :])#[:, -RECURSIVE_SIZE:, :])

    net = ConvLSTM()
    net.initialize_weights()
    return net

def make_cnn_imu2(recursive_size=160, total_size=162):
    class CNN_IMU2(nn.Module):
        def __init__(self):
            super(CNN_IMU2, self).__init__()

            inp_sizes = [1, 13, 13, 13]
            
            self.inp_sizes = inp_sizes

            self.fcs = nn.Sequential(
                nn.Dropout(),
                nn.Linear(len(inp_sizes)*512,512),
                nn.ReLU(),   
            )

            self.conv_b1 = self.conv_layers()
            self.conv_b2 = self.conv_layers()
            self.conv_b3 = self.conv_layers()
            self.conv_b4 = self.conv_layers()

            self.conv_nets = [
                self.conv_b1,
                self.conv_b2,
                self.conv_b3,
                self.conv_b4
                ]
            
            
            self.lin_net_b1 = self.linear_layers(1)
            self.lin_net_b2 = self.linear_layers(13)
            self.lin_net_b3 = self.linear_layers(13)
            self.lin_net_b4 = self.linear_layers(13)

            self.lin_nets = [
            self.lin_net_b1,
            self.lin_net_b2,
            self.lin_net_b3,
            self.lin_net_b4,
            ]


            self.final_fc = nn.Linear(512, 1)
            self.initialize_weights()

        def initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        def linear_layers(self, inp_size):
            i = inp_size*25*64
            return nn.Sequential(nn.Linear(i, 512),nn.ReLU())#.to(args["device"])

        def conv_layers(self):
            return nn.Sequential(
                nn.Conv2d(1, 64, (5,1), padding=(2,0)),
                nn.ReLU(),
                nn.LocalResponseNorm(5), 
                nn.Conv2d(64,64,(5,1), padding=(2,0)),
                nn.ReLU(),
                nn.LocalResponseNorm(5),
                nn.MaxPool2d((2,1),(2,1)),
                nn.Conv2d(64, 64, (5,1), padding=(2,0)),
                nn.ReLU(),
                nn.Conv2d(64, 64, (5,1), padding=(2,0)),
                nn.ReLU(),
                nn.MaxPool2d((2,1),(2,1)),
                nn.Dropout(),
            )

        def forward(self,x):
            ends = [sum(self.inp_sizes[0:i+1])
                    for i in range(len(self.inp_sizes))]
            starts = [0,*ends[:-1]]
            
            fs = [c(x[:,:,:, s:e]).reshape(x.shape[0], 64, -1, 25, e-s).transpose(1,2)
                for (s,e),c in zip(zip(starts, ends), self.conv_nets)]
            
            ls = [l(torch.flatten(i, start_dim=2)) for i,l in zip(fs, self.lin_nets)]

            joint = torch.cat(ls, dim=2)

            o = self.final_fc(self.fcs(joint))[:, :recursive_size]
            torch.cumsum(o, dim=1)
            return o
    

    net =  CNN_IMU2()
    net.initialize_weights()
    return net

def make_our_conv_lstm(sensor_count =40, output_count=1, model_type="regular"):

    class_mapping = {
        "regular": HiddenInitializationConvLSTMAssembler,
        "nois": NoISHiddenInitializationConvLSTMAssembler,
        "lstmis": LSTMISHiddenInitializationConvLSTMAssembler
    }
    
    ts_h_size = 32

    ts_encoder = nn.Sequential(
        nn.Conv1d(40, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        #nn.Dropout(),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        #nn.Dropout(),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(ts_h_size, 128, kernel_size=(2,), stride=(2,)),
        nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.01),
    )
  
    is_encoder = nn.Sequential(
        nn.Conv1d(129, 32, kernel_size=(2,), stride=(2,)),
        nn.LeakyReLU(negative_slope=0.01),
    )

    h0_fc_net = nn.Linear(in_features=32, out_features=32, bias=True)
    c0_fc_net = nn.Linear(in_features=32, out_features=32, bias=True)

    fc_net = nn.Sequential(
        nn.Linear(in_features=32, out_features=32, bias=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(in_features=32, out_features=32, bias=True),
        nn.Dropout(p=0, inplace=False),
    )
  
    predictor = nn.Linear(in_features=32, out_features=output_count)

    lstm = nn.LSTM(128, 32, batch_first=True)

    net = class_mapping[model_type](
        ts_encoder= ts_encoder,
        is_encoder = is_encoder,
        h0_fc_net = h0_fc_net,
        c0_fc_net = c0_fc_net,
        lstm= lstm,
        fc_net = fc_net,
        predictor = predictor
    )

    net.initialize_weights()
    return net

def make_attention_transormer_model(device, total_size=162, recursive_size=160):
    encoded_size = 127
    ts_h_size = 32

    ts_encoder = nn.Sequential(
        nn.Conv1d(40, 32, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(32, 32, kernel_size=(3,), stride=(2,), padding=(1,)),
        #nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(32, 32, kernel_size=(3,), stride=(2,)),
        #nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(32, 32, kernel_size=(3,), stride=(2,), padding=(1,)),
        #nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(32, 32, kernel_size=(3,), stride=(2,), padding=(1,)),
        #nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(32, encoded_size, kernel_size=(3,), stride=(2,)),
        #nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.01),
    )

    # ts_encoder = nn.Sequential(
    #     nn.Conv1d(40, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
    #     nn.LeakyReLU(negative_slope=0.01),
    #     nn.Dropout(),
    #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
    #     nn.LeakyReLU(negative_slope=0.01),
    #     nn.Dropout(),
    #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
    #     nn.LeakyReLU(negative_slope=0.01),
    #     nn.Dropout(),
    #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
    #     nn.LeakyReLU(negative_slope=0.01),
    #     nn.Dropout(),
    #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
    #     nn.LeakyReLU(negative_slope=0.01),
    #     nn.Dropout(),
    #     nn.Conv1d(ts_h_size, 127, kernel_size=(3,), stride=(2,)),
    #     nn.LeakyReLU(negative_slope=0.01),
    # )

    class PositionalEncoding(nn.Module):

        def __init__(self, time_size, device, multiplier=1):
            super(PositionalEncoding, self).__init__()
            time_range  = torch.arange(0, time_size, dtype=torch.float)
            normalized = (time_range - torch.mean(time_range))/torch.std(time_range)
            self.pos = multiplier*normalized.reshape(-1,1,1).to(device)
            

        def forward(self, x):
            return torch.cat([x, self.pos], dim=2)
    
    class PositionalEncoding2(nn.Module):

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding2, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)


    class ModuleTranspose(nn.Module):

        def __init__(self, *args):
            super(ModuleTranspose, self).__init__()
            self.args = args
        def forward(self, x):
            return x.transpose(*self.args)


    class MyTransformer(nn.Module):

        def __init__(self, embeder, transformer, regressor, recursive_size):
            super(MyTransformer, self).__init__()
            self.embeder = embeder
            self.transformer = transformer
            self.regressor = regressor
            self.recursive_size = recursive_size
        



        def forward(self, x):
            p_encs = [self.embeder(xin) for xin in x]
            p_enc = torch.cat(p_encs, dim=1)#.transpose(0,1)
            trans = self.transformer(p_enc, p_enc[-self.recursive_size:])
            p = self.regressor(trans)
            
            return p.transpose(0,1)


    transformer = nn.Transformer(encoded_size+1, nhead=16, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=128*4)

    m_transpose = ModuleTranspose(1,2)

    pos_encoder = PositionalEncoding(total_size, device)

    #pos_encoder = PositionalEncoding2(128)

    embed = nn.Sequential(ts_encoder, m_transpose, pos_encoder)

    regressor = nn.Sequential(
        nn.Linear(encoded_size+1, 32),nn.ReLU(),
        nn.Linear(32,1)
    )    
    net = MyTransformer(embed, transformer, regressor, recursive_size).to(device)
    initialize_weights(net)
    return net
    
def make_fcnn():
    class FCNN(nn.Module):
        def __init__(self):
            super(FCNN, self).__init__()
            self.fcs = nn.Sequential(
                nn.Linear(4,16),
                nn.ReLU(),
                nn.Linear(16,16),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(16,1),   
            )

        def forward(self,x):
            
            # putting windows first
            xwf = x.transpose(0,1)
            hr0 = xwf[0,:, 0:1]            
            pred = self.fcs(xwf[0])
            preds = [pred]
            csum = hr0 + pred
            for xi in xwf[1:]:
                inps = torch.cat([csum, xi[:,1:]], dim=1)
                pred = self.fcs(inps)
                csum = csum + pred
                preds.append(csum - hr0)
            return torch.stack(preds).transpose(0,1)

    net  = FCNN()
    initialize_weights(net)
    return net

class TrainOurConvLSTM():
    def __init__(
            self,
            net,
            criterion,
            optimizer,
            loader_tr,
            loader_val,
            loader_ts,
            normdz,
            ztransformer,
            device
            ):
        
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.loader_ts = loader_ts
        self.normdz = normdz
        self.ztransformer = ztransformer
        self.device = device
                 

    def inverse_transform_label(self, yis, yrs):
        inverse_delta_y = self.normdz.reverse_transform((None,yrs))[-1]

        last_y = yis[:, -1]

        ys = inverse_delta_y.reshape(inverse_delta_y.shape[:2]) + last_y

        ymean = self.ztransformer.transformer.mean_[0] 
        ystd = self.ztransformer.transformer.var_[0]**0.5

        inverse_ys = (ys*ystd)+ymean
        return inverse_ys


    def plot_predictions(self, yi, yp, predictions, print_indices = [0]):
        yi = yi.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(yi,yp)
        pr = self.inverse_transform_label(yi,p)

        for i in print_indices:
            plt.figure()
            plt.plot(np.linspace(1,3*yr.shape[1]-1, yr.shape[1]), yr[i], 'b', label="label")
            plt.plot(np.linspace(1,3*yr.shape[1]-1, yr.shape[1]), pr[i], 'k', label="predictions")
            plt.xlabel("seconds")
            plt.legend()
            plt.show()

    def plot_prediction(self, yi, yp, predictions, print_index = 0):
        yi = yi.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(yi,yp)
        pr = self.inverse_transform_label(yi,p)

        i = print_index
        fig = plt.figure()
        plt.plot(np.linspace(1,3*yr.shape[1]-1, yr.shape[1]), yr[i], 'b', label="label")
        plt.plot(np.linspace(1,3*yr.shape[1]-1, yr.shape[1]), pr[i], 'k', label="predictions")
        plt.xlabel("seconds")
        plt.legend()
        return fig 

    def plot_heart_rate(self, loader, indices=[0,]):
        batch = loader.__iter__().__next__()
        self.net.eval()
        with torch.no_grad():
            xi, yi, xp, yp = map(lambda v: v.to(self.device), batch)
            self.plot_predictions(yi,yp, self.net(xi,yi,xp), indices)



    def HR_MAE(self, yi,yp, predictions):
        yi = yi.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(yi,yp)
        pr = self.inverse_transform_label(yi,p)

        return np.abs(yr-pr).mean()


    def HR_RMSE(self, yi, yp, predictions):
        yi = yi.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(yi,yp)
        pr = self.inverse_transform_label(yi,p)

        return (((yr-pr)**2).mean())**0.5



    def train(self, batch):
        self.net.train()
        xi,yi,xr, yr = map(lambda v: v.to(self.device), batch)
        self.net.zero_grad()

        p = self.net(xi,yi,xr)
        loss = self.criterion(p, yr)
        
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def validate(self):
        self.net.eval()
        losses = list()
        for batch in self.loader_val:
            with torch.no_grad():
                xi,yi,xr,yr = map(lambda v: v.to(self.device), batch)
                p = self.net(xi,yi,xr)
                loss = self.criterion(p, yr)
                losses.append(loss.cpu().item())
        
        return torch.mean(torch.FloatTensor(losses))
        
    #mae_criterion = nn.L1Loss()

    def reverse_transformed_prediction_labels(self, loader):
        xis, yis, xrs, yrs, ps = map(lambda v: v.detach().cpu().numpy(),
                         self.get_data_epoch(loader))
        yr, pr = map(lambda v: self.inverse_transform_label(yis, v), [yrs, ps])
        return yr, pr

    def get_data_epoch(self, loader):
        self.net.eval()
        xis, yis, xrs, yrs, ps = [],[],[],[],[]
        with torch.no_grad():
            for batch in loader:
                with torch.no_grad():
                    xi,yi,xr,yr = map(lambda v: v.to(self.device), batch)
                    p = self.net(xi, yi, xr)
                    xis.append(xi.detach().cpu())
                    xrs.append(xr.detach().cpu())
                    yis.append(yi.detach().cpu())
                    yrs.append(yr.detach().cpu())
                    ps.append(p.detach().cpu())
        
        return torch.cat(xis), torch.cat(yis), torch.cat(xrs), torch.cat(yrs), torch.cat(ps)

    def compute_batch_MAE(self, batch):
        self.net.eval()
        with torch.no_grad():
            xi,yi,xr,yr = map(lambda v: v.to(self.device), batch)
            return self.HR_MAE(yi,yr, self.net(xi,yi,xr)) 
        
    
    def compute_mean_MAE(self, loader):
        xi, yi, xr, yr, p = self.get_data_epoch(loader)
        return self.HR_MAE(yi, yr, p)
    


    def train_epochs(self, n_epoch):
        best_val_model = copy.deepcopy(self.net.state_dict()) 
        train_losses = list()
        train_accuracies = list()
        validation_losses = list()
        validation_accuracies = list()
        test_accuracies = list()

        validation_losses.append(self.compute_mean_MAE(self.loader_val))

        for epoch in range(1, n_epoch+1):           
            losses = []
            for batch_idx, batch in enumerate(self.loader_tr):
                losses.append(self.train(batch))

            val_loss = self.compute_mean_MAE(self.loader_val) 
            if val_loss < np.min(validation_losses):
                print("best val epoch:", epoch)
                best_val_model = copy.deepcopy(self.net.state_dict()) 

            train_losses.append(torch.mean(torch.FloatTensor(losses)) )
            validation_losses.append(val_loss)
            train_accuracies.append(self.compute_mean_MAE(self.loader_tr))
            validation_accuracies.append(val_loss)
            test_accuracies.append(self.compute_mean_MAE(self.loader_ts))   
            print('[%d/%d]: loss_train: %.3f loss_val %.3f loss_ts %.3f' % (
                    (epoch), n_epoch, train_accuracies[-1],
                    validation_accuracies[-1], test_accuracies[-1]))
            
            if (epoch % 10) == 0:
                print("Test")
                self.plot_heart_rate(self.loader_ts)
                print("Validation")
                self.plot_heart_rate(self.loader_val)
                print("Train")
                self.plot_heart_rate(self.loader_tr)
                self.compute_mean_MAE(self.loader_ts)

        return {
            "best_val_model": best_val_model,
            "train_mae": train_losses,
            "validation_mae": train_accuracies,
            "test_mae": test_accuracies
        }

class DefaultPamapPreprocessing():
    def __init__(self, ts_per_sample=162, ts_per_is=2, last_transformer = IdentityTransformer(),
                 ts_count = 100, donwsampling_ratio = 0.3, sample_multiplier =2):


        self.last_transformer = last_transformer
        self.recursive_hr_masker = RecursiveHrMasker(0)
        self.label_cum_sum = LabelCumSum()
        self.hr_lin_imputation = LinearImputation("heart_rate")
        self.meansub = HZMeanSubstitute()
        self.deltahztolabel = DeltaHzToLabel()
        self.normdz = IsSplitNormalizeDZ()

        self.local_mean_imputer = LocalMeanReplacer()
        self.ztransformer = ZTransformer()
        self.zero_imputer = ImputeZero()
        self.activity_id_relabeler = ActivityIdRelabeler()
        self.downsampler = Downsampler(donwsampling_ratio)
        self.feature_label_splitter = FeatureLabelSplit(
            label_column="heart_rate",
            feature_columns = [
                'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
                'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
                'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
                'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
                'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
                'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
                'a_ymag', 'a_zmag'
            ]
        )
        self.ts_aggregator = TimeSnippetAggregator(
            size=ts_count,
            label_collapser_function= lambda v: np.mean(v, axis=1)
        )

        self.label_remover = RemoveLabels([0])

        self.sample_maker = SampleMaker(ts_per_sample, ts_per_sample//sample_multiplier)

        self.sample_maker_ts = SampleMaker(ts_per_sample, ts_per_sample)

        self.is_pred_split = InitialStatePredictionSplit(ts_per_sample, ts_per_is)

        self.transformers = TransformerPipeline(
            self.ztransformer, self.hr_lin_imputation, self.local_mean_imputer,
            self.activity_id_relabeler, self.downsampler, 
            self.feature_label_splitter,
            self.ts_aggregator, #self.meansub, self.deltahztolabel, self.normdz,
            self.sample_maker, self.is_pred_split, self.normdz,
            self.recursive_hr_masker, self.last_transformer)
        
        self.transformers_ts = TransformerPipeline(
            self.ztransformer, self.hr_lin_imputation, self.local_mean_imputer,
            self.activity_id_relabeler, self.downsampler, 
            self.feature_label_splitter,
            self.ts_aggregator, #self.meansub, self.deltahztolabel, self.normdz,
            self.sample_maker_ts, self.is_pred_split, self.normdz,
            self.recursive_hr_masker, self.last_transformer)
        

        # self.transformers_ts = TransformerPipeline(
        #     self.ztransformer, self.hr_lin_imputation, self.local_mean_imputer,
        #     self.activity_id_relabeler, self.downsampler,
        #     self.feature_label_splitter,
        #     self.ts_aggregator, self.meansub, self.deltahztolabel, self.normdz,
        #     self.sample_maker_ts, self.label_cum_sum, self.is_pred_split,
        #     self.recursive_hr_masker, self.last_transformer)

class FcPamapPreprocessing():
    def __init__(self, ts_per_sample=162, ts_per_is=2, last_transformer = IdentityTransformer(),
                 ts_count = 100, donwsampling_ratio = 0.3, sample_multiplier =2):


        self.last_transformer = last_transformer
        self.recursive_hr_masker = RecursiveHrMasker(0)
        self.label_cum_sum = LabelCumSum()
        self.hr_lin_imputation = LinearImputation("heart_rate")
        self.meansub = HZMeanSubstitute()
        self.deltahztolabel = DeltaHzToLabel()
        self.normdz = FakeNormalizeDZ()

        self.local_mean_imputer = LocalMeanReplacer()
        self.ztransformer = ZTransformer()
        self.zero_imputer = ImputeZero()
        self.activity_id_relabeler = ActivityIdRelabeler()
        self.downsampler = Downsampler(donwsampling_ratio)
        self.feature_label_splitter = FeatureLabelSplit(
            label_column="heart_rate"
            feature_columns =["heart_rate", 'h_xacc16', 'h_yacc16', 'h_zacc16']
        )
        self.ts_aggregator = TimeSnippetAggregator(
            size=ts_count,
            label_collapser_function= lambda v: np.mean(v, axis=1))
        self.label_remover = RemoveLabels([0])

        recursive_size = ts_per_sample - ts_per_is
        self.sliding_window = SlidingWindow(recursive_size,recursive_size//2)
        self.sliding_window_ts = SlidingWindow(recursive_size,recursive_size)
        self.sample_maker = SampleMaker(recursive_size, recursive_size//sample_multiplier)

        self.sample_maker_ts = SampleMaker(recursive_size, recursive_size)

        self.is_pred_split = InitialStatePredictionSplit(ts_per_sample, ts_per_is)

        self.feature_mean_substitute = FeatureMeanSubstitute()

        self.offset_label = OffsetLabel()

        self.transformers = TransformerPipeline(
            self.ztransformer, self.hr_lin_imputation, self.local_mean_imputer,
            self.downsampler, self.feature_label_splitter,
            self.ts_aggregator,  self.deltahztolabel, #self.normdz,
            self.sliding_window,  self.feature_mean_substitute, self.label_cum_sum)        

        self.transformers = TransformerPipeline(
            self.ztransformer, self.hr_lin_imputation, self.local_mean_imputer,
            self.downsampler, self.feature_label_splitter,
            self.ts_aggregator,  self.deltahztolabel, #self.normdz,
            self.sliding_window_ts,  self.feature_mean_substitute, self.label_cum_sum)

        # self.transformers_ts = TransformerPipeline(
        #     self.ztransformer, self.hr_lin_imputation, self.local_mean_imputer,
        #     self.activity_id_relabeler, self.downsampler, self.feature_label_splitter,
        #     self.ts_aggregator, self.meansub, self.deltahztolabel, self.normdz,
        #     self.sliding_window_ts, self.feature_mean_substitute, self.label_cum_sum)

class TrainXY():
    def __init__(
            self,
            net,
            criterion,
            optimizer,
            loader_tr,
            loader_val,
            loader_ts,
            normdz,
            ztransformer,
            device,
            get_last_y_from_x,
            ):
        
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.loader_ts = loader_ts
        self.normdz = normdz
        self.ztransformer = ztransformer
        self.device = device
        self.get_last_y_from_x = get_last_y_from_x                 

    def inverse_transform_label(self, x, y):
        inverse_delta_y = self.normdz.reverse_transform((None,y))[-1]

        last_y = self.get_last_y_from_x(x)

        ys = inverse_delta_y.reshape(inverse_delta_y.shape[:2]) + last_y

        ymean = self.ztransformer.transformer.mean_[0] 
        ystd = self.ztransformer.transformer.var_[0]**0.5

        inverse_ys = (ys*ystd)+ymean
        return inverse_ys
    
    def plot_predictions(self, x, y, predictions, print_indices = [0]):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(x,y)
        pr = self.inverse_transform_label(x,p)

        for i in print_indices:
            plt.figure()
            plt.plot(np.linspace(1,3*yr.shape[1]-1, yr.shape[1]), yr[i], 'b', label="label")
            plt.plot(np.linspace(1,3*yr.shape[1]-1, yr.shape[1]), pr[i], 'k', label="predictions")
            plt.xlabel("seconds")
            plt.legend()
            plt.show()
 

    def plot_heart_rate(self, loader, indices=[0,]):
        batch = loader.__iter__().__next__()
        self.net.eval()
        x,y = map(lambda v: v.to(self.device), batch)
        self.plot_predictions(x,y, self.net(x), indices)


    
    def reverse_transformed_prediction_labels(self, loader):
        xs, ys, ps = map(lambda v: v.detach().cpu().numpy(),
                         self.get_data_epoch(loader))
        yr, pr = map(lambda v: self.inverse_transform_label(xs, v), [ys, ps])
        return yr, pr


    def HR_MAE(self, x, y, predictions):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(x,y)
        pr = self.inverse_transform_label(x,p)

        return np.abs(yr-pr).mean()


    def HR_RMSE(self, x, y, predictions):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(x,y)
        pr = self.inverse_transform_label(x,p)

        return (((yr-pr)**2).mean())**0.5



    def train(self, batch):
        self.net.train()
        x, y = map(lambda v: v.to(self.device), batch)
        self.net.zero_grad()
        p = self.net(x)
        loss = self.criterion(p, y)
        
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def validate(self):
        self.net.eval()
        losses = list()
        for batch in self.loader_val:
            with torch.no_grad():
                x,y = map(lambda v: v.to(self.device), batch)
                p = self.net(x)
                loss = self.criterion(p, y)
                losses.append(loss.cpu().item())
        
        return torch.mean(torch.FloatTensor(losses))
        

    def get_data_epoch(self, loader):
        self.net.eval()
        xs, ys, ps = [], [], []
        with torch.no_grad():
            for batch in loader:
                with torch.no_grad():
                    x,y = map(lambda v: v.to(self.device), batch)
                    p = self.net(x)
                    xs.append(x.detach().cpu())
                    ys.append(y.detach().cpu())
                    ps.append(p.detach().cpu())
        
        return torch.cat(xs), torch.cat(ys),torch.cat(ps)



    def compute_batch_MAE(self, batch):
        self.net.eval()
        with torch.no_grad():
            x,y = map(lambda v: v.to(self.device), batch)
            mae = self.HR_MAE(x, y, self.net(x)) 
        return mae
    
    def compute_mean_MAE(self, loader):
        x, y, p = self.get_data_epoch(loader)
        return self.HR_MAE(x, y, p)
    
    def train_epochs(self, n_epoch):
        best_val_model = copy.deepcopy(self.net.state_dict()) 
        train_losses = list()
        train_accuracies = list()
        validation_losses = list()
        validation_accuracies = list()
        test_accuracies = list()

        validation_losses.append(self.compute_mean_MAE(self.loader_val))

        for epoch in range(1, n_epoch+1):           
            losses = []
            for batch_idx, batch in enumerate(self.loader_tr):
                losses.append(self.train(batch))

            val_loss = self.compute_mean_MAE(self.loader_val) 
            if val_loss < np.min(validation_losses):
                print("best val epoch:", epoch)
                best_val_model = copy.deepcopy(self.net.state_dict()) 

            train_losses.append(torch.mean(torch.FloatTensor(losses)) )
            validation_losses.append(val_loss)
            train_accuracies.append(self.compute_mean_MAE(self.loader_tr))
            validation_accuracies.append(val_loss)
            test_accuracies.append(self.compute_mean_MAE(self.loader_ts))   
            print('[%d/%d]: loss_train: %.3f loss_val %.3f loss_ts %.3f' % (
                    (epoch), n_epoch, train_accuracies[-1],
                    validation_accuracies[-1], test_accuracies[-1]))
            
            if (epoch % 10) == 0:
                print("Test")
                self.plot_heart_rate(self.loader_ts)
                print("Validation")
                self.plot_heart_rate(self.loader_val)
                print("Train")
                self.plot_heart_rate(self.loader_tr)
                self.compute_mean_MAE(self.loader_ts)

        return {
            "best_val_model": best_val_model,
            "train_mae": train_losses,
            "validation_mae": train_accuracies,
            "test_mae": test_accuracies
        }




# %%
