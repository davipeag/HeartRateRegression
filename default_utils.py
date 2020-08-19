
from torch import nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy


def make_our_conv_lstm(sensor_count =40, output_count=1, mask_hidden=False):
    
    ts_h_size = 32

    ts_encoder = nn.Sequential(
        nn.Conv1d(40, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
        nn.LeakyReLU(negative_slope=0.01),
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
        nn.Conv1d(129, 32, kernel_size=(3,), stride=(2,)),
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

    multiplier = 0 if mask_hidden == True else 1 

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
            
            encoded_xp = torch.cat(
                [self.ts_encoder(b).transpose(0,2).transpose(1,2)
                for b in xp], dim=0)

            xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)

            encoded_xi = ts_encoder(xin)

            ie = torch.cat([encoded_xi, yi.transpose(2,1)], axis=1)
            
            i_enc = is_encoder(ie).reshape(xi.shape[0], -1)
            h = self.h0_fc_net(i_enc)*multiplier
            c = self.c0_fc_net(i_enc)*multiplier
            
            hsf, _ = self.lstm(encoded_xp, (h.unsqueeze(0),c.unsqueeze(0)))
            ps = self.predictor(self.fc_net(hsf))
            
            return ps  

    net = HiddenInitializationConvLSTMAssembler(
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
        _, inverse_delta_y = self.normdz.reverse_transform((None,yrs))

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
        xi, yi, xp, yp = map(lambda v: v.to(self.device), batch)
        self.plot_predictions(yi,yp, self.net(xi,yi,xp), indices)


    def plot_single_heart_rate(self, loader, index):
        batch = loader.__iter__().__next__()
        self.net.eval()
        xi, yi, xp, yp = map(lambda v: v.to(self.device), batch)
        return self.plot_prediction(yi,yp, self.net(xi,yi,xp), index)


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
            self.net.eval()
            xi,yi,xr,yr = map(lambda v: v.to(self.device), batch)
            p = self.net(xi,yi,xr)
            loss = self.criterion(p, yr)
            losses.append(loss.cpu().item())
        
        return torch.mean(torch.FloatTensor(losses))
        
    #mae_criterion = nn.L1Loss()

    def get_data_epoch(self, loader):
        self.net.eval()
        xis, yis, xrs, yrs, ps = [],[],[],[],[]
        with torch.no_grad():
            for batch in loader:
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
