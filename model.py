import torch
import math
from torch import nn
from modules import *
import torch.nn.functional as F
from utils import *
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence

class Model(nn.Module):
    def __init__(self, config, data_setting):
        super(Model, self).__init__()

        # --- Model hyperparameters ---
        self._nepoch = config["training"]["n_epochs"]
        self.nlayers = config["model"]["n_layers"]
        self.nhid = config["model"]["hidden_dim"]
        self._B = config["training"]["batch_size"]

        # --- data setting ---
        self._ninp = data_setting["N_FEATURES"]
        self._nclasses = data_setting["N_CLASSES"]

    def setDevice(self, device):
        self.device = device

    def initHidden(self, bsz):
        h1 = torch.zeros(self.nlayers, bsz, self.nhid)
        h1.requires_grad = True
        #h2 = torch.zeros(self.nlayers, bsz, self.nhid)
        #h2.requires_grad = True
        #return (h1, h2)
        return h1

    def computeLoss(self, logits, labels):
        # N x C is wrong?
        if len(labels.shape) > 1:
            labels = torch.argmax(labels, 1)
        return F.cross_entropy(logits, labels)
        
class RNN(Model):
    def __init__(self, config, data_setting):
        super(RNN, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNNMean"

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # RNN takes in "imputed"
        x, m, d = data[0]

        print(x[0, :, ].squeeze())
        assert 2 == 3

        B, T, V = x.shape # Assume timesteps x batch x variables input
        #k = x.shape[1] // self.nref
        #x = F.avg_pool1d(x.transpose(1, 2), k).transpose(1, 2)
        # REVISIT
        #x = x.reshape(B, self.nref, -1, V).mean(2)
        x = x.transpose(0, 1) # sequence first
        self.reference_timesteps = torch.zeros(T)
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(x, hidden)
        logits = self.predict(out[-1]).squeeze()
        return logits

class RNNVals(Model):
    def __init__(self, config, data_setting):
        super(RNNVals, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNN_vals"

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # RNN takes in "imputed"
        _, v, lengths = data[2]
        B, T, V = v.shape # Assume timesteps x batch x variables input

        v = v.transpose(0, 1) # sequence first
        pack = torch.nn.utils.rnn.pack_padded_sequence(v, lengths, batch_first=False)
        self.reference_timesteps = torch.zeros(T)
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(pack, hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        logits = self.predict(unpacked[-1]).squeeze()
        return logits

class RNNValsGaps(Model):
    def __init__(self, config, data_setting):
        super(RNNValsGaps, self).__init__(config=config,
                                          data_setting=data_setting)
        self.NAME = "RNN_vals_gaps"

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp*2, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # RNN takes in "imputed"
        t, v, lengths = data[2]
        B, T, V = t.shape # Assume timesteps x batch x variables input

        t = t.transpose(0, 1) # sequence first
        v = v.transpose(0, 1) # sequence first
        diffs = t[1:] - t[:-1]
        diffs = torch.cat((torch.zeros((1, B, V)), diffs), dim=0)
        x = torch.cat((v, diffs), dim=2)
        pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False)
        self.reference_timesteps = torch.zeros(T)
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(pack, hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        logits = self.predict(unpacked[-1]).squeeze()
        return logits

class RNNSimple(Model):
    def __init__(self, config, data_setting):
        super(RNNSimple, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNN_simple"

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp*2, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # RNNSimple takes in "imputed"
        x, m, d = data[0]
        x = torch.cat((x, m), 2)

        B, T, V = x.shape # Assume timesteps x batch x variables input
        x = x.transpose(0, 1) # sequence first
        self.reference_timesteps = torch.zeros(T)
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(x, hidden)
        logits = self.predict(out[-1]).squeeze()
        return logits

class RNNDelta(Model):
    def __init__(self, config, data_setting):
        super(RNNDelta, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNN_delta"

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp*2, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # RNN takes in "imputed"
        x, m, d = data[0]
        x = torch.cat((x, d), 2)

        B, T, V = x.shape # Assume timesteps x batch x variables input
        x = x.transpose(0, 1) # sequence first
        self.reference_timesteps = torch.zeros(T)
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(x, hidden)
        logits = self.predict(out[-1]).squeeze()
        return logits

class RNNInterp(Model):
    def __init__(self, config, data_setting, nref):
        super(RNNInterp, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNNInterp"
        self.nref = nref

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # RNN_interp takes in "interpolated"
        t, x, l = data[1]

        B, T, V = x.shape # Assume timesteps x batch x variables input
        x = x.transpose(0, 1) # sequence first
        x = x.reshape(-1, T//self.nref, B, V).mean(1)
        self.reference_timesteps = torch.zeros(T)
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(x, hidden)
        logits = self.predict(out[-1]).squeeze()
        return logits

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        """Multiply an input matrix with a diagonally-weighted
        parameter matrix.
        """
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear_filter = nn.Linear(in_features, out_features)
        self.linear_filter.weight = torch.nn.Parameter(self.linear_filter.weight * torch.eye(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear_filter.weight.size(1))
        self.linear_filter.weight.data.uniform_(-stdv, stdv)
        self.linear_filter.bias.data.uniform_(-stdv, stdv)

    def forward(self, matrix):
        return self.linear_filter(matrix)

class GRU_D(Model):
    def __init__(self, config, data_setting):
        # --- hyperparameters ---
        self.NAME = "GRU_D"
        super(GRU_D, self).__init__(config, data_setting)
        combined_dim = self.nhid + 2*self._ninp # Input and missingness vector
        self._identity = torch.eye(self._ninp)
        self._zeros_x = torch.zeros(self._ninp)
        self._zeros_h = torch.zeros(self.nhid)
        self._h_grads = []

        # --- mappings ---
        self.z = nn.Linear(combined_dim, self.nhid) # Update gate
        self.r = nn.Linear(combined_dim, self.nhid) # Reset gate
        self.h = nn.Linear(combined_dim, self.nhid)
        self.out = nn.Linear(self.nhid, self._nclasses)
        self.gamma_x = FilterLinear(self._ninp,
                                    self._ninp,
                                    self._identity)
        self.gamma_h = nn.Linear(self._ninp, self.nhid)

    def gru_d_cell(self, x, h, m, dt, x_prime):
        # --- compute decays ---
        delta_x = torch.exp(-torch.max(self._zeros_x, self.gamma_x(dt)))

        # --- apply state-decay ---
        delta_h = torch.exp(-torch.max(self._zeros_h, self.gamma_h(dt)))
        h = delta_h * h

        x_prime = m*x + (1-m)*x_prime # Update last-observed value
        # TEST: x_prime should be last-observed values.

        # --- estimate new x value ---
        x = m*x + (1-m)*(delta_x*x_prime + (1-delta_x)*self._x_mean)

        # --- gating functions ---
        combined = torch.cat((x, h, m), dim=2)
        r = torch.sigmoid(self.r(combined))
        z = torch.sigmoid(self.z(combined))
        new_combined = torch.cat((x, torch.mul(r, h), m), dim=2)
        h_tilde = torch.tanh(self.h(new_combined))
        h = (1 - z)*h + z*h_tilde
        return h, x_prime
    
    def forward(self, data, **kwargs):
        # GRU-D takes in "imputed"
        vals, masks, diffs = data[0]
        vals = vals.transpose(0, 1)
        masks = masks.transpose(0, 1)
        diffs = diffs.transpose(0, 1)
        T, B, V = vals.shape
        self._x_mean = vals.mean(0) # Mean over timesteps (B x V)
        h = torch.zeros(self.nlayers, B, self.nhid)
        x_prime = torch.zeros(self._ninp)
        for t in range(T):
            x = vals[t].unsqueeze(0)
            m = masks[t].unsqueeze(0)
            diff = diffs[t].unsqueeze(0)
            h, x_prime = self.gru_d_cell(x, h, m, diff, x_prime)

        logits = self.out(h.squeeze(0)) # Remove time dimension
        return logits

class CAT(Model):
    def __init__(self, config, data_setting, nhop=3, intensity=True, ngran=2,
                 nsample=0.2, scaling_factor=5, nemb=50, std=0.15, explore=False):
        super(CAT, self).__init__(config, data_setting)
        self._T = data_setting["num_timesteps"]
        self.NAME = "CAT"
        self.nhop = nhop # Number of steps to take
        self.ngranularity = ngran # How many levels of granularity
        self.nsample = nsample # Number of timesteps to collect around l_t
        #self.gwidth = gwidth # How big steps should be in sensor
        self.nemb = nemb # Dimensions into which to embed the [glimpse, location] info
        self.scaling_factor = scaling_factor
        self.std = std
        self.explore = explore
        self.intensity = intensity
        self.epsilons = exponentialDecay(self._nepoch)
        self._bsz = config["training"]["batch_size"]
        
        # --- Mappings ---
        self.Controller = Controller(self.nhid, self.std)
        self.BaselineNetwork = BaselineNetwork(self.nhid, 1)
        self.GlimpseNetwork = GN(self._ninp, self.nemb, int(self.nsample*500), self.ngranularity, self.scaling_factor, self.intensity)
        #self.GlimpseNetwork = MVGlimpseNetwork(self._ninp, self.nemb, self.ngranularity, self.nsample, self.gwidth)
        self.RNN = torch.nn.GRU(self.nemb, self.nhid, self.nlayers)
        self.predict = torch.nn.Linear(self.nhid, self._nclasses)

    def denorm(self, T, l):
        #return ((0.5*(l_t + 1.0))*T)
        #return (T*0.5*(1.0+l_t)).long()
        return (0.5*((l+1.0)*T)).long()

    def forward(self, data, epoch, test):
        # CAT takes in "interpolated"
        data = data[1]
        t = data[0]
        B, T, V = t.shape
        v = data[1]
        l = data[2]
        v[torch.isnan(v)] = 0.0
        #t_max = t[:, -1, :].squeeze()

        if self.explore:
            if test:
                self.Controller.std = 0.05
            else:
                self.Controller.std = self.epsilons[epoch]
 #               self.Controller.std = 0.3
            
        self.means = torch.zeros((self._nclasses, self.nhop)) # For saving class-wise means
        baselines = [] # Predicted baselines
        reference_timesteps = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        glimpses = []
        
        # --- initial glimpse ---
        loc = torch.ones(self._bsz, 1).uniform_(-1, 1)
        loc.requires_grad = False
        hidden = self.initHidden(self._bsz)
        self.greps = []
        for i in range(self.nhop+1):
            reference_timesteps.append(self.denorm(T, loc.squeeze())/float(T))
            out, hidden, log_probs, loc, b_t = self.CATCell(data, hidden, loc)
            log_pi.append(log_probs)
            baselines.append(b_t)
            #glimpses.append(glimpse)
            
        #self.greps = torch.stack(self.greps).squeeze()
        #self.glimpses = torch.stack(glimpses).squeeze().transpose(0, 1)[:, :-1] # B x R
        self.reference_timesteps = torch.stack(reference_timesteps).squeeze().transpose(0, 1)[:, 1:] # B x R
        self.baselines = torch.stack(baselines).squeeze().transpose(0, 1)[:, :-1] # B x R
        self.log_pi = torch.stack(log_pi).squeeze().transpose(0, 1)[:, :-1] # B x R
        logits = self.predict(out.squeeze())
        return logits
    
    def CATCell(self, data, h_prev, l):
        grep = self.GlimpseNetwork(data, l)
        out, hidden = self.RNN(grep.unsqueeze(0), h_prev)
        self.greps.append(out)
        log_probs, l_next = self.Controller(out)
        b_t = self.BaselineNetwork(out)
        return out, hidden, log_probs, l_next, b_t#, grep
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        #for i in y.unique():
        #    self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)

        # --- compute reward ---
        predicted = torch.max(torch.softmax(logits, 1), 1)[1]
        if len(y.shape) > 1:
            y = torch.argmax(y, 1)
#         self.r = (predicted.float().detach() == y.float()).float()
        self.r = ((2*(predicted.float().detach() == y.float()).float())-1) # B x 1
        self.R = self.r.mean()
        self.r = self.r.unsqueeze(1).repeat(1, self.nhop) # B x nref
        
        self.loss_c = F.cross_entropy(logits, y)
        self.loss_b = F.mse_loss(self.baselines, self.r)  # Baseline should approximate mean reward

        self.adjusted_reward = self.r# - self.baselines.detach()
        self.loss_r = torch.sum(-self.log_pi*self.adjusted_reward, 1) # sum over time
        self.loss_r = 0.5*torch.mean(self.loss_r, 0) # mean over batch
        
        # --- putting it all together ---
        loss = self.loss_r + self.loss_c# + self.loss_b
        return loss

class PolicyFreeCAT(Model):
    def __init__(self, config, data_setting, nhop=3, intensity=True, ngran=2,
                 nsample=0.2, scaling_factor=5, nemb=50, std=0.15, explore=False):
        super(PolicyFreeCAT, self).__init__(config, data_setting)
        self._T = data_setting["num_timesteps"]
        self.NAME = "PolicyFreeCAT"
        self.nhop = nhop # Number of steps to take
        self.ngranularity = ngran # How many levels of granularity
        self.nsample = nsample # Number of timesteps to collect around l_t
        #self.gwidth = gwidth # How big steps should be in sensor
        self.nemb = nemb # Dimensions into which to embed the [glimpse, location] info
        self.scaling_factor = scaling_factor
        self.std = std
        self.explore = explore
        self.intensity = intensity
        self.epsilons = exponentialDecay(self._nepoch)
        self._bsz = config["training"]["batch_size"]
        
        # --- Mappings ---
        self.GlimpseNetwork = GN(self._ninp, self.nemb, int(self.nsample*500), self.ngranularity, self.scaling_factor, self.intensity)
        self.RNN = torch.nn.GRU(self.nemb, self.nhid, self.nlayers)
        self.predict = torch.nn.Linear(self.nhid, self._nclasses)

    def forward(self, data, **kwargs):
        # CAT takes in "interpolated"
        data = data[1]
        t = data[0]
        t_max = t[:, -1, :].squeeze()

        # --- initial glimpse ---
        loc = torch.ones(self._bsz, 1).uniform_(-1, 1)
        loc.requires_grad = False
        hidden = self.initHidden(self._bsz)
        for i in range(self.nhop+1):
            out, loc, hidden = self.PolicyFreeCATCell(data, loc, hidden)
            
        logits = self.predict(out.squeeze())
        return logits
    
    def PolicyFreeCATCell(self, data, l):
        grep = self.GlimpseNetwork(data, l)
        out, hidden = self.RNN(grep.unsqueeze(0), self.hidden)
        l_next = torch.ones(self._bsz, 1).uniform_(-1, 1)
        return out, l_next, hidden
    
    def computeLoss(self, logits, y):
        return F.cross_entropy(logits, y)

class PolicyFree(Model):
    def __init__(self, config, data_setting, nref=10, ngran=2, nsample=7,
                 gwidth=0.2, nemb=20):
        super(PolicyFree, self).__init__(config, data_setting)
        self.NAME = "PolicyFree"
        self.nref = nref # Number requested timesteps
        self.ngranularity = ngran # How many levels of granularity
        self.nsample = nsample # Number of timesteps to collect around l_t
        self.gwidth = gwidth # How big steps should be in sensor
        self.nemb = nemb # Dimensions into which to embed the [glimpse, location] info
        self.r = torch.tensor(np.linspace(0, 1, nref), dtype=torch.float)
        self.r = self.r.unsqueeze(0).repeat(self._B, 1)
        
        # --- Mappings ---
        self.GlimpseNetwork = GlimpseNetwork(self._ninp, self.nemb,
                                             self.ngranularity, self.nsample,
                                             self.gwidth)
        self.RNN = torch.nn.LSTM(self.nemb, self.nhid, self.nlayers)
        self.predict = torch.nn.Linear(self.nhid, self._nclasses)

    def forward(self, data, epoch, test):
        timesteps, values = data
            
        timesteps = timesteps.transpose(0, 1)
        values = values.transpose(0, 1)
        T, B, V = timesteps.shape
        self.means = torch.zeros((self._nclasses, self.nref)) # For saving class-wise means
        
        # --- initial glimpse ---
        glimpses = []
        hidden = self.initHidden(B)

        for i in range(self.nref):
            grep, glimpse = self.GlimpseNetwork(timesteps, values, self.r[:, i].unsqueeze(1))
            out, hidden = self.RNN(grep.unsqueeze(0), hidden)
            glimpses.append(glimpse)
            
        self.glimpses = torch.stack(glimpses).squeeze().transpose(0, 1) # B x R
        logits = self.predict(out.squeeze())
        return logits
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        for i in y.unique():
            self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)

        return F.cross_entropy(logits, y)

class RNN_interp(Model):
    def __init__(self, config, data_setting, adapter="linear", nref=10):#nhid, nclasses, ninp, nlayers, R, adapter="linear"):
        super(RNN_interp, self).__init__(config, data_setting)

        self.nref = nref

        # --- Load sub-networks ---
        self.Interpolator = LinearInterpolator()
        self.NAME = "RNN_interp"
        self.rnn = nn.GRU(self._ninp, self.nhid)
        self.fc = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # Interpolator takes raw input
        t, v = data[2]
        B, T, V = v.shape
        self.reference_timesteps = torch.linspace(t.min(), t.max(), self.nref).unsqueeze(0).repeat(self._B, 1)
        glimpses = []
        #self.means = torch.zeros((self._nclasses, self.nref)) # For saving class-wise means

        # Interpolation
        X_regular = self.Interpolator.forward(t, v, self.reference_timesteps)
        # X_regular has nans: Because of 0s?
        if (self._ninp == 1) & (len(X_regular.shape) < 3):
            X_regular = X_regular.unsqueeze(2)
        X_regular = X_regular.transpose(0, 1)

        # --- inference ---
        state = torch.zeros(self.nlayers, B, self.nhid)
        out, state = self.rnn(X_regular, state)
        logits = self.fc(out[-1])
            
        #self.glimpses = X_regular.transpose(0, 1).squeeze()
        return logits
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        #for i in y.unique():
        #    self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)
        return F.cross_entropy(logits, y)

class IPN(Model):
    def __init__(self, config, data_setting, nref=10):#nhid, nclasses, ninp, nlayers, R, adapter="linear"):
        super(IPN, self).__init__(config, data_setting)
        self.nref = nref

        # --- Load sub-networks ---
        self.Interpolator = GaussianAdapter(self._ninp)
        self.NAME = "IPN"
        self.rnn = nn.GRU(self._ninp, self.nhid)
        self.fc = nn.Linear(self.nhid, self._nclasses)
    
    def forward(self, data, **kwargs):
        # Interpolator takes raw input
        t, v, lengths = data[2]
        B, T, V = v.shape
        self.reference_timesteps = torch.linspace(t.min(), t.max(), self.nref).unsqueeze(0).repeat(B, 1)
        glimpses = []
        self.means = torch.zeros((self._nclasses, self.nref)) # For saving class-wise means

        # Interpolation
        X_regular = self.Interpolator.forward(t, v, self.reference_timesteps)
        if (self._ninp == 1) & (len(X_regular.shape) < 3):
            X_regular = X_regular.unsqueeze(2)
        X_regular = X_regular.transpose(0, 1)

        # --- inference ---
        state = torch.zeros(self.nlayers, B, self.nhid)
        out, state = self.rnn(X_regular, state)
        logits = self.fc(out[-1])
        #self.glimpses = X_regular.transpose(0, 1).squeeze()
        return logits
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        #for i in y.unique():
        #    self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)
        return F.cross_entropy(logits, y)

#class IPN(Model):
#    def __init__(self, nhid, nclasses, ninp, nlayers, R, BATCH_SIZE, REF_STEPS=None):
#        super(IPN, self).__init__()
#        self.nlayers = nlayers
#        self.BATCH_SIZE = BATCH_SIZE
#        self.nhid = nhid
#        self.nclasses = nclasses
#        self.nref = R
#        self.reference_timesteps = torch.tensor(np.linspace(0, 1, R), dtype=torch.float)
#
#        # --- Load sub-networks ---
#        self.Discriminator = Discriminator(1*ninp, nhid, nclasses, nlayers)
#        self.Interpolator = Interpolator(ninp)
#    
#    def forward(self, data):
#        timesteps, values = data
#        self.reference_timesteps = torch.linspace(timesteps.min(), timesteps.max(), self.nref)
#        X_regular = self.Interpolator(timesteps, values, self.reference_timesteps)
#        prediction = self.Discriminator(X_regular)
#        return prediction
#    
#    def computeLoss(self, y_hat, y):
#        if not self.criterion:
#            self.criterion = torch.nn.CrossEntropyLoss()
#        return self.criterion(y_hat, y)
#
#class Interpolator(Model):
#    def __init__(self, config, data_setting, adapter="linear", nref=10):#nhid, nclasses, ninp, nlayers, R, adapter="linear"):
#        super(Interpolator, self).__init__(config, data_setting)
#        self.nref = nref
#        self.r = torch.tensor(np.linspace(0, 1, nref), dtype=torch.float)
#        self.r = self.r.unsqueeze(0).repeat(self._B, 1)
#
#        # --- Load sub-networks ---
#        if adapter == "gaussian":
#            self.Interpolator = GaussianAdapter(self._ninp)
#            self.NAME = "IPN"
#        if adapter == "linear":
#            self.Interpolator = LinearInterpolator()
#            self.NAME = "LinearInterpolator"
#        self.rnn = nn.GRU(self._ninp, self.nhid)
#        self.fc = nn.Linear(self.nhid, self._nclasses)
#        #self.Discriminator = Discriminator(1*self._ninp, self.nhid, self._nclasses, self.nlayers)
#    
#    def forward(self, data, **kwargs):
#        # Interpolator takes raw input
#        t, v = data[2]
#        B, T, V = v.shape
#        #t, v = data
#        self.reference_timesteps = torch.linspace(t.min(), t.max(), self.nref).unsqueeze(0).repeat(self._B, 1)
#        glimpses = []
#        self.means = torch.zeros((self._nclasses, self.nref)) # For saving class-wise means
#
#        # Interpolation
#        print(torch.isnan(t).sum())
#        print(torch.isnan(v).sum())
#        X_regular = self.Interpolator.forward(t, v, self.reference_timesteps)
#        print(torch.isnan(X_regular).sum())
#        if (self._ninp == 1) & (len(X_regular.shape) < 3):
#            X_regular = X_regular.unsqueeze(2)
#        X_regular = X_regular.transpose(0, 1)
#
#        # --- inference ---
#        state = torch.zeros(self.nlayers, B, self.nhid)
#        out, state = self.rnn(X_regular, state)
#        logits = self.fc(out[-1])
#        print(logits)
#        print()
#            
#        #self.glimpses = X_regular.transpose(0, 1).squeeze()
#        return logits
#    
#    def computeLoss(self, logits, y):
#        # --- save class-specific means ---
#        #for i in y.unique():
#        #    self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)
#        return F.cross_entropy(logits, y)

#class CAT(Model):
#    def __init__(self, config, data_setting, gtype="flatten", nref=10):#ninp, nhidENSION, nclasses, nlayers, nref=10):
#        super(CAT, self).__init__(config=config,
#                                  data_setting=data_setting)
#        self.NAME = "CAT"
#        self.nref = nref # Number requested timesteps
#        self.ngranularity = 1 # How many levels of granularity
#        self.nglimpse = 21 # Number of timesteps to collect around l_t
#        self.glimpse_width = 0.2 # How big steps should be in sensor
#        self.nemb = 20 # Dimensions into which to embed the [glimpse, location] info
#        self.std = 0.2
#        self.gtype = gtype
#        #self._epsilons = exponentialDecay(config["training"]["n_epochs"])
#        
#        # --- Mappings ---
#        self.Controller = Controller(self.nhid, self.std)
#        self.GlimpseNetwork = GlimpseNetwork(self._ninp, self.nemb,
#                                             self.ngranularity, self.nglimpse,
#                                             self.glimpse_width, gtype=gtype)
#        self.BaselineNetwork = BaselineNetwork(self.nhid, 1)
#        self.RNN = torch.nn.LSTM(self.nemb, self.nhid, self.nlayers)
#        self.predict = torch.nn.Linear(self.nhid, self._nclasses)
#
#    def initHidden(self, bsz):
#        return (torch.zeros(self.nlayers, bsz, self.nhid),
#                torch.zeros(self.nlayers, bsz, self.nhid))
#        
#    def forward(self, data, epoch, test):
#        timesteps, values = data
#        if test:
#            self.Controller._epsilon = 0.0
#        else:
#            #self.Controller._epsilon = self._epsilons[epoch]
#            self.Controller._epsilon = 0.0
#            
#        timesteps = timesteps.transpose(0, 1)
#        values = values.transpose(0, 1)
#        T, B, V = timesteps.shape
#        baselines = [] # Predicted baselines
#        reference_timesteps = [] # Which classes to halt at each step
#        log_pi = [] # Log probability of chosen actions
#        
#        # --- initial glimpse ---
#        l_t = 2*torch.rand(B, 1) - 1
#        reference_timesteps.append(self.GlimpseNetwork.denormalize(timesteps[-1], l_t).squeeze())
##         reference_timesteps.append(l_t.squeeze())
#        hidden = self.initHidden(B)
#        grep, glimpse = self.GlimpseNetwork(timesteps, values, l_t.detach())
#        out, hidden = self.RNN(grep.unsqueeze(0), hidden)
#        
#        glimpses = [glimpse]
#        for i in range(self.nref-1):
#            out, hidden, log_probs, l_t, b_t, glimpse = self.CATCell(timesteps, values, out, hidden)
#            log_pi.append(log_probs)
#            baselines.append(b_t)
#            reference_timesteps.append(self.GlimpseNetwork.denormalize(timesteps[-1], l_t).squeeze())
#            glimpses.append(glimpse)
#
#        self.glimpses = torch.stack(glimpses).squeeze().transpose(0, 1)#[:, 1:] # B x nref
#        self.reference_timesteps = torch.stack(reference_timesteps).squeeze().transpose(0, 1)#[:, 1:] # B x nref
#        self.baselines = torch.stack(baselines).squeeze().transpose(0, 1)#[:, :-1] # B x nref
#        self.log_pi = torch.stack(log_pi).squeeze().transpose(0, 1)#[:, :-1] # B x nref
#        
#        logits = self.predict(out.squeeze())
#        return logits
#    
#    def CATCell(self, timesteps, values, out, hidden):
#        log_probs, l_t = self.Controller(out.squeeze(0).detach())
#        b_t = self.BaselineNetwork(out.squeeze(0).detach())
#        grep, glimpse = self.GlimpseNetwork(timesteps, values, l_t)
#        out, hidden = self.RNN(grep.unsqueeze(0), hidden)
#        return out, hidden, log_probs, l_t, b_t, glimpse
#    
#    def computeLoss(self, logits, y):
#        self.loss_c = F.cross_entropy(logits, y)
#        
#        # --- compute reward ---
#        predicted = torch.max(torch.softmax(logits, 1), 1)[1]
#        self.r = (predicted.float() == y.float()).float().detach().unsqueeze(1)
##         self.r = ((2*(predicted.float().detach() == y.float()).float())-1).detach().unsqueeze(1) # B x 1
#        #self.R = self.r.mean()
#        self.r = self.r.repeat(1, self.nref-1) # B x nref
#        
#        # --- rescale reward with baseline ---
#        self.adjusted_reward = self.r# - self.baselines.detach()
#        self.loss_b = F.mse_loss(self.baselines, self.r)  # Baseline should approximate mean reward
#        self.loss_r = 0.8*(-self.log_pi*self.adjusted_reward).sum(1).mean() # sum over time, mean over batch
#        
#        # --- putting it all together ---
#        loss = self.loss_r + self.loss_c# + self.loss_b
#        return loss
