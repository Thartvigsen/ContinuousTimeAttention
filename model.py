import torch
from torch import nn
from modules import *
import torch.nn.functional as F
from utils import *
import numpy as np

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
        h2 = torch.zeros(self.nlayers, bsz, self.nhid)
        h2.requires_grad = True
        return (h1, h2)
        
    #def initHidden(self, bsz):
    #    """Initialize hidden states"""
    #    if self._CELL_TYPE == "LSTM":
    #        h = (torch.zeros(self.nlayers,
    #                         bsz,
    #                         self.nhid),
    #             torch.zeros(self.nlayers,
    #                         bsz,
    #                         self.nhid))
    #    else:
    #        h = torch.zeros(self.nlayers,
    #                        bsz,
    #                        self.nhid)
    #    return h

    def getCriterion(self, name):
        """PyTorch implementations of Loss functions"""
        if name == "mse":
            criterion = torch.nn.MSELoss(size_average=True, reduction="mean")
        elif name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=True,
                                             ignore_index=-100, reduction="mean")
        elif name == "bce":
            criterion = torch.nn.BCELoss(weight=None, size_average=True,
                                    reduction="mean")
        else:
            raise NotImplementedError
        return criterion

    def computeLoss(self, pred, label):
        """
        Basic loss computation, this can be written over in individual
        models to use custom loss functions.
        """
        criterion = self.getCriterion(self._LOSS_NAME)
        loss = criterion.forward(pred, label)
        return loss

class RNN(Model):
    def __init__(self, config, data_setting):
        """
        Example model implementation.

        Parameters
        ----------
        config : dict
            A dictionary that contains the experimental configuration (number
            of dimensions in the hidden state, batch size, etc.). This
            dictionary gets passed into the parent Model() and initialized.

        data_setting : dict
            A dictionairy containing relevant settings from the dataset (number
            of classes to predict, whether or not the time series are of
            variable-length, how many variables the data have, etc.)
        """
        super(RNN, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNN" # Name of the model for creating the log files

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._ninp, self.nhid, self.nlayers)
        self.predict = nn.Linear(self.nhid, self._nclasses)

        # Example: using a sub-network by loading a module. Delete this if you
        # don't want to use modules!
        self.MyModule = SampleModule(self._ninp, self._nclasses)
    
    def forward(self, X):
        """
        The main method of your model, completely mapping the input data to the
        output predictions. In this example, the RNN outputs a classification
        using only the final hidden state (out[-1]).
        """
        T, B, V = X.shape # Assume timesteps x batch x variables input
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(X, hidden)
        y_hat = self.out_nonlin(self.predict(out[-1]).squeeze())
        return y_hat

class CAT(Model):
    def __init__(self, config, data_setting, nref=10, ngran=2, nsample=7,
                 gwidth=0.2, nemb=20, std=0.15):
        super(CAT, self).__init__(config, data_setting)
        self.NAME = "CAT"
        self.nref = nref # Number requested timesteps
        self.ngranularity = ngran # How many levels of granularity
        self.nsample = nsample # Number of timesteps to collect around l_t
        self.gwidth = gwidth # How big steps should be in sensor
        self.nemb = nemb # Dimensions into which to embed the [glimpse, location] info
        self.std = std
        self.epsilons = exponentialDecay(self._nepoch)
        self._bsz = config["training"]["batch_size"]
        
        # --- Mappings ---
        self.Controller = Controller(self.nhid, self.std)
        self.BaselineNetwork = BaselineNetwork(self.nhid, 1)
        self.GlimpseNetwork = MVGlimpseNetwork(self._ninp, self.nemb,
                                             self.ngranularity, self.nsample,
                                             self.gwidth)
        self.RNN = torch.nn.LSTM(self.nemb, self.nhid, self.nlayers)
        self.predict = torch.nn.Linear(self.nhid, self._nclasses)

    def forward(self, data, epoch, test):
        #values, timesteps, masks, lengths = data
        #timesteps, values = data
#         if test:
#             self.Controller.std = 0.05
#         else:
# #             self.Controller.std = self.epsilons[epoch]
#             pass
# #             self.Controller.std = 0.3
            
        #timesteps = timesteps.transpose(0, 1) # MAYBE
        #values = values.transpose(0, 1)
        #print(timesteps.shape)
        #print(values.shape)
        #T, B, V = values.shape
        self.means = torch.zeros((self._nclasses, self.nref)) # For saving class-wise means
        baselines = [] # Predicted baselines
        reference_timesteps = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        glimpses = []
        
        # --- initial glimpse ---
        l_t = torch.FloatTensor(self._bsz, 1).uniform_(0, 1)
        l_t.requires_grad = False
        hidden = self.initHidden(self._bsz)
        reference_timesteps.append(l_t.squeeze())

        self.greps = []
        for i in range(self.nref+1):
            #out, hidden, log_probs, l_t, b_t, glimpse = self.CATCell(timesteps, values, hidden, l_t)
            out, hidden, log_probs, l_t, b_t, glimpse = self.CATCell(data, hidden, l_t)
            log_pi.append(log_probs)
            baselines.append(b_t)
            reference_timesteps.append(l_t.squeeze())
            glimpses.append(glimpse)
            
        self.greps = torch.stack(self.greps).squeeze()
        self.glimpses = torch.stack(glimpses).squeeze().transpose(0, 1)[:, :-1] # B x R
        self.reference_timesteps = torch.stack(reference_timesteps).squeeze().transpose(0, 1)[:, :-2] # B x R
        self.baselines = torch.stack(baselines).squeeze().transpose(0, 1)[:, :-1] # B x R
        self.log_pi = torch.stack(log_pi).squeeze().transpose(0, 1)[:, :-1] # B x R
        logits = self.predict(out.squeeze())
        return logits
    
    #def CATCell(self, timesteps, values, h_prev, l):
    def CATCell(self, data, h_prev, l):
        #grep, glimpse = self.GlimpseNetwork(values, timesteps, masks, lengths, l)
        grep, glimpse = self.GlimpseNetwork(data, l)
        out, hidden = self.RNN(grep.unsqueeze(0), h_prev)
        self.greps.append(out)
        log_probs, l_next = self.Controller(out)
        b_t = self.BaselineNetwork(out)
        return out, hidden, log_probs, l_next, b_t, glimpse
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        for i in y.unique():
            self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)

        # --- compute reward ---
        predicted = torch.max(torch.softmax(logits, 1), 1)[1]
#         self.r = (predicted.float().detach() == y.float()).float()
        self.r = ((2*(predicted.float().detach() == y.float()).float())-1) # B x 1
        self.R = self.r.mean()
        self.r = self.r.unsqueeze(1).repeat(1, self.nref) # B x nref
        
        self.loss_c = F.cross_entropy(logits, y)
        self.loss_b = F.mse_loss(self.baselines, self.r)  # Baseline should approximate mean reward

        self.adjusted_reward = self.r# - self.baselines.detach()
        self.loss_r = torch.sum(-self.log_pi*self.adjusted_reward, 1) # sum over time
        self.loss_r = 0.5*torch.mean(self.loss_r, 0) # mean over batch
        
        # --- putting it all together ---
        loss = self.loss_r + self.loss_c# + self.loss_b
        return loss

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

class Interpolator(Model):
    def __init__(self, config, data_setting, adapter="linear", nref=10):#nhid, nclasses, ninp, nlayers, R, adapter="linear"):
        super(Interpolator, self).__init__(config, data_setting)
        self.nref = nref
        self.r = torch.tensor(np.linspace(0, 1, nref), dtype=torch.float)
        self.r = self.r.unsqueeze(0).repeat(self._B, 1)

        # --- Load sub-networks ---
        if adapter == "gaussian":
            self.Interpolator = GaussianAdapter(self._ninp)
            self.NAME = "IPN"
        if adapter == "linear":
            self.Interpolator = LinearInterpolator()
            self.NAME = "LinearInterpolator"
        self.Discriminator = Discriminator(1*self._ninp, self.nhid, self._nclasses, self.nlayers)
    
    def forward(self, data, epoch, test):
        t, v = data
        glimpses = []
        self.means = torch.zeros((self._nclasses, self.nref)) # For saving class-wise means
        X_regular = self.Interpolator(t, v, self.r)
        if (self._ninp == 1) & (len(X_regular.shape) < 3):
            X_regular = X_regular.unsqueeze(2)
        X_regular = X_regular.transpose(0, 1)
        prediction = self.Discriminator(X_regular)
            
        self.glimpses = X_regular.transpose(0, 1).squeeze()
        return prediction
    
    def computeLoss(self, logits, y):
        # --- save class-specific means ---
        for i in y.unique():
            self.means[i] = self.glimpses[y == i].mean(0).unsqueeze(0)
        return F.cross_entropy(logits, y)

class IPN(Model):
    def __init__(self, nhid, nclasses, ninp, nlayers, R, BATCH_SIZE, REF_STEPS=None):
        super(IPN, self).__init__()
        self.nlayers = nlayers
        self.BATCH_SIZE = BATCH_SIZE
        self.nhid = nhid
        self.nclasses = nclasses
        self.reference_timesteps = torch.tensor(np.linspace(0, 1, R), dtype=torch.float)

        # --- Load sub-networks ---
        self.Discriminator = Discriminator(1*ninp, nhid, nclasses, nlayers)
        self.Interpolator = Interpolator(ninp)
    
    def forward(self, data):
        timesteps, values = data
        X_regular = self.Interpolator(timesteps, values, self.reference_timesteps)
        prediction = self.Discriminator(X_regular)
        return prediction
    
    def computeLoss(self, y_hat, y):
        if not self.criterion:
            self.criterion = torch.nn.CrossEntropyLoss()
        return self.criterion(y_hat, y)

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
