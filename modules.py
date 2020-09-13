from torch import nn
import numpy as np
from torch.distributions import Normal
import torch
from scipy import interpolate
from utils import RegularGridInterpolator

class Controller(nn.Module):
    """Look at hidden state and decide where to move the sensor"""
    def __init__(self, ninp, std):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.std = std
        nemb = 10
        self.fc1 = nn.Linear(ninp, nemb)
        self.fc2 = nn.Linear(nemb, 1)
        
    def forward(self, h_t):
        #print("H: ", h_t)
        feat = torch.relu(self.fc1(h_t.squeeze(0)))
        self.mu = torch.sigmoid(self.fc2(feat))
        distribution = Normal(self.mu, self.std)
        l_t = distribution.rsample()
        l_t = l_t.detach()

        l_t = torch.clamp(l_t, 0, 1)
        log_pi = distribution.log_prob(l_t)
        self.log_pi = log_pi
        return log_pi, l_t

class BaselineNetwork(nn.Module):
    """
    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """

    def __init__(self, ninp, nout):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)

    def forward(self, x):
#         b = torch.relu(self.fc(x))
        b = self.fc(x)
        return b 

class GaussianAdapter(nn.Module):
    def __init__(self, N_FEATURES, a=50, k=10.):
        super(GaussianAdapter, self).__init__()
        self.N_FEATURES = N_FEATURES
        self.k = k # Controls how much alpha gets scaled in coarse interpolation
        
        # --- Parameters ---
        self.alpha = [a]
        self.alpha = torch.nn.Parameter(torch.tensor(self.alpha, dtype=torch.float), requires_grad=True)
        self.rho = torch.nn.Parameter(torch.ones((N_FEATURES, N_FEATURES)))
        
    def squaredExponentialKernel(self, r, t, alpha):
        dist = torch.exp(torch.mul(-alpha, 1*torch.sub(r, t).pow(2)))
        mask = torch.zeros_like(t)
        mask[t > 0] = 1 #
        return dist*mask + 1e-07 # If dist goes to 0, still need to divide

    def intensity(self, reference_timesteps, timesteps, alpha, reduce=True):
        if len(reference_timesteps.shape) == 2: # Already batched
            reference_broadcast = torch.repeat_interleave(reference_timesteps.unsqueeze(2), timesteps.shape[1], dim=2)
        elif len(reference_timesteps.shape) == 1: # Just one vector
            reference_broadcast = reference_timesteps.view(-1, 1).repeat(1, timesteps.shape[1]).repeat(timesteps.shape[0], 1, 1)
        else:
            print("wrong reference timestep shape")
        
        timesteps = timesteps.unsqueeze(1) # Add R dim to real timesteps
        reference_broadcast = reference_broadcast.unsqueeze(3).repeat(1, 1, 1, timesteps.shape[-1])
        dist = self.squaredExponentialKernel(reference_broadcast, timesteps, alpha)
        if reduce:
            return dist.sum(2)
        else:
            return dist

    def interpolate(self, reference_timesteps, timesteps, values, smooth):
        """Compute new values for each reference timestep"""
        if smooth:
            a = self.alpha
        else:
            a = self.k*self.alpha
        lam = self.intensity(reference_timesteps, timesteps, a, reduce=True)
        weights = self.intensity(reference_timesteps, timesteps, a, reduce=False)
        return torch.sum(weights * values.unsqueeze(1), 2)/lam
    
    def crossDimensionInterpolation(self, lam, sigma):
        return torch.matmul(lam*sigma, self.rho)/lam.sum(2).unsqueeze(2)

    def forward(self, timesteps, values, reference_timesteps):
        """Params are updated as the model learns"""
        lam = self.intensity(reference_timesteps, timesteps, self.alpha, reduce=True)
        self.lam = lam
        self.smooth = self.interpolate(reference_timesteps, timesteps, values, smooth=True)
        #self.coarse = self.interpolate(reference_timesteps.detach(), timesteps.detach(), values.detach(), smooth=False)
        return self.smooth

class LinearInterpolator(nn.Module):
    def __init__(self):
        super(LinearInterpolator, self).__init__()
    
    def forward(self, t, v, r):
        vals = []
        for b in range(len(t)): # Assume t is of shape (B x T x V)
            f = interpolate.interp1d(t[b].numpy().squeeze(), v[b].numpy().squeeze(), fill_value=(v[b][0].numpy().squeeze(), v[b][-1].numpy().squeeze()), bounds_error=False)
            v_new = f(r[b].numpy().squeeze())
            vals.append(v_new)
        return torch.from_numpy(np.stack(vals).squeeze())

class GlimpseNetwork(nn.Module):
    def __init__(self, ninp, nhid, ngran, nglimpse, gwidth, adapter="linear", gtype="flatten"):
        super(GlimpseNetwork, self).__init__()
        self.ngran = ngran
        self.nglimpse = nglimpse
        self.gwidth = gwidth
        if adapter == "gaussian":
            self.Interpolator = GaussianAdapter(self._ninp, a=200)
        if adapter == "linear":
            self.Interpolator = LinearInterpolator()
        self.gtype = gtype
        
        # mappings
        #self.fc = nn.Linear(ngran*nglimpse, nhid) # Assume Flattened input
        self.fc = nn.Linear(ngran*nglimpse+1, nhid) # Assume Flattened input

    def getGlimpseTimesteps(self, l_t):
        lin = []
        for i in range(0, self.ngran):
            lin.append(torch.linspace(-self.gwidth*((i*5)+1)/2., self.gwidth*((i*5)+1)/2., self.nglimpse).unsqueeze(0).repeat(l_t.shape[0], 1))
        lin = torch.stack(lin).transpose(0, 2)
        return lin + l_t.unsqueeze(0)#.unsqueeze(2)
    
    def denormalize(self, T, l_t):
#         print("T:", T.shape)
#         print("l_t:", l_t.shape)
        # Range [-1, 1] -> [0, T]
        return ((0.5*(l_t + 1.0))*T)

    def glimpseSensor(self, timesteps, values, l_t):
        # timesteps is of shape [B x T x V]
        timesteps = timesteps.transpose(0, 1)
        values = values.transpose(0, 1)
#         l_t = self.denormalize(timesteps.max(1)[0].max(1)[0].unsqueeze(1), l_t) # Make l_t in proper range
        ref_steps = self.getGlimpseTimesteps(l_t) # T x B x NGRAN
#         ref_steps = ref_steps.transpose(0, 1)[:, :, 0] # B x T x NGRAN
        all_glimpses = []
        self.ref_steps = ref_steps
        for G in range(self.ngran):
            self.t = timesteps
            self.v = values
            glimpse = self.Interpolator.forward(timesteps, values, ref_steps[:, :, G].transpose(0, 1))
            all_glimpses.append(glimpse)
        glimpses = torch.stack(all_glimpses).transpose(0, 2).transpose(0, 1)
        # Each new value has been computed w.r.t. all old values
        return glimpses

    def RNNCell(self, x):
        out, hidden = self.RNN(x)
        return out[:, -1, :]
    
    def forward(self, timesteps, values, l_t):
        glimpse = self.glimpseSensor(timesteps, values, l_t) # It collects 10 points centered around l_t (B x nglimpse)
        if len(glimpse.shape) > 3:
            glimpse = glimpse.squeeze()

        self.glimpse = glimpse
        g = glimpse.reshape(glimpse.shape[0], -1)
        self.fglimpse = g
        grep = self.fc(torch.cat((g, l_t), dim=1)) # SHOULD GLIMPSE OVER INDEP RESOLUTIONS, THEN WEIGHTED SUM REPS FROM SAME MAPPER?
        #grep = self.fc(g) # SHOULD GLIMPSE OVER INDEP RESOLUTIONS, THEN WEIGHTED SUM REPS FROM SAME MAPPER?
        return grep, g[:, g.shape[1]//2]#, glimpse.mean(-2).squeeze()

class Discriminator(nn.Module):
    def __init__(self, N_FEATURES, HIDDEN_DIMENSION, N_CLASSES, N_LAYERS):
        """Classify regular time series"""
        super(Discriminator, self).__init__()
        self.N_CLASSES = N_CLASSES
        self.HIDDEN_DIMENSION = HIDDEN_DIMENSION
        self.N_LAYERS = N_LAYERS
        
        # --- mappings ---
        self.RNN = nn.LSTM(N_FEATURES, HIDDEN_DIMENSION)
        self.out = nn.Linear(HIDDEN_DIMENSION, N_CLASSES)
        
        # --- nonlinearities ---
        self.output_nonlin = nn.Softmax(dim=1)
        
    def forward(self, X):
        T, B, V = X.shape
        # --- initialize hidden state ---
        state = (torch.zeros(self.N_LAYERS, B, self.HIDDEN_DIMENSION),
                 torch.zeros(self.N_LAYERS, B, self.HIDDEN_DIMENSION))
        
        # --- inference ---
        hidden, state = self.RNN(X, state)
        logits = self.out(hidden[-1])
        return logits

class LinearInterpolator(object):
    def __init__(self):
        pass
    
    def forward(self, t, v, r):
        v, t = v.numpy(), t.numpy()
        f = interpolate.interp1d(t, v, fill_value=(v[0], v[-1]), bounds_error=False)
        v_new = f(r.numpy())
        return torch.from_numpy(v_new).float()
    
class MVGlimpseNetwork(nn.Module):
    def __init__(self, ninp, nhid, ngran, nglimpse, gwidth, gtype="flatten"):
        super(MVGlimpseNetwork, self).__init__()
        self.ngran = ngran
        self.nglimpse = nglimpse
        self.gwidth = gwidth
#         self.Interpolator = Interpolator(ninp, a=200) # Bigger alpha means sharper
        self.Interpolator = LinearInterpolator()
        self.gtype = gtype

        self.fc = nn.Linear(ninp*ngran*nglimpse+1, nhid) # Assume Flattened input

    def getGlimpseTimesteps(self, l_t):
        lin = []
        for i in [1, 5]:
            lin.append(torch.linspace(-self.gwidth*i/2., self.gwidth*i/2., self.nglimpse))
        lin = torch.stack(lin).transpose(0, 1)
        return lin + l_t

    def MVGlimpseSensor(self, t, v, m, r):
        new_vals = []
        for V in range(v.shape[1]):
            time = [t[m[:, V] == 1].squeeze()]
            vals = v[m[:, V] == 1, V]
            GI = RegularGridInterpolator(time, vals)
            new_vals.append(GI(r))
        new_vals = torch.stack(new_vals)
        return new_vals.T

    def glimpseSensor(self, timesteps, values, l_t):
        ref_steps = self.getGlimpseTimesteps(l_t)
        ref_steps = self.denormalize(timesteps.max(0)[0], ref_steps) # Make l_t in proper range
        self.r = ref_steps
        all_glimpses = []
        self.ref_steps = ref_steps
        for G in range(self.ngran):
            self.t = timesteps
            self.v = values
            glimpse = self.Interpolator.forward(timesteps, values, ref_steps[:, G])
            self.g = glimpse
            all_glimpses.append(glimpse)
        glimpses = torch.stack(all_glimpses).reshape(1, -1)
        return glimpses

    def denormalize(self, T, l_t):
        return l_t*T
        #return ((0.5*(l_t + 1.0))*T)
    
    def forward(self, data, l_t):
        # Input: Values, masks, lengths, reference timesteps
        vals, time, masks, lengths = data
        B, T, V = vals.shape
        glimpses = []
        for b in range(B):
            b_glimpses = []
            vals_i = vals[b]
            time_i = time[b]
            masks_i = masks[b]
            l = l_t[b]
            for v in range(0, V): # Chop off timesteps
                #l = self.denormalize(time_i.max(0)[0], l_t[b]) # Make l_t in proper range

                #print(torch.nonzero(masks_i[:, v]))
                #val_sum = len(torch.nonzero(masks_i[:, v]))
                #v_in = vals_i[masks_i[:, v] == 1, v]
                #t_in = time_i[masks_i[:, v] == 1]
                num_vals = masks_i[:, v].sum()
                if num_vals > 1:
                    #print("2+VAL")
                    v_in = vals_i[masks_i[:, v] == 1, v]
                    t_in = time_i[masks_i[:, v] == 1]
                    #print("value sum: ", v_in.sum())
                    #print("time sum: ", t_in.sum())
                    glimpse = self.glimpseSensor(t_in, v_in, l)
                    #print("inner glimpse sum: ", glimpse.sum())
                    if len(glimpse.shape) > 3:
                        glimpse = glimpse.squeeze()
                elif num_vals == 1:
                    # Just return the value
                    #print("1VAL")
                    glimpse = torch.ones(1, self.nglimpse*self.ngran)*torch.unique(vals_i[masks_i[:, v]==1, v]).float()
                else: # Variable is totally missing
                    # Variable is totally missing
                    #print("ELSE")
                    glimpse = torch.zeros(1, self.nglimpse*self.ngran)
                #print("Glimpse sum: ", glimpse.sum())
                #if torch.isnan(glimpse.sum()):
                    #print(masks_i[:, v])
                    #print(vals_i[:, v])
                    #print(time_i)
                    #print(v_in)
                    #print(t_in)
                    #print("num vals: ", num_vals)
                    #print("value sum: ", v_in.sum())
                    #print("time sum: ", t_in.sum())
                    #assert 2 == 3
                b_glimpses.append(glimpse)
            #assert 2 == 3
            b_glimpses = torch.stack(b_glimpses).squeeze()
            glimpses.append(b_glimpses)
        glimpses = torch.stack(glimpses)
        g = glimpses.reshape(B, -1)
        #print("sum g: ", g.sum(1))
        #print("g: ", g)
        #print("l_t: ", l_t)
        #print()
        self.fglimpse = g
        grep = self.fc(torch.cat((g, l_t), dim=1))
        return grep, g[:, g.shape[1]//2]

#class Controller(nn.Module):
#    """Look at hidden state and decide where to move the sensor"""
#    def __init__(self, ninp, std):
#        super(Controller, self).__init__()
#
#        # --- Mappings ---
#        self.std = std
#        self.fc = nn.Linear(ninp, 1)
#        
#    def forward(self, x):
#        mu = torch.tanh(self.fc(x))
#        #mu = (1-self._epsilon)*mu + self._epsilon*(2*torch.rand(x.shape[0], 1)-1) # Explore/exploit
#        self.mu = mu
#        l_t = Normal(mu, self.std).rsample().detach()
#        log_pi = Normal(mu, self.std).log_prob(l_t)
#        l_t = torch.clamp(l_t, -1, 1)
#        return log_pi, l_t

#class Interpolator(nn.Module):
#    def __init__(self, N_FEATURES, k=10.):
#        super(Interpolator, self).__init__()
#        self.N_FEATURES = N_FEATURES
#        self.k = k # Controls how much alpha gets scaled in coarse interpolation
#        
#        self.alpha = [5.]
#        self.alpha = torch.nn.Parameter(torch.tensor(self.alpha, dtype=torch.float), requires_grad=True)
#        self.rho = torch.nn.Parameter(torch.ones((N_FEATURES, N_FEATURES)))
#    
#        
#    def squaredExponentialKernel(self, r, t, alpha):
#        dist = torch.exp(torch.mul(-alpha, 100*torch.sub(r, t).pow(2)))
#        return dist + 1e-07 # If dist goes to 0, adding a small value prevents nans
#
#    def intensity(self, reference_timesteps, timesteps, alpha, reduce=True):
#        if len(reference_timesteps.shape) == 2: # Already batched
#            reference_broadcast = torch.repeat_interleave(reference_timesteps.unsqueeze(2), timesteps.shape[1], dim=2)
#        elif len(reference_timesteps.shape) == 1: # Just one vector
#            reference_broadcast = reference_timesteps.view(-1, 1).repeat(1, timesteps.shape[1]).repeat(timesteps.shape[0], 1, 1)
#        else:
#            print("wrong reference timestep shape")
#
#        dist = self.squaredExponentialKernel(reference_broadcast, timesteps.unsqueeze(1).squeeze(3), alpha)
#        if reduce:
#            return dist.sum(2)
#        else:
#            return dist
#
#    def interpolate(self, reference_timesteps, timesteps, values, smooth):
#        """Compute new values for each reference timestep"""
#        if smooth:
#            a = self.alpha
#        else:
#            a = self.k*self.alpha
#        lam = self.intensity(reference_timesteps, timesteps, a, reduce=True)
#        weights = self.intensity(reference_timesteps, timesteps, a, reduce=False)
#        return torch.sum(weights * values.unsqueeze(1), 2)/lam
#    
#    def crossDimensionInterpolation(self, lam, sigma):
#        return torch.matmul(lam*sigma, self.rho)/lam.sum(2).unsqueeze(2)
#
#    def forward(self, timesteps, values, reference_timesteps):
#        """Params are updated as the model learns"""
#        # For each feature, I need to
#        intensities = []
#        smooth_interpolations = []
#        coarse_interpolations = []
#        for i in range(values.shape[2]):
#            lam = self.intensity(reference_timesteps, timesteps, self.alpha, reduce=True)
#            intensities.append(lam)
#            self.smooth = self.interpolate(reference_timesteps, timesteps, values[:, :, i], smooth=True)
#            self.coarse = self.interpolate(reference_timesteps.detach(), timesteps.detach(), values[:, :, i].detach(), smooth=False)
#            smooth_interpolations.append(self.smooth)
#            coarse_interpolations.append(self.coarse)
#        intensities = torch.transpose(torch.stack(intensities), 0, 2)
#        smooth_interpolations = torch.transpose(torch.stack(smooth_interpolations), 0, 2)
#        self.interpolated = smooth_interpolations
#        return smooth_interpolations

#class GlimpseNetwork(nn.Module):
#    def __init__(self, ninp, nhid, ngran, nglimpse, gwidth, gtype="flatten"):
#        super(GlimpseNetwork, self).__init__()
#        self.ngran = ngran
#        self.nglimpse = nglimpse
#        self.gwidth = gwidth
#        self.Interpolator = Interpolator(ninp)
#        self.gtype = gtype
#        
#        if self.gtype == "fixed_weights":
#            self.weights = torch.nn.Parameter(torch.rand(ninp, 1))
#            self.fc = nn.Linear(ngran*nglimpse+1, nhid) # Assume Flattened input
#        elif self.gtype == "RNN":
#            self.RNN = torch.nn.LSTM(ninp*ngran, 10, batch_first=True)
#            self.fc = nn.Linear(10+1, nhid) # Assume Flattened input
#        else:
#            self.fc = nn.Linear(ninp*ngran*nglimpse+1, nhid) # Assume Flattened input
#
#    def getGlimpseTimesteps(self, l_t):
#        # l_t is of shape [B x 1]
#        lin = []
#        for i in range(0, self.ngran):
#            lin.append(torch.linspace(-self.gwidth*(i+1)/2., self.gwidth*(i+1)/2., self.nglimpse).unsqueeze(0).repeat(l_t.shape[0], 1))
#        lin = torch.stack(lin).transpose(0, 2)
#        return lin + l_t.unsqueeze(0)#.unsqueeze(2)
#    
#    def denormalize(self, T, l_t):
#        # Range [-1, 1] -> [0, T]
#        return (T*(l_t + 1.0)/2.)
#
#    def glimpseSensor(self, timesteps, values, l_t):
#        # timesteps is of shape [B x T x V]
#        timesteps = timesteps.transpose(0, 1)
#        values = values.transpose(0, 1)
#        l_t = self.denormalize(timesteps.max(1)[0].max(1)[0].unsqueeze(1), l_t) # Make l_t in proper range
#        ref_steps = self.getGlimpseTimesteps(l_t) # T x B x NGRAN
#        ref_steps = ref_steps.transpose(0, 1) # B x T x NGRAN
#        self.ref_steps = ref_steps
#        glimpses = (self.Interpolator(timesteps, values, ref_steps[:, :, 0]))
#        #return torch.stack(glimpses).squeeze()#.transpose(1, 2)# Stack over variables.transpose(0, 3)
#        return glimpses
#    
#    def RNNCell(self, x):
#        out, hidden = self.RNN(x)
#        return out[:, -1, :]
#    
#    def forward(self, timesteps, values, l_t):
#        l_t = torch.rand_like(l_t)
#        glimpse = self.glimpseSensor(timesteps, values, l_t) # It collects 10 points centered around l_t (B x nglimpse)
#        if len(glimpse.shape) > 3:
#            glimpse = glimpse.squeeze()
#        # flatten
#        self.glimpse = glimpse
#        self.t = timesteps
#        self.v = values
#        if self.gtype == "flatten":
#            g = glimpse.reshape(glimpse.shape[0], -1)
#        elif self.gtype == "fixed_weights":
#            g = torch.matmul(glimpse, self.weights).squeeze()
#        elif self.gtype == "attention":
#            g = self.attentionWeights(glimpse)
#        elif self.gtype == "RNN":
#            g = self.RNNCell(glimpse)
#        else:
#            print("bad gtype")
#        self.fglimpse = g
#        #print(g.mean())
#        if torch.isnan(g.mean()):
#            assert 2 == 3
#        grep = self.fc(torch.cat((g, l_t), dim=1).detach()) # SHOULD GLIMPSE OVER INDEP RESOLUTIONS, THEN WEIGHTED SUM REPS FROM SAME MAPPER?
##         grep = torch.relu(self.fc1(torch.cat((flat_glimpse, l_t), dim=1))) # SHOULD GLIMPSE OVER INDEP RESOLUTIONS, THEN WEIGHTED SUM REPS FROM SAME MAPPER?
#        return grep, g[:, g.shape[1]//2]#, glimpse.mean(-2).squeeze()
