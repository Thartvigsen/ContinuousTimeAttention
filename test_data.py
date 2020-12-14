#import matplotlib.pyplot as plt
import torch
from modules import *
from dataset import *
from model import CAT
from metric import *
from expConfig import *

pink = "#FA4563"
dark_blue = "#065472"
light_blue = "#63bae7"

config = {
    "model"   : {
        "hidden_dim"          : 50,
        "embed_dim"           : 20,
        "dropout_probability" : 0.2,
        "n_layers"            : 1,
        "rnn_type"            : "GRU",
        "ortho_init"          : False,
        "noisin"              : False,
    },
    "training" : {
        "batch_size"          : 32,
        "n_epochs"            : 10,
        "learning_rate"       : 1e-2,
        "num_workers"         : 6,
        "use_scheduler"       : False,
        "scheduler_param"     : 0.95,
        "use_cuda"            : False,
        "resume"              : False,
        "split_props"         : [0.8, 0.1, 0.1],
        "loss_name"           : "crossentropy",
        "multilabel"          : True,
        "optimizer_name"      : "adam",
        "device"              : "cpu",
        "checkpoint"          : 50,
    },
}

d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, nref=500)

##d = MVSynth()
#M = CAT(config, d.data_setting, ngran=10)
#e = [Accuracy()]
#E = ExpConfig(d, M, e, config)
#E.run()

#loader = torch.utils.data.DataLoader(d, batch_size=16, drop_last=True)
#for x, y in loader:
#    v, t, m, l = x
#
#logits = M(x, epoch=0, test=False)
##print(v.shape, t.shape, m.shape, l.shape)
##
##G = MVGlimpseNetwork(ninp=d.data_setting["N_FEATURES"], nhid=10, ngran=2, nglimpse=7, gwidth=1)
##R = torch.linspace(0, 1, 5)[None, :].repeat(16, 1)
##
##grep, middle = G(v, t, m, l, R[:, 3].unsqueeze(1))
##
#fig, ax = plt.subplots(figsize=(16, 8))
#t = M.GlimpseNetwork.t
#v = M.GlimpseNetwork.v
#r = M.GlimpseNetwork.r
#g = M.GlimpseNetwork.g
#
#ax.scatter(t, v, c=dark_blue, marker="x")
#ax.errorbar(r[:, -1], g, marker="o", c=pink)
#ax.axvline(r[len(r)//2, 0], c= "gray", ls=":", lw=2)
#ax.set_xlim([0, 48])
#plt.show()
##
