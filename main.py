from expConfig import *
from model import *
from dataset import *
from metric import *
from utils import ConfigReader
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')
args = parser.parse_args()

# parse parameters
t = args.taskid

c_reader = ConfigReader()

if t == 0:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = SeqLengthUniform(T=50, N=500, nref=50)
    m = CAT(config, d.data_setting, nhop=3, nsample=100, scaling_factor=5, explore=False, intensity=False)
    e = [Accuracy(), AUC_macro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 1:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = SeqLengthUniform(T=50, N=500, nref=50)
    m = CAT(config, d.data_setting, nhop=3, nsample=100, scaling_factor=5, explore=False, intensity=False)
    e = [Accuracy(), AUC_macro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 2:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = SeqLengthUniform(T=50, N=500, nref=50)
    m = CAT(config, d.data_setting, nhop=3, nsample=100, scaling_factor=5, explore=False, intensity=False)
    e = [Accuracy(), AUC_macro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 3:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = SeqLengthUniform(T=50, N=500, nref=50)
    m = CAT(config, d.data_setting, nhop=3, nsample=100, scaling_factor=5, explore=False, intensity=False)
    e = [Accuracy(), AUC_macro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 4:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = SeqLengthUniform(T=50, N=500, nref=50)
    m = CAT(config, d.data_setting, nhop=3, nsample=100, scaling_factor=5, explore=False, intensity=False)
    e = [Accuracy(), AUC_macro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

