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
    d = PhysioNet2()
    m = CAT(config, d.data_setting, nhop=2, nsample=0.05, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()
