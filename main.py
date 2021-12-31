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
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 1:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 2:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 3:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 4:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 5:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 6:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 7:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 8:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 9:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 10:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 11:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 12:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 13:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 14:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 15:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 16:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 17:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 18:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 19:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 20:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 21:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 22:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 23:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 24:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 25:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 26:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 27:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 28:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 29:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 30:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 31:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 32:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 33:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 34:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 35:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 36:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 37:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 38:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 39:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 40:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 41:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 42:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 43:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 44:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 45:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 46:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 47:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 48:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 49:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 50:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 51:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 52:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 53:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 54:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 55:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 56:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 57:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 58:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 59:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 60:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 61:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 62:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 63:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 64:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 65:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 66:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 67:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 68:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 69:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 70:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 71:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 72:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 73:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 74:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 75:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 76:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 77:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 78:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 79:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 80:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 81:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 82:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 83:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 84:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 85:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 86:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 87:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 88:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 89:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 90:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 91:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 92:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 93:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 94:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 95:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 96:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 97:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 98:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

if t == 99:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)
    m = PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)
    e = [Accuracy(), AUC_micro()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  iteration=t%5)
    p.run()

