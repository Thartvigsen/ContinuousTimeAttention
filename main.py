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
    d = UWave()
    m = Interpolator(config, d.data_setting, adapter="gaussian", nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 1:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = UWave()
    m = Interpolator(config, d.data_setting, adapter="gaussian", nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 2:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = UWave()
    m = Interpolator(config, d.data_setting, adapter="gaussian", nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 3:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = UWave()
    m = Interpolator(config, d.data_setting, adapter="gaussian", nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 4:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = UWave()
    m = Interpolator(config, d.data_setting, adapter="gaussian", nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 5:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = UWave()
    m = CAT(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 6:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = UWave()
    m = CAT(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 7:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = UWave()
    m = CAT(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 8:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = UWave()
    m = CAT(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 9:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = UWave()
    m = CAT(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 10:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = UWave()
    m = PolicyFree(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 11:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = UWave()
    m = PolicyFree(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 12:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = UWave()
    m = PolicyFree(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 13:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = UWave()
    m = PolicyFree(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

if t == 14:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = UWave()
    m = PolicyFree(config, d.data_setting, nref=10)
    e = [Accuracy()]
    p = ExpConfig(d=d,
                  m=m,
                  e=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5)
    p.run()

