import torch
import numpy as np
import csv
import os.path
import os
from itertools import product

class MainWriter(object):
    def __init__(self):
        self.header = ("from expConfig import *\n"
                       "from model import *\n"
                       "from dataset import *\n"
                       "from metric import *\n"
                       "from utils import ConfigReader\n"
                       "import argparse\n\n"
        
                       "parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)\n"
                       "parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')\n"
                       "args = parser.parse_args()\n\n"
        
                       "# parse parameters\n"
                       "t = args.taskid\n\n"
        
                       "c_reader = ConfigReader()\n\n")
        
        # --- set experimental configs ---
        self.datasets = [
            #"""PhysioNet()""",
            #"""UWave()""",
            """InHospitalMortality()""",
            """MVSynth()""",
        ]

        self.metrics = """[Accuracy(), AUC_micro()]"""

        self.models = [
            #"""Interpolator(config, d.data_setting, adapter="linear", nref=10)""",
            #"""Interpolator(config, d.data_setting, adapter="linear", nref=140)""",
            #"""Interpolator(config, d.data_setting, adapter="gaussian", nref=10)""",
            #"""Interpolator(config, d.data_setting, adapter="gaussian", nref=140)""",
            """RNN(config, d.data_setting)""",
            """CAT(config, d.data_setting, nref=10)""",
            #"""PolicyFree(config, d.data_setting, nref=10)""",
        ]

        self.n_iterations = 5

    def write(self):
        t = 0
        for d in self.datasets:
            for model in self.models:
                for i in range(self.n_iterations):
                    text = ("if t == {0}:\n"
                            "    # --- Iteration: {1} ---\n"
                            "    config = c_reader.read(t)\n"
                            "    d = {2}\n"
                            "    m = {3}\n"
                            "    e = {4}\n"
                            "    p = ExpConfig(d=d,\n"
                            "                  m=m,\n"
                            "                  e=e,\n"
                            "                  config=config,\n"
                            #"                  data_setting=d.data_setting,\n"
                            "                  iteration=t%{5})\n"
                            "    p.run()\n\n".format(t,
                                                     i+1,
                                                     d,
                                                     model,
                                                     self.metrics,
                                                     self.n_iterations))
                    self.header += text
                    t += 1

        with open("main.py", "w") as f:
            f.write(self.header)
            f.close()

#assert hasattr(
#    torch, "bucketize"), "Need torch >= 1.7.0; install at pytorch.org"

class RegularGridInterpolator:
    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * \
                torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return numerator / denominator
def attrToString(obj, prefix,
                exclude_list=["NAME", "name", "desc", "training", "bsz",
                              "test_ix", "train_ix", "values", "timesteps",
                              "deltas", "masks", "epsilons", "val_ix", "r",
                              "device", "lengths", "ids", "values", "masks",
                              "timesteps",
                              "data", "labels", "signal_locs", "round",
                              "train", "test", "train_labels", "test_labels",
                              "data_setting", "y_train", "y_test", "seq_length"]):
    """Convert the attributes of an object into a unique string of
    path for result log and model checkpoint saving. The private
    attributes (starting with '_', e.g., '_attr') and the attributes
    in the `exclude_list` will be excluded from the string.
    Args:
        obj: the object to extract the attribute values prefix: the
        prefix of the string (e.g., MODEL, DATASET) exclude_list:
        the list of attributes to be exclude/ignored in the
        string Returns: a unique string of path with the
        attribute-value pairs of the input object
    """
    out_dir = prefix #+"-"#+obj.name
    for k, v in obj.__dict__.items():
        if not k.startswith('_') and k not in exclude_list:
            out_dir += "/{}-{}".format(k, ",".join([str(i) for i in v]) if type(v) == list else v)
    return out_dir

def writeCSVRow(row, name, path="./", round=False):
    """
    Given a row, rely on the filename variable to write
    a new row of experimental results into a log file

    New Idea: Write a header for the csv so that I can
    clearly understand what is going on in the file

    Parameters
    ----------
    row : list
        A list of variables to be logged
    name : str
        The name of the file
    path : str
        The location to store the file
    """

    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def exponentialDecay(N):
    tau = 1
    tmax = 7
    t = np.linspace(0, tmax, N)
    y = torch.tensor(np.exp(-t/tau), dtype=torch.float)
    return y#/5.

class ConfigReader(object):
    def __init__(self):
        self.path = "./configs/"

    def read(self, t):
        """Read config file t as a dictionary.
        If the requested file does not exist, load the base
        configuration file instead.
        """
        if os.path.isfile(self.path+"input_{}.txt".format(t)):
            s = open(self.path+"input_{}.txt".format(t), "r").read()
        else:
            print("Loading base config file.")
            s = open(self.path+"base_config.txt", "r").read()
        return eval(s)

def hardSigma(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output

def printParams(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def makedir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except:
        pass
