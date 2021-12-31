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
            #"""MVSynth()""",
            #"""Computers()""",
            #"""PersonActivity()""",
            #"""InHospitalMortality()""",
            
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.1)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.2)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.3)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.4)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.5)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.6)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.7)""",
            #"""HawkesIrregularUCR('Computers', a_neg=0.8, a_pos=0.8)""",

            #"""MultiModalIrregularUCR('Computers', R=500)""",
            #"""MultiModalIrregularUCR('Computers', R=450)""",
            #"""MultiModalIrregularUCR('Computers', R=400)""",
            #"""MultiModalIrregularUCR('Computers', R=350)""",
            #"""MultiModalIrregularUCR('Computers', R=300)""",
            #"""MultiModalIrregularUCR('Computers', R=250)""",
            #"""MultiModalIrregularUCR('Computers', R=200)""",
            #"""MultiModalIrregularUCR('Computers', R=150)""",
            #"""MultiModalIrregularUCR('Computers', R=100)""",
            #"""MultiModalIrregularUCR('Computers', R=50)""",

            #"""SeqLengthUniform(T=200, N=5000, nref=200, signal_prop=0.01)""",
            #"""SeqLengthUniform(T=200, N=5000, nref=200, signal_prop=0.1)""",
            #"""SeqLengthUniform(T=200, N=5000, nref=200, signal_prop=0.2)""",
            #"""SeqLengthUniform(T=200, N=5000, nref=200, signal_prop=0.3)""",
            #"""SeqLengthUniform(T=200, N=5000, nref=200, signal_prop=0.4)""",
            #"""SeqLengthUniform(T=200, N=5000, nref=200, signal_prop=0.5)""",

            # Looping over nref
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=25)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=50)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=100)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=150)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=250)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=300)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=350)""", # RERUN
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=400)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=450)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)""",

            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.002, width=400, nref=100)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=400, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=300)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.002, width=200, nref=100)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.002, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.002, width=200, nref=300)""",
            #"""UWave2(nref=20)""",
            #"""UWave2(nref=40)""",
            #"""UWave2(nref=60)""",
            #"""UWave2(nref=80)""",
            #"""UWave2(nref=100)""",
            #"""UWave2(nref=120)""",
            #"""UWave2(nref=140)""",
            #"""UWave2(nref=160)""",
            #"""UWave2(nref=180)""",
            #"""UWave2(nref=200)""",

            # HYPERPARAMS
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:LYING_DOWN', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=500)""",

            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=200)""", # RUN
            #"""ExtraSensoryUser(label_name='label:FIX_running', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:LYING_DOWN', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:LYING_DOWN', threshold=0.001, width=200, nref=500)""",
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=500)""",
            #"""ExtraSensoryUser(label_name='label:FIX_running', threshold=0.001, width=200, nref=500)""",
            """ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=500)""", # RUN

            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=300)""", # RUN
            #"""ExtraSensoryUser(label_name='label:FIX_running', threshold=0.001, width=200, nref=300)""",
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=300)""",
            #"""ExtraSensoryUser(label_name='label:LYING_DOWN', threshold=0.001, width=200, nref=300)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=400)""", # RUN
            #"""ExtraSensoryUser(label_name='label:FIX_running', threshold=0.001, width=200, nref=400)""",
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=400)""",
            #"""ExtraSensoryUser(label_name='label:LYING_DOWN', threshold=0.001, width=200, nref=400)""",

            #"""ExtraSensoryUser(label_name='label:SITTING', threshold=0.003, width=200, nref=200)""",
            # BAD
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=200, nref=200)""",
            #"""ExtraSensoryUser(label_name='label:BICYCLING', threshold=0.002, width=200, nref=500)""", # STILL NEED TO RUN

            #"""MTable(T=500, N=5000, delta=0.01, nref=500)""", # 1% of timeline
            #"""MTable(T=500, N=5000, delta=0.02, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.03, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.04, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.05, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.06, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.07, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.08, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.09, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.1, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.11, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.12, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.13, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.14, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.15, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.16, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.17, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.18, nref=500)""",
            #"""MTable(T=500, N=5000, delta=0.19, nref=500)""",
        ]

        #self.metrics = """[Accuracy()]"""
        self.metrics = """[Accuracy(), AUC_micro()]"""

        self.models = [
            #"""RNN(config, d.data_setting)""", # RUN ALONE
            #"""RNNInterp(config, d.data_setting)""",
            #"""RNNSimple(config, d.data_setting)""",
            #"""RNN(config, d.data_setting)""",
            #"""RNNDelta(config, d.data_setting)""",
            #"""RNNDecay(config, d.data_setting)""",
            #"""GRU_D(config, d.data_setting)""",
            #"""IPN(config, d.data_setting, nref=200)""",
            #"""IPN(config, d.data_setting, nref=500)""",

            #"""IPN(config, d.data_setting, nref=20)""",
            #"""IPN(config, d.data_setting, nref=40)""",
            #"""IPN(config, d.data_setting, nref=60)""",
            #"""IPN(config, d.data_setting, nref=80)""",
            #"""IPN(config, d.data_setting, nref=100)""",
            #"""IPN(config, d.data_setting, nref=120)""",
            #"""IPN(config, d.data_setting, nref=140)""",
            #"""IPN(config, d.data_setting, nref=160)""",
            #"""IPN(config, d.data_setting, nref=180)""",


            #"""CAT(config, d.data_setting, nhop=3, nsample=0.01, scaling_factor=100, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.02, scaling_factor=50, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.04, scaling_factor=25, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.06, scaling_factor=16, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.08, scaling_factor=12, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.1, scaling_factor=10, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.12, scaling_factor=8, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.14, scaling_factor=7, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.16, scaling_factor=6, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.18, scaling_factor=5, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.2, scaling_factor=5, explore=False, intensity=False)""",

            #"""CAT(config, d.data_setting, nhop=3, nsample=0.22, scaling_factor=4, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.24, scaling_factor=4, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.26, scaling_factor=3, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.28, scaling_factor=3, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.30, scaling_factor=3, explore=False, intensity=False)""",

            #"""CAT(config, d.data_setting, nhop=3, nsample=0.32, scaling_factor=3, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.34, scaling_factor=3, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.36, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.38, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.40, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.42, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.44, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.46, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.48, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.50, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.52, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.54, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.56, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.58, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.60, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.62, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.64, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.66, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.68, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.70, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.72, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.74, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.76, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.78, scaling_factor=1, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.80, scaling_factor=1, explore=False, intensity=False)""",

            """PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",

            """PolicyFreeCAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            """PolicyFreeCAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",

            #"""CAT(config, d.data_setting, nhop=1, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=4, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=5, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=6, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=7, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=8, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=9, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=10, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",

            #"""CATMask(config, d.data_setting, nhop=3, nsample=0.10, scaling_factor=10, explore=False, intensity=False)""",
            #"""CATMask(config, d.data_setting, nhop=3, nsample=0.10, scaling_factor=10, explore=False, intensity=True)""",

            # START
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.1, nemb=50, scaling_factor=10, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.1, nemb=50, scaling_factor=10, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.2, nemb=50, scaling_factor=5, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.2, nemb=50, scaling_factor=5, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.5, nemb=50, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=2, nsample=0.5, nemb=50, scaling_factor=2, explore=False, intensity=True)""",

            #"""CAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.05, nemb=50, scaling_factor=20, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.1, nemb=50, scaling_factor=10, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.1, nemb=50, scaling_factor=10, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.2, nemb=50, scaling_factor=5, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.2, nemb=50, scaling_factor=5, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.5, nemb=50, scaling_factor=2, explore=False, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=0.5, nemb=50, scaling_factor=2, explore=False, intensity=True)""",
            # STOP

            #"""CAT(config, d.data_setting, nhop=3, nsample=0.1, scaling_factor=10, explore=False, intensity=True)""",

            #"""CAT(config, d.data_setting, nhop=3, nsample=10, explore=True, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=20, explore=True, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=20, scaling_factor=3, explore=False, intensity=True)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=10, explore=True, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=20, explore=True, intensity=False)""",
            #"""CAT(config, d.data_setting, nhop=3, nsample=20, scaling_factor=3, explore=False, intensity=False)""",
            #"""PolicyFree(config, d.data_setting, nref=10)""",
            #"""NCDE(config, d.data_setting)""",
            #"""Transformer(config, d.data_setting)""",
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
                exclude_list=["NAME", "name", "desc", "training", "bsz", "intensities",
                              "test_ix", "train_ix", "values", "timesteps", "signal_length",
                              "deltas", "masks", "epsilons", "val_ix", "r",
                              "M", "t0", "stop", "table",
                              "device", "lengths", "ids", "values", "masks",
                              "timesteps", "signal_start", "signal_end",
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

def writeCSVRow(row, name, path="./", round=True):
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
    path = os.path.join(path, name)
    with open(f"{path}.csv", "a+") as csvfile:
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

# I really want the likelihood of sampling to appear in clumps.. is that a markov process?
class MarkovChain(object):
    def __init__(self, transition_prob):
        """
        Initialize the MarkovChain instance.
 
        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition 
            probabilities in Markov Chain. 
            Should be of the form: 
                {'state1': {'state1': 0.1, 'state2': 0.4}, 
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys())
 
    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states, 
            p=[self.transition_prob[current_state][next_state] 
               for next_state in self.states]
        )
 
    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
 
        Parameters
        ----------
        current_state: str
            The state of the current random variable.
 
        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

def instantiate_markov(pos_to_neg, neg_to_pos):
    """
    Instantiate a Markov Chain.

    Parameters
    ----------
    pos_to_neg: float between 0. and 1.
        Probability of transitioning to negative state 
        if current state is positive
    neg_to_pos: float between 0. and 1.
        Probability of transitioning from positivite state
        if current state is negative
    """
    trans_prob = {}
    trans_prob['Pos'] =  {'Pos': 1. - pos_to_neg, 'Neg': pos_to_neg}
    trans_prob['Neg'] =  {'Pos': neg_to_pos, 'Neg': 1. - neg_to_pos}
    markov_chain = MarkovChain(transition_prob=trans_prob)
    
    return markov_chain

def get_markov_sequence(markov_chain, seq_length, start_state='Pos'):
    """
    Get a sequence generated from markov chain.

    Parameters
    ----------
    markov_chain: Instantiated MarkovChain 
        Output of instantiante_markov
    seq_length: an int.
        Length of desired output sequence
    start_state: string, either 'Pos' or 'Neg'
        Indicates whether markov chain starts in pos or neg state 
    """
    seq_string = markov_chain.generate_states('Pos', seq_length)
    seq_binary = [ int(i == 'Pos') for i in seq_string]
    
    return np.array(seq_binary)

def hawkes_intensity(mu, alpha, points, t):
    """Find the hawkes intensity:
    mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )
    """
    p = np.array(points)
    p = p[p <= t]
    p = np.exp(p - t) * alpha
    return mu + np.sum(p)

def simulate_hawkes(mu, alpha, num_instances):
    t = 0
    points = []
    all_samples = []
    while len(points) < num_instances:
        m = hawkes_intensity(mu, alpha, points, t)
        s = np.random.exponential(scale = 1/m)
        ratio = hawkes_intensity(mu, alpha, points, t + s) / m
        if ratio >= np.random.uniform():
            points.append(t + s)
        all_samples.append(t + s)
        t = t + s
    return np.array(points), all_samples
