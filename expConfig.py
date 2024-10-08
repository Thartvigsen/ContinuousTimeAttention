from utils import *

import torch
import numpy as np
import torch.nn as nn
import math
import torch.optim as optim
import csv
import os
import itertools
import json
import time
import shutil
import gc
from abc import ABCMeta, abstractmethod
from utils import *
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence
np.random.seed(3)

class ExpConfig():
    """
    Combining data, model, and evaluation metrics.
    Train the specified model on the training data,
    then testing the model on the testing data,
    evaluate results, and write them into logging files
    """

    def __init__(self, d, m, e, config, iteration=0):
        # --- load model pieces ---
        self.dataset = d
        self.model = m
        self.metrics = e
        self.iter = iteration

        # --- unpack hyperparameters ---
        self.BATCH_SIZE = config["training"]["batch_size"]
        self.SCHEDULE_LR = config["training"]["use_scheduler"]
        self._GAMMA = config["training"]["scheduler_param"]
        self.LEARNING_RATE = config["training"]["learning_rate"]
        self.resume = config["training"]["resume"]
        self.optimizer_name = config["training"]["optimizer_name"]
        self._checkpoint = config["training"]["checkpoint"]
        self.N_EPOCHS = config["training"]["n_epochs"]
        self.NUM_WORKERS = config["training"]["num_workers"]
        self.split_props = config["training"]["split_props"]
        self.bootstrap = config["eval"]["bootstrap"]

        # --- CUDA ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.setDevice(self.device)
        self.model = self.model.to(self.device)
        #self.dataset.data = self.dataset.data.to(device)
        #self.dataset.labels = self.dataset.labels.to(device)

        # --- build directories for logging ---
        self.LOG_PATH = self.setLogPath()
        self.addToPath_(SCHEDULE_LR=self.SCHEDULE_LR,
                        BATCH_SIZE=self.BATCH_SIZE,
                        LEARNING_RATE=self.LEARNING_RATE)
        makedir(self.LOG_PATH) # Create a directory for saving the results
        print("Writing log file: {}".format(self.LOG_PATH))
        self.saveConfig(config) # Write the current config to that directory
        self.writeFileHeaders() # Add column names to log files (e.g., ["Precision", "Recall"])

        # --- computing the number of trainable parameters ---
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}".format(params))

        # --- retrieve dataloaders ---
        loaders = self.getLoaders(self.dataset)
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]


        # --- resume training ---
        if self.resume:
            self.model = torch.load(self.LOG_PATH + "model.pt")

        # --- set optimizer ---
        self.optimizer = self.getOptimizer(self.model, self.optimizer_name)
        self.scheduler = None
        if self.SCHEDULE_LR:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                              gamma=self._GAMMA)

    def getSplitIndex(self, regen=False, save=False):
        """Choosing which examples are used
        for training, development, and testing

        If the indices already exist, load them. Otherwise recreate
        them - a numpy seed has been set already.
        """
        self.N = self.dataset.__len__()
        indices = range(self.N)
        split_points = [int(self.N*i) for i in self.split_props]
        train_ix = np.random.choice(indices,
                                    split_points[0],
                                    replace=False)
        val_ix = np.random.choice((list(set(indices) - set(train_ix))),
                                   split_points[1],
                                   replace=False)
        test_ix = list(set(indices) - set(train_ix) - set(val_ix))

        if regen: # Want new indices
            # --- save indices ---
            if save:
                np.save(self.dataset._load_path + "train_ix.npy", np.array(train_ix))
                np.save(self.dataset._load_path + "val_ix.npy", np.array(val_ix))
                np.save(self.dataset._load_path + "test_ix.npy", np.array(test_ix))
        else: # Want to load old indices
            try:
                train_ix = np.load(self.dataset._load_path + "train_ix.npy")
                val_ix = np.load(self.dataset._load_path + "val_ix.npy")
                test_ix = np.load(self.dataset._load_path + "test_ix.npy")
            except:
                if save:
                    # --- save indices ---
                    np.save(self.dataset._load_path + "train_ix.npy", np.array(train_ix))
                    np.save(self.dataset._load_path + "val_ix.npy", np.array(val_ix))
                    np.save(self.dataset._load_path + "test_ix.npy", np.array(test_ix))
        return train_ix, val_ix, test_ix

    def getBootstrapLoaders(self, dataset):
        loader_list = []
        for _ in range(10):
            ix = np.random.choice(self.test_ix, # test_ix was initialized in the original getLoaders() method
                                  int(0.5*len(self.test_ix)),
                                  replace=False)
            sampler = SubsetRandomSampler(ix)
            loader = data.DataLoader(dataset,
                                     batch_size=self.BATCH_SIZE,
                                     sampler=sampler,
                                     drop_last=True,
                                     num_workers=self.NUM_WORKERS)
            loader_list.append(loader)
        return loader_list

    def getLoaders(self, dataset):
        """define dataloaders"""
        try: # If indices exist in dataset, load them
            self.train_ix = dataset.train_ix
            self.val_ix = dataset.val_ix
            self.test_ix = dataset.test_ix
        except: # If not, get random split indices
            self.train_ix, self.val_ix, self.test_ix = self.getSplitIndex(regen=True)
        train_sampler = SubsetRandomSampler(self.train_ix)
        val_sampler = SubsetRandomSampler(self.val_ix)
        test_sampler = SubsetRandomSampler(self.test_ix)

        if self.dataset.NAME == "PhysioNet":
            collate_fn = self.VariableTimeCollateFunction
        else:
            collate_fn = self.passCollate

        train_loader = data.DataLoader(dataset,
                                       batch_size=self.BATCH_SIZE,
                                       sampler=train_sampler,
                                       drop_last=True,
                                       num_workers=self.NUM_WORKERS)
        val_loader = data.DataLoader(dataset,
                                     batch_size=self.BATCH_SIZE,
                                     sampler=val_sampler,
                                     drop_last=True,
                                     num_workers=self.NUM_WORKERS)
        test_loader = data.DataLoader(dataset,
                                      batch_size=self.BATCH_SIZE,
                                      sampler=test_sampler,
                                      drop_last=True,
                                      num_workers=self.NUM_WORKERS)
        return train_loader, val_loader, test_loader

    def passCollate(self, args):
        pass

    def variableTimeCollateFunction(self, batch, args, device=torch.device("cpu"),
                                    data_type="train", data_min=None, data_max=None):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = batch[0][2].shape[1]
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
        combined_tt = combined_tt.to(device)

        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
        combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
        
        combined_labels = None
        N_labels = 1

        combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
        combined_labels = combined_labels.to(device = device)
        
        for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
            tt = tt.to(device)
            vals = vals.to(device)
            mask = mask.to(device)
            if labels is not None:
                labels = labels.to(device)

            indices = inverse_indices[offset:offset + len(tt)]
            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask

            if labels is not None:
                combined_labels[b] = labels

        combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
            att_min = data_min, att_max = data_max)

        if torch.max(combined_tt) != 0.:
            combined_tt = combined_tt / torch.max(combined_tt)
            
        data_dict = {
            "data": combined_vals, 
            "time_steps": combined_tt,
            "mask": combined_mask,
            "labels": combined_labels}

        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
        return data_dict

    def getOptimizer(self, model, name):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        # --- get optimizer ---
        if name == "adam":
            optimizer = optim.Adam(trainable_params,
                                   lr=self.LEARNING_RATE,
                                   weight_decay=1e-5)
        elif name == "rmsprop":
            optimizer = optim.RMSprop(trainable_params,
                                      lr=self.LEARNING_RATE,
                                      weight_decay=1e-5)
        else:
            raise NotImplementedError

        return optimizer

    def addToPath_(self, **kwargs):
        """Add an argument to the logging path"""
        for key, val in kwargs.items():
            self.LOG_PATH += "{}-{}/".format(key, val)

    def setLogPath(self):
        """
        Using summaries of different elements of the
        pipeline to create names for logging files

        Returns
        -------
        path : string
            directory in which files for the currect experiment are stored
        """
        path = "./log/DATASET-{}/MODEL-{}".format(self.dataset.NAME,
                                                  self.model.NAME)
        path = attrToString(self.dataset, path)
        path = attrToString(self.model, path)
        path += "/"
        return path

    def saveConfig(self, config):
        if not os.path.exists(self.LOG_PATH + "config.txt"):
            try:
                with open(self.LOG_PATH + "config.txt", "a+") as file:
                    file.write(json.dumps(config))
            except:
                print("Tried to log config but the file exists already")

    def writeFileHeaders(self):
        if not os.path.exists(self.LOG_PATH+"train_results_{}.csv".format(self.iter)):
            row = ["Loss"]
            #row = ["Loss", "C Loss", "R Loss"]
            for metric in self.metrics: # Add metric names to csv headers
                row.append(metric.name)
            row.append("EpochTime")

            writeCSVRow(row, self.LOG_PATH + "train_results_{}".format(self.iter))
            writeCSVRow(row, self.LOG_PATH + "val_results_{}".format(self.iter))
            writeCSVRow(row, self.LOG_PATH + "test_results_{}".format(self.iter))

    def computeMetrics(self, predictions, labels):# , losses):
        results = []
        for metric in self.metrics:
            m = metric.compute(predictions.copy(), labels.copy())
            results.append(np.round(m, 3))
        return results

    def run(self):
        """Run the training and testing graphs in sequence."""
        for e in range(self.N_EPOCHS):
            # Train model
            start = time.time()
            self.model.train()
            self.runEpoch(model=self.model,
                          loader=self.train_loader,
                          mode="train",
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          test=False,
                          epoch=e)

            #print("Epoch {}/{} completed in {} minutes.".format(e+1, self.N_EPOCHS, np.round((end-start)/60., 3)))
        #    # Validate and Test model
        #    self.model.eval()
        #    self.runEpoch(self.model, self.val_loader, "val")

        # --- bootstrap ---
        self.model.eval()
        results = []
        if self.bootstrap:
            self.test_loader_list = self.getBootstrapLoaders(self.dataset)
            for i, loader in enumerate(self.test_loader_list):
                row = self.runEpoch(self.model, loader, "test", ix=i)
                results.append(row)
            accs = np.array(results)[:, 2]
            acc_mean = np.mean(accs)
            acc_std = np.std(accs)
            writeCSVRow([acc_mean, acc_std], self.LOG_PATH+f"cv_results_{self.iter}")

        else:
            self.runEpoch(self.model, self.test_loader, "test")

    def runEpoch(self, model, loader, mode, optimizer=None, scheduler=None, test=True, epoch=0, ix=None):
        """Given a data loader and model, run through the dataset once."""
        predictions = []
        labels = []
        total_loss = 0
        start = time.time()
        for i, (X, y) in enumerate(loader):
            logits = model(X, epoch=epoch, test=test)
            loss = model.computeLoss(logits, y.to(self.device))
            total_loss += loss.item()
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_hat = torch.softmax(logits, 1).cpu()
            [predictions.append(j) for j in y_hat.detach()]
            [labels.append(j) for j in y.cpu()]

        if scheduler:
            scheduler.step()

        end = time.time()
        total_loss = total_loss/len(loader)
        predictions = torch.stack(predictions).squeeze().detach().numpy()#.astype(np.int32)#.transpose(0, 1)
        labels = torch.stack(labels).squeeze().detach().numpy()

        # --- logging ---
        row = [total_loss]
        metrics = self.computeMetrics(predictions, labels)
        [row.append(metric) for metric in metrics]
        row.append(np.round((end-start)/60, 4))
        if ix:
            writeCSVRow(row, self.LOG_PATH+f"{mode}_results_{self.iter}_{ix}", round=True)
        else:
            writeCSVRow(row, self.LOG_PATH+f"{mode}_results_{self.iter}", round=True)
        return row
