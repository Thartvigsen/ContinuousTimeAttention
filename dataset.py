#from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
import os
from torch.utils import data
import numpy as np
import torch
import pandas as pd
from utils import *
import pandas as pd
import re
from readers import InHospitalMortalityReader, PhenotypingReader
import time
from scipy import interpolate
from torch.nn.utils.rnn import pad_sequence
#from pyts.approximation import MultipleCoefficientBinning

class Dataset(data.Dataset):
    """
    This class leans on the PyTorch Dataset class, giving access to easy
    batching and shuffling.

    Methods ending with underscores indicate in-place operations.

    Attributes starting with underscores will not have their values written
    into the log path.

    Example
    -------
    Let's say we want to run a new model on the MNIST dataset. First, we need
    to define a new class in this file called MNIST(). In this new class we
    follow the SimpleSignal() example, creating attribues self.data and
    self.labels along with defining the path to the directory in which we would
    like to store our files (e.g., /home/twhartvigsen/data/MNIST).
    """

    def __init__(self):
        self._data_path = "/home/twhartvigsen/data/" # Example: /home/twhartvigsen/data/

        # initialize data_setting dictionary to be passed to the model.
        self.data_setting = {}
        self.data_setting["MULTILABEL"] = False
        self.data_setting["VAR_LEN"] = False
        self.data_setting["N_FEATURES"] = 1 # Write over this in new dataset
        self.data_setting["N_CLASSES"] = 2 # Write over this in new dataset

    # --- DATA LOADING -------------------------------------------------------
    def loadData(self, path, regen=False):
        """Check if data exists, if so load it, else create new dataset."""
        if regen: # If forced regeneration
            data, labels = self.generate()
        else:
            try: # Try to load the files
                data = torch.load(path + "data.pt")
                labels = torch.load(path + "labels.pt")
            except: # If no files, generate new files
                print("No files found, creating new dataset")
                makedir(path) # make sure that there's a directory
                data, labels = self.generate()

        # --- save data for next time ---
        arrays = [data, labels]
        tensors = self.arraysToTensors(arrays, "FL")
        data, labels = tensors
        if len(data.shape) < 3:
            data = data.unsqueeze(2)
        outputs = [data, labels]
        names = ["data", "labels"]
        self.saveTensors(outputs, names, self._data_path)
        return data, labels

    # ------------------------------------------------------------------------
    def saveTensors(self, tensors, names, path):
        """Save a list of tensors as .pt files

        Parameters
        ----------
        tensors : list
            A list of pytorch tensors to save
        names : list
            A list of strings, each of which will be used to name
            the data saved with the same index
        """
        for data, name in zip(tensors, names):
            torch.save(data, path+"{}.pt".format(name))

    # ------------------------------------------------------------------------
    def __len__(self):
        """Compute the number of examples in a dataset

        This method is required for use of torch.utils.data.Dataset
        If this method does not apply to a new dataset, rewrite this
        method in the class.
        """
        return len(self.labels)

    # ------------------------------------------------------------------------
    def __getitem__(self, idx):
        """Extract an example from a dataset.

        This method is required for use of torch.utils.data.Dataset

        Parameters
        ----------
        idx : int
            Integer indexing the location of an example.

        Returns
        -------
        X : torch.FloatTensor()
            One example from the dataset.
        y : torch.LongTensor()
            The label associated with the selected example.
        """
        X = self.data[idx]
        y = self.labels[idx]
        return X, y#, idx

    # ------------------------------------------------------------------------
    def toCategorical(self, y, n_classes):
        """1-hot encode a tensor.

        Also known as comnverting a vector to a categorical matrix.

        Paremeters
        ----------
        y : torch.LongTensor()
            1-dimensional vector of integers to be one-hot encoded.
        n_classes : int
            The number of total categories.

        Returns
        -------
        categorical : np.array()
            A one-hot encoded matrix computed from vector y

        """
        categorical = np.eye(n_classes, dtype='uint8')[y]
        return categorical

class SimpleSignal(Dataset):
    def __init__(self, config, dist="uniform"):
        """
        This class provides access to one dataset (in this case called
        "SimpleSignal").

        Key Elements
        ------------
        self.data : torch.tensor (dtype=torch.float)
            Here be data of shape (instances x timesteps x variables)

        self.labels : torch.tensor
            Here be labels of shape (instances x nclasses). For regression,
            nclasses = 1.
        """
        self._n_examples = 500
        self.name = "SimpleSignal"
        # directory in which to store all files relevant to this dataset
        self._load_path = self._data_path + "{}/".format(self.name)
        utils.makedirs(self._load_path)
        super(SimpleSignal, self).__init__(config=config)
        self.data, self.labels = self.loadData(dist)
        self._N_CLASSES = len(np.unique(self.labels))
        self._N_FEATURES = 1
        self.seq_length = 10

        # Log attributes of the dataset to our data_setting dictionary
        self.data_setting["N_FEATURES"] = self._N_FEATURES
        self.data_setting["N_CLASSES"] = self._N_CLASSES

    def generate(self, dist="uniform", pos_signal=1, neg_signal=0):
        """
        The key method in your dataset. Define data and labels here. In this
        example, we create a synthetic dataset from scratch. More commonly,
        this method is used to load data from your dataset folder.
        """
        self.signal_locs = np.random.randint(self.seq_length,
                                             size=int(self._n_examples))
        X = np.zeros((self._n_examples,
                      self.seq_length,
                      self._N_FEATURES))
        y = np.zeros((self._n_examples))

        for i in range(int(self._n_examples)):
            if i < self._n_examples/2:
                X[i, self.signal_locs[i], 0] = pos_signal
                y[i] = 1
            else:
                X[i, self.signal_locs[i], 0] = neg_signal
        data = torch.tensor(np.asarray(X).astype(np.float32), dtype=torch.float)
        labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)
        return data, labels

class PhysioNet(Dataset):
    def __init__(self):
        self.NAME = "PhysioNet"
        super(PhysioNet, self).__init__()
        self._load_path = "/home/twhartvigsen/data/PhysioNet/processed/"
        makedir(self._load_path)
        self._N_FEATURES = 41
        self.ids, self.timesteps, self.values, self.masks, self.labels, self.lengths = self.loadData()
        #self.timesteps, self.values, self.masks, self.labels = self.loadData()
        self.timesteps = self.timesteps.unsqueeze(2).repeat(1, 1, self._N_FEATURES)
        #self.train_ix = torch.linspace(0, len(self.ids)/2., len(self.ids)/2)
        #self.train_ix = torch.linspace(0, len(self.ids)/2., len(self.ids)/2)
        #self.train_ix = torch.linspace(0, len(self.ids)/2., len(self.ids)/2)
        self.data = torch.stack([self.timesteps, self.values])
        self.data_setting["N_FEATURES"] = self.data.shape[-1]
        self.data_setting["N_CLASSES"] = len(torch.unique(self.labels))
    
    def __getitem__(self, ix):
        return self.data[:, ix, :, :], self.labels[ix]
    
    def __len__(self):
        return self.data.shape[1]

    def loadData(self):
        set_a = torch.load(self._load_path + "set-a_0.016.pt")
        set_b = torch.load(self._load_path + "set-b_0.016.pt")
        max_a = self.getMaxLength(set_a)
        max_b = self.getMaxLength(set_b)
        max_length = max(max_a, max_b)
        ids_a, timestamps_a, values_a, masks_a, labels_a, lengths_a = self.gatherData(set_a, max_length)
        ids_b, timestamps_b, values_b, masks_b, labels_b, lengths_b = self.gatherData(set_b, max_length)
        ids = torch.cat((ids_a, ids_b), 0)
        timestamps = torch.cat((timestamps_a, timestamps_b), 0)
        values = torch.cat((values_a, values_b), 0)
        masks = torch.cat((masks_a, masks_b), 0)
        labels = torch.cat((labels_a, labels_b), 0)
        lengths = torch.cat((lengths_a, lengths_b), 0)
        return ids, timestamps, values, masks, labels, lengths

    def getMaxLength(self, data):
        max_length = 0 
        for i, item in enumerate(data):
            if len(item[1]) > max_length:
                max_length = len(item[1])
        return max_length

    def gatherData(self, data, MAX_LENGTH):
        ids = torch.zeros(len(data))
        timestamps = torch.zeros((len(data), MAX_LENGTH))
        values = torch.zeros((len(data), MAX_LENGTH, self._N_FEATURES))
        masks = torch.zeros((len(data), MAX_LENGTH, self._N_FEATURES))
        labels = torch.zeros(len(data))
        lengths = torch.zeros(len(data))
        for i, item in enumerate(data):
            if item[4] is not None:
                ids[i] = torch.tensor(int(item[0]), dtype=torch.long)
                timestamps[i, :len(item[1])] = item[1]
                values[i, :len(item[2])] = item[2]
                masks[i, :len(item[3])] = item[3]
                labels[i] = item[4]
                lengths[i] = len(item[1])
        return ids, timestamps, values, masks, labels.long(), lengths

    def get_ix_splits(self):
        split_props = [0.8, 0.1, 0.1]
        indices = range(len(self.data))
        split_points = [int(len(self.data)*i) for i in split_props]
        train_ix = np.random.choice(indices,
                                    split_points[0],
                                    replace=False)
        test_ix = np.random.choice((list(set(indices)-set(train_ix))),
                                    split_points[1],
                                    replace=False)
        return train_ix, test_ix

class UWave(Dataset):
    def __init__(self):
        super(UWave, self).__init__()
        self.NAME = "UWave"
        self._load_path = self._data_path + "UWave/"
        self.timesteps, self.values, self.labels = self.loadData()
        self.data = torch.stack([self.timesteps, self.values])
        self.data_setting["N_FEATURES"] = self.data.shape[-1]
        self.data_setting["N_CLASSES"] = 8
    
    def __getitem__(self, ix):
        return self.data[:, ix, :, :], self.labels[ix]
    
    def __len__(self):
        return len(self.labels)

    def loadData(self):
        x_train = torch.tensor(np.load(self._load_path + "x_train.npy"), dtype=torch.float)
        y_train = torch.tensor(np.load(self._load_path + "y_train.npy"), dtype=torch.float)
        x_test = torch.tensor(np.load(self._load_path + "x_test.npy"), dtype=torch.float)
        y_test = torch.tensor(np.load(self._load_path + "y_test.npy"), dtype=torch.float)
        l_train = torch.tensor(np.load(self._load_path + "l_train.npy"), dtype=torch.long)
        l_test = torch.tensor(np.load(self._load_path + "l_test.npy"), dtype=torch.long)
        all_ix = np.arange(len(x_train) + len(x_test))
        np.random.shuffle(all_ix)
        self.train_ix = all_ix[:len(x_train)]
        self.test_ix = all_ix[len(x_train):]
        self.val_ix = all_ix[len(x_train):]
        timesteps = torch.cat((x_train, x_test), 0).unsqueeze(2)
        values = torch.cat((y_train, y_test), 0).unsqueeze(2)
        labels = torch.cat((l_train, l_test), 0)
        return timesteps, values, labels

class UWaveImpute(torch.utils.data.Dataset):
    def __init__(self, T=10):
        super(UWaveImpute, self).__init__()
        self.NAME = "UWave"
        timesteps, values, self.labels = self.loadData()
        new_values, masks, deltas = self.preprocess(timesteps, values, T)
        self.data = torch.stack([new_values, masks, deltas]).unsqueeze(3)
        self.data_setting["N_FEATURES"] = self.data.shape[-1]
        self.data_setting["N_CLASSES"] = 8
    
    def __getitem__(self, ix):
        return self.data[:, ix, :, :], self.labels[ix]
    
    def __len__(self):
        return len(self.labels)
    
    def getDelta(self, arr):
        deltas = [0]
        for i in range(1, len(arr)+1):
            if arr[i-1] == 1:
                deltas.append(deltas[i-1] + arr[i-1])
            else:
                deltas.append(0)
        return np.array(deltas[1:])

    def preprocess(self, timesteps, values, T):
        # For each t, what is the index of its nearest value?
        new_values = np.zeros((len(timesteps), T))
        masks = np.zeros_like(new_values)
        deltas = np.zeros_like(new_values)
        for i in range(len(timesteps)):
            # Then, add that point to that vector's mean
            t = timesteps[i].numpy()
            v = values[i].numpy()
            centroids = np.linspace(0, t[-1], T)
            means = torch.zeros(T)
            counts = torch.zeros(T)
            for x, y in zip(t, v):
                ix = np.abs(centroids - x).argmin() # index of closest
                means[ix] = means[ix] + y
                counts[ix] = counts[ix] + 1
            mask = (counts == 0).long()
            masks[i] = mask
            new_values[i] = means/counts
            deltas[i] = self.getDelta(mask)
        return torch.tensor(new_values), torch.tensor(masks), torch.tensor(deltas)
        
    def loadData(self):
        x_train = torch.tensor(np.load(self._data_path + "UWave/x_train.npy"), dtype=torch.float)
        y_train = torch.tensor(np.load(self._data_path + "UWave/y_train.npy"), dtype=torch.float)
        x_test = torch.tensor(np.load(self._data_path + "UWave/x_test.npy"), dtype=torch.float)
        y_test = torch.tensor(np.load(self._data_path + "UWave/y_test.npy"), dtype=torch.float)
        l_train = torch.tensor(np.load(self._data_path + "UWave/l_train.npy"), dtype=torch.long)
        l_test = torch.tensor(np.load(self._data_path + "UWave/l_test.npy"), dtype=torch.long)

        all_ix = np.arange(len(x_train) + len(x_test))
        np.random.shuffle(all_ix)
        self.train_ix = all_ix[:len(x_train)]
        self.test_ix = all_ix[len(x_train):]
        timesteps = torch.cat((x_train, x_test), 0).unsqueeze(2)
        values = torch.cat((y_train, y_test), 0).unsqueeze(2)
        labels = torch.cat((l_train, l_test), 0)
        return timesteps, values, labels

class InHospitalMortality(Dataset):
    def __init__(self):
        super(InHospitalMortality, self).__init__()
        start = time.time()
        self.NAME = "InHospitalMortality_interpolate3"
        self._load_path = self._data_path + "MIMIC/in-hospital-mortality/"
        self._train_path = self._load_path + "train/"
        self._test_path = self._load_path + "test/"
        #self.timesteps, self.values, self.labels = self.loadData()
        self._train_reader = InHospitalMortalityReader(self._train_path)
        self._test_reader = InHospitalMortalityReader(self._test_path)
        self._mean = False
        self.R = 10
        #self.data, self.labels = self.loadData()
        self.values, self.timesteps, self.intensities, self.masks, self.labels, self.lengths = self.loadData()
        print("Loaded data of shape {}".format(self.values.shape))
        self.data_setting["N_FEATURES"] = 17
        self.data_setting["N_CLASSES"] = 2
        end = time.time()
        print("preprocessing took {} minutes.".format(np.round((end-start)/60., 3)))
    
    def __getitem__(self, ix):
        return (self.timesteps[ix], self.values[ix], self.intensities[ix], self.masks[ix], self.lengths[ix]), self.labels[ix]
        #return self.data[ix], self.labels[ix]
    
    def __len__(self):
        return len(self.labels)

    def collectData(self, load_path, reader, train=True):
        indices = range(len(os.listdir(load_path))-1)
        #if train:
        #    indices = np.arange(0, 5000)
        #else:
        #    indices = np.arange(0, 1000)
        count = 0
        values = []
        y = []
        t = []
        m = []
        for i in indices:
            x = reader.read_example(i)
            X = x["X"]
            y.append(x["y"])
            X[X == ""] = np.nan
            X[X == "None"] = np.nan
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        X[i, j] = X[i, j].astype(np.float32)
                    except:
                        try:
                            v = np.array(re.findall(r'\d*\.?\d+', X[i, j]), dtype=np.float32).item()
                        except: # No numbers?
                            v = np.nan
                        X[i, j] = v
            X = X.astype(np.float32)
            t.append(X[:, 0])
            df = X[:, 1:]
            df = (df-np.nanmean(df, 0))/np.nanstd(df, 0)
            m.append(~np.isnan(df)) # Save the elements that are not missing!

            # fill in mean for missing values
            col_mean = np.nanmean(df, axis=0)
            col_mean[np.isnan(col_mean)] = 0.0
            inds = np.where(np.isnan(df))
            df[inds] = np.take(col_mean, inds[1])
            values.append(df)
        return values, t, m, y, len(indices)
    
    def pad(self, x):
        # X is a list, return a matrix of shape (len(x), max_len(items in x), -1)
        lens = [len(i) for i in x]
        maxlen = max(lens)
        if len(x[0].shape) > 1:
            out = np.zeros((len(x), maxlen, x[0].shape[-1]))
            for i in range(len(x)):
                out[i, :lens[i], :] = x[i]
            out = out[:, :150, :] # Take only first 150 timesteps
            #out = out[:, ::2, :] # Take every other value
        else:
            out = np.zeros((len(x), maxlen))
            for i in range(len(x)):
                out[i, :lens[i]] = x[i]
            out = out[:, :150]
            #out = out[:, ::2] # Take every other value
        return out

    def irregularMean(self, timesteps, values):
        # Output: N x R x V
        N = len(timesteps)
        V = values[0].shape[1]
        final_vals = np.zeros((N, self.R, V))
        final_timesteps = np.zeros((N, self.R))
        for n in range(N): # For each instance
            ref_steps = np.linspace(timesteps[n].min(), timesteps[n].max(), self.R+1)
            new_vals = np.zeros((self.R, V))
            for i in range(self.R): # For each reference timestep
                index = np.where(np.logical_and((timesteps[n] >= ref_steps[i]), (timesteps[n] < ref_steps[i+1])))
                vals = np.mean(values[n][index], 0)
                new_vals[i] = vals
                # But some will have no means!
            col_mean = np.nanmean(new_vals, axis=0)
            col_mean[np.isnan(col_mean)] = 0.0
            inds = np.where(np.isnan(new_vals))
            new_vals[inds] = np.take(col_mean, inds[1])
            final_vals[n] = new_vals
            final_timesteps[n] = ref_steps[:-1]
        return final_timesteps, final_vals

    def loadData(self, regen=False):
        if ((os.path.exists(self._load_path+"values.pt")) and not (regen)):
            #values = torch.load(self._load_path+"values_mean_{}.pt".format(self._mean))
            #masks = torch.load(self._load_path+"masks_mean_{}.pt".format(self._mean))
            #lengths = torch.load(self._load_path+"lengths_mean_{}.pt".format(self._mean))
            #timesteps = torch.load(self._load_path+"timesteps_mean_{}.pt".format(self._mean))
            #labels = torch.load(self._load_path+"labels_mean_{}.pt".format(self._mean))
            values = torch.load(self._load_path+"values_new.pt")
            #masks = torch.load(self._load_path+"masks_new.pt")
            masks = torch.load(self._load_path+"masks_mean_{}.pt".format(self._mean))
            lengths = torch.load(self._load_path+"lengths_mean_{}.pt".format(self._mean))
            timesteps = torch.load(self._load_path+"timesteps_new.pt")
            labels = torch.load(self._load_path+"labels_new.pt")
            intensities = torch.load(self._load_path+"intensities.pt")
            self.train_ix = np.load(self._load_path+"train_ix_mean_{}.npy".format(self._mean))
            self.val_ix = np.load(self._load_path+"val_ix_mean_{}.npy".format(self._mean))
            self.test_ix = np.load(self._load_path+"test_ix_mean_{}.npy".format(self._mean))
            print("Successfully loaded data")
        else: # Regenerate everything, then save it
            vals_train, time_train, mask_train, y_train, count_train = self.collectData(self._train_path, self._train_reader) 
            self.train_ix = np.random.choice(np.arange(count_train), int(count_train*0.8), replace=False)
            self.val_ix = list(set(np.arange(count_train)) - set(self.train_ix))

            # Test
            vals_test, time_test, mask_test, y_test, count_test = self.collectData(self._test_path, self._test_reader, train=False)
            self.test_ix = np.arange(count_test)+count_train

            values = vals_train + vals_test
            timesteps = time_train + time_test
            labels = y_train + y_test
            masks = mask_train + mask_test
            # Shape: N x T x V

            # If we want to take the mean every R steps, then here we need to
            # have some method for taking an evenly space sample from values
            # and timesteps
            if self._mean:
                timesteps, values = self.irregularMean(timesteps, values)

            # Choose if I want to downsample!

            # Pad data and collect into tensors
            lengths = torch.tensor([len(i) for i in values], dtype=torch.int)
            values = torch.tensor(self.pad(values), dtype=torch.float)
            masks = torch.tensor(self.pad(masks), dtype=torch.int)
            timesteps = torch.tensor(self.pad(timesteps), dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)

            # Save files
            torch.save(values, self._load_path+"values_mean_{}.pt".format(self._mean))
            torch.save(masks, self._load_path+"masks_mean_{}.pt".format(self._mean))
            torch.save(lengths, self._load_path+"lengths_mean_{}.pt".format(self._mean))
            torch.save(timesteps, self._load_path+"timesteps_mean_{}.pt".format(self._mean))
            torch.save(labels, self._load_path+"labels_mean_{}.pt".format(self._mean))
            np.save(self._load_path+"train_ix_mean_{}.npy".format(self._mean), self.train_ix)
            np.save(self._load_path+"val_ix_mean_{}.npy".format(self._mean), self.val_ix)
            np.save(self._load_path+"test_ix_mean_{}.npy".format(self._mean), self.test_ix)
        return values, timesteps, intensities, masks, labels, lengths

class MVSynth(Dataset):
    def __init__(self):
        super(MVSynth, self).__init__()
        self.NAME = "MVSynth"
        self.N = 500
        self.max_num_obs = 20
        self.H_max = 48
        self.nvariables = 2
        self._load_path = self._data_path + "MVSynth/"
        self.values, self.timesteps, self.labels, self.masks, self.lengths = self.loadData()
        #self.data = torch.stack([self.values, self.timesteps.unsqueeze(2).expand_as(self.masks), self.masks])
        self.data_setting["N_FEATURES"] = 2
        self.data_setting["N_CLASSES"] = 2
        
    def __getitem__(self, ix):
        #return self.data[ix], self.labels[ix]
        return (self.values[ix], self.timesteps[ix], self.masks[ix], self.lengths[ix]), self.labels[ix]
        
    def __len__(self):
        return len(self.labels)
    
    def loadData(self):
        labels = np.concatenate((np.zeros(self.N//2), np.ones(self.N//2)))
        timesteps = np.linspace(0, self.H_max, self.max_num_obs)[:, None].repeat(self.N, 1).T
        v1 = np.linspace(0, self.H_max, self.max_num_obs)[:, None].repeat(self.N, 1).T/self.H_max
        v2 = 1-np.linspace(0, self.H_max, self.max_num_obs)[:, None].repeat(self.N, 1).T/self.H_max
        values = np.transpose(np.array([v1, v2]), (1, 2, 0))
        masks = np.random.binomial(1, p=[0.4], size=values.shape)
        values = values*masks
        values[:self.N//2] = np.abs(1-values[:self.N//2]) # Flip classes
        lengths = np.random.randint(0, self.max_num_obs, self.N)

        for n in range(self.N):
            l = lengths[n]
            values[n, l:] = 0.0
            masks[n, l:] = 0.0
            
        values = torch.tensor(values, dtype=torch.float)
        timesteps = torch.tensor(timesteps, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.float)
        lengths = torch.tensor(lengths, dtype=torch.float)
        return values, timesteps, labels, masks, lengths

class PersonActivity(Dataset):
    def __init__(self):
        super(PersonActivity, self).__init__()
        self.NAME = "PersonActivity"
        self._load_path = self._data_path + "person_activity/"
        self.timesteps, self.values, self.intensities, self.labels = self.loadData()
        #self.values, self.labels = self.loadData()
        self.data_setting["N_FEATURES"] = 12
        self.data_setting["N_CLASSES"] = 7
    
    def __getitem__(self, ix):
        return (0, self.values[ix], self.intensities[ix], 0, 0), self.labels[ix]
    
    def __len__(self):
        return len(self.labels)

    def loadData(self):
        values = torch.load(self._load_path+"values.pt")
        timesteps = torch.load(self._load_path+"timesteps.pt")
        labels = torch.load(self._load_path+"labels.pt")
        intensities = torch.load(self._load_path+"intensities.pt")
        all_ix = np.arange(len(values))
        np.random.shuffle(all_ix)
        self.train_ix = all_ix[:int(len(values)*0.8)]
        self.val_ix = all_ix[int(len(values)*0.8):]
        self.test_ix = all_ix[int(len(values)*0.8):]
        return timesteps, values, intensities, labels

class IrregularTimeSeries(Dataset):
    def __init__(self):
        super(IrregularTimeSeries, self).__init__()

    def __getitem__(self, ix):
        return (0, self.values[ix], self.intensities[ix], 0, 0), self.labels[ix]
    
    def __len__(self):
        return len(self.labels)

    # --- Preparing ISTS data ------------------------------------------------
    #def prepISTS(self, timesteps, values, num_timesteps):
    #    interpolated = self.getInterpolated(timesteps, values, num_timesteps)
    #    masked = self.getMasked(timesteps, values)
    #    raw = self.padISTS(timesteps, values)
    #    return interpolated, masked, raw

    def getInterpolations(self, x, new_t):
        f = interpolate.interp1d(np.arange(len(x)), x, fill_value=(x[0], x[-1]), bounds_error=False)
        x = f(new_t)
        return x

    def makeIrregular(self, x, y):
        timesteps = []
        values = []
        seq_length = 500
        for i in range(len(x)):
            t_i = np.sort(np.random.uniform(0, len(x[i]), seq_length))
            if y[i] == 0:
                m_seq = get_markov_sequence(self._m_chain_neg, seq_length)
            else:
                m_seq = get_markov_sequence(self._m_chain_pos, seq_length)
            t_i = t_i[np.where(m_seq==1)]
            x_i = self.getInterpolations(x[0].squeeze(), t_i)
            timesteps.append(t_i)
            values.append(x_i)
        return timesteps, values

    def getIntensity(self, t, r, alpha=0.1):
        r = r.unsqueeze(1).repeat(1, t.shape[0]) # Add a column of all r values for each of t timesteps
        dist = torch.exp(torch.mul(-alpha, 1*torch.sub(r, t).pow(2)))    
        return dist.sum(1)

    def resample(self, t, v, R):
        new_r = torch.linspace(t.min(), t.max(), R)
        new_v = []
        lam = []
        f = interpolate.interp1d(t, v, fill_value=(v[0], v[-1]), bounds_error=False)
        new_v = f(new_r)
        lam = self.getIntensity(t, new_r)
        return new_r, torch.tensor(new_v), lam

    def irregularize(self, timesteps, values, R):
        V = []
        L = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            t, v, lam = self.resample(torch.tensor(t), torch.tensor(v), R)
            V.append(v)
            L.append(lam)
        V = torch.stack(V).unsqueeze(2)
        L = torch.stack(L).unsqueeze(2)
        return V, L

class MarkovIrregularUCR(IrregularTimeSeries):
    def __init__(self, name, R=200, p2n_pos=0.0, n2p_pos=1.0, p2n_neg=0.0, n2p_neg=1.0):
        super(MarkovIrregularUCR, self).__init__()
        self.p2n_pos = p2n_pos
        self.n2p_pos = n2p_pos
        self.p2n_neg = p2n_neg
        self.n2p_neg = n2p_neg
        self._m_chain_neg = instantiate_markov(self.p2n_pos, self.n2p_pos)
        self._m_chain_pos = instantiate_markov(self.p2n_neg, self.n2p_neg)
        self._ucr_name = name
        self.NAME = "UCR_{}_markov3".format(name)
        self._load_path = self._data_path + "UCR/"
        self.nsteps_interp = 500
        self.nsteps_impute = 500
        #self.values, self.labels, self.intensities = self.loadData(self.NAME)
        self._imputed, self._interpolated, self._raw, self.labels = self.loadData(self.NAME)

    def __getitem__(self, ix):
        # CAT: interpolated, IPN: raw, GRU-D/mask/delta: imputed
        return (self._imputed[ix], self._interpolated[ix], self._raw[ix]), self.labels[ix]

    def makeIrregular(self, x, y, num_instances):
        timesteps = []
        values = []
        seq_length = 500
        for i in range(len(x)):
            if y[i] == 0:
                m_seq = get_markov_sequence(self._m_chain_neg, seq_length)
            else:
                m_seq = get_markov_sequence(self._m_chain_pos, seq_length)
            t_i = np.sort(np.random.uniform(0, len(x[i]), seq_length))
            #t_i = t_i[np.where(m_seq==1)[0][:seq_length]]
            t_i = t_i[np.where(m_seq==1)[0]]
            #t_i = (t_i-t_i.min())/(t_i.max()-t_i.min())
            x_i = self.getInterpolations(x[i].squeeze(), t_i)
            timesteps.append(t_i)
            values.append(x_i)
        return timesteps, values

    def irregularize(self, timesteps, values, R):
        interpolated = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            t, v, lam = self.resample(torch.tensor(t), torch.tensor(v), R)
            v = v.unsqueeze(1)
            lam = torch.tensor(lam, dtype=torch.float).unsqueeze(1)
            interpolated.append((v, lam))
        return interpolated

    def valMaskPad(self, timesteps, values, nsteps):
        # Goal: Bin into nsteps evenly-spaced bins and then collect masks/delta
        imputed = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            bins = np.round(np.linspace(np.min(timesteps[i]), np.max(timesteps[i]), nsteps)[:, None], 3)
            t = np.array(t)[None, :]
            v = np.array(v)[None, :]
            buckets = np.abs((t-bins)).argmin(0)
            val_means = []
            timestep_means = []
            for n in range(nsteps):
                ix = np.where(buckets==n)
                val_means.append(np.nanmean(np.take(v, ix)))
                timestep_means.append(np.nanmean(np.take(t, ix)))
            val_means = np.array(val_means)
            timestep_means = np.array(timestep_means)
            masks = np.zeros_like(val_means)
            masks[np.isnan(val_means)] = 1
            deltas = []
            curr_val = 0
            for t in range(len(timestep_means)):
                if np.isnan(timestep_means[t]): # increase counter by the amount of time that has passed
                    curr_val += np.abs(np.float(bins[t]))
                else:
                    curr_val = 0
                deltas.append(curr_val)
            deltas = np.round(np.array(deltas), 3)
            val_means[np.isnan(val_means)] = np.nanmean(val_means)
            val_means = torch.tensor(val_means, dtype=torch.float)
            masks = torch.tensor(masks, dtype=torch.float)
            deltas = torch.tensor(deltas, dtype=torch.float)
            imputed.append((val_means.unsqueeze(1), masks.unsqueeze(1), deltas.unsqueeze(1)))
        return imputed

    def prepISTS(self, timesteps, values, nsteps_interp, nsteps_impute):
        # Assume timesteps/values are numpy arrays (same # steps per series)

        # Masked
        imputed = self.valMaskPad(timesteps, values, nsteps_impute)

        # Interpolated
        interpolated = self.irregularize(timesteps, values, nsteps_interp)
        #interpolated = (V, L)
        #interpolated = tuple([torch.tensor(i, dtype=torch.float) for i in interpolated])

        timesteps = [torch.tensor(i, dtype=torch.float) for i in timesteps]
        #lengths = []
        #for item in timesteps:
        #    lengths.append(len(item))
        #lengths = torch.tensor(np.array(lengths))
        timesteps = pad_sequence(timesteps).T.unsqueeze(2)
        values = [torch.tensor(i, dtype=torch.float) for i in values]
        values = pad_sequence(values).T.unsqueeze(2)
        raw = []
        for i in range(len(timesteps)):
            raw.append((timesteps[i], values[i]))#, lengths[i]))

        return imputed, interpolated, raw

    def loadData(self, name):
        x_train = np.loadtxt(self._load_path+"{}/{}_TRAIN.txt".format(self._ucr_name, self._ucr_name))
        x_test = np.loadtxt(self._load_path+"{}/{}_TEST.txt".format(self._ucr_name, self._ucr_name))
        x = np.concatenate((x_train, x_test), 0)

        # Separate labels and time series
        y = x[:, 0] - 1
        x = x[:, 1:]

        # Shuffle
        ix = np.random.choice(len(x), len(x), replace=False)
        x = x[ix]
        y = y[ix]

        # Make irregular
        t, v = self.makeIrregular(x, y, self.nsteps_interp)
        imputed, interpolated, raw = self.prepISTS(t, v, self.nsteps_interp, self.nsteps_impute)
        #V, L = self.irregularize(t, v, self.num_instances)
        y = torch.tensor(y, dtype=torch.long)
        return imputed, interpolated, raw, y
        #return V.float(), y, L.float()

class MultiModalIrregularUCR(IrregularTimeSeries):
    def __init__(self, name, R, nmode_pos=10, nmode_neg=10):
        super(MultiModalIrregularUCR, self).__init__()
        self._ucr_name = name
        self.NAME = "UCR_{}_multimodal_R2".format(name)
        self._load_path = self._data_path + "UCR/"
        self.R = R
        self._nmode_pos = R # Num distributions. Decrease to make signal come from patterns.
        self._nmode_neg = R # Num distributions. Decrease to make signal come from patterns.
        self._nsteps_interp = 500
        self._nsteps_impute = R
        self._imputed, self._interpolated, self._raw, self.labels = self.loadData(self.NAME)

    def __getitem__(self, ix):
        return (self._imputed[ix], self._interpolated[ix], self._raw[ix]), self.labels[ix]

    def getTimesteps(self, N, nmode):
        assert nmode <= N, "nmode must be <= to N"
        std = 1/(4*nmode)
        mus = torch.linspace(std, 1-std, nmode)
        n_per_mode = N//nmode
        timesteps = []
        for n in range(nmode):
            timesteps.append(np.random.normal(mus[n], std, n_per_mode).squeeze())
        timesteps = np.stack(timesteps).reshape(-1)
        if ((n_per_mode)*nmode) < N:
            extra_timesteps = np.random.uniform(0, 1, N-((n_per_mode)*nmode))
            timesteps = np.concatenate((timesteps, extra_timesteps), 0)
        return np.sort(np.clip(timesteps, 0, 1))

    def makeIrregular(self, x, y, num_instances):
        timesteps = []
        values = []
        seq_length = self.R
        for i in range(len(x)):
            #if y[i] == 0:
            #    t = self.getTimesteps(self.R, self.nmode_pos)
            #else:
            #    t = self.getTimesteps(self.R, self.nmode_neg)
            t = np.linspace(0, len(x[i]), self.R)
            x_i = self.getInterpolations(x[i].squeeze(), t)
            timesteps.append(t)
            values.append(x_i)
        return timesteps, values

    def irregularize(self, timesteps, values, R):
        interpolated = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            t, v, lam = self.resample(torch.tensor(t), torch.tensor(v), R)
            t = t.unsqueeze(1).float()
            v = v.unsqueeze(1).float()
            lam = lam.unsqueeze(1).float()
            interpolated.append((t, v, lam))
        return interpolated

    def valMaskPad(self, timesteps, values, nsteps):
        # Goal: Bin into nsteps evenly-spaced bins and then collect masks/delta
        imputed = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            bins = np.round(np.linspace(np.min(timesteps[i]), np.max(timesteps[i]), nsteps)[:, None], 3)
            t = np.array(t)[None, :]
            v = np.array(v)[None, :]
            buckets = np.abs((t-bins)).argmin(0)
            val_means = []
            timestep_means = []
            for n in range(nsteps):
                ix = np.where(buckets==n)
                val_means.append(np.nanmean(np.take(v, ix)))
                timestep_means.append(np.nanmean(np.take(t, ix)))
            val_means = np.array(val_means)
            timestep_means = np.array(timestep_means)
            masks = np.zeros_like(val_means)
            masks[np.isnan(val_means)] = 1
            deltas = []
            curr_val = 0
            for t in range(len(timestep_means)):
                if np.isnan(timestep_means[t]): # increase counter by the amount of time that has passed
                    curr_val += np.abs(np.float(bins[t]))
                else:
                    curr_val = 0
                deltas.append(curr_val)
            deltas = np.round(np.array(deltas), 3)
            val_means[np.isnan(val_means)] = np.nanmean(val_means)
            val_means = torch.tensor(val_means, dtype=torch.float)
            masks = torch.tensor(masks, dtype=torch.float)
            deltas = torch.tensor(deltas, dtype=torch.float)
            imputed.append((val_means.unsqueeze(1), masks.unsqueeze(1), deltas.unsqueeze(1)))
        return imputed

    def prepISTS(self, timesteps, values, nsteps_interp, nsteps_impute):
        # Assume timesteps/values are numpy arrays (same # steps per series)

        # Masked
        imputed = self.valMaskPad(timesteps, values, nsteps_impute)

        # Interpolated
        interpolated = self.irregularize(timesteps, values, nsteps_interp)

        # Raw
        timesteps = [torch.tensor(i, dtype=torch.float) for i in timesteps]
        timesteps = pad_sequence(timesteps).T.unsqueeze(2)
        values = [torch.tensor(i, dtype=torch.float) for i in values]
        values = pad_sequence(values).T.unsqueeze(2)
        raw = []
        for i in range(len(timesteps)):
            raw.append((timesteps[i], values[i]))

        return imputed, interpolated, raw

    def loadData(self, name):
        x_train = np.loadtxt(self._load_path+"{}/{}_TRAIN.txt".format(self._ucr_name, self._ucr_name))
        x_test = np.loadtxt(self._load_path+"{}/{}_TEST.txt".format(self._ucr_name, self._ucr_name))
        x = np.concatenate((x_train, x_test), 0)

        # Separate labels and time series
        y = x[:, 0] - 1
        x = x[:, 1:]

        # Shuffle
        ix = np.random.choice(len(x), len(x), replace=False)
        x = x[ix]
        y = y[ix]

        # Make irregular
        t, v = self.makeIrregular(x, y, self.R)
        imputed, interpolated, raw = self.prepISTS(t, v, self._nsteps_interp, self._nsteps_impute)
        #V, L = self.irregularize(t, v, self.num_instances)
        y = torch.tensor(y, dtype=torch.long)
        return imputed, interpolated, raw, y
        #return V.float(), y, L.float()

class HawkesIrregularUCR(IrregularTimeSeries):
    def __init__(self, name, num_timesteps=500, a_neg=0.01, a_pos=0.8):
        super(HawkesIrregularUCR, self).__init__()
        self._ucr_name = name
        self.NAME = "UCR_{}_hawkes".format(name)
        self.mu = 0.2
        self.a_pos = a_pos
        self.a_neg = a_neg
        self._load_path = self._data_path + "UCR/"
        self.num_timesteps = num_timesteps
        self.values, self.labels, self.intensities = self.loadData(self.NAME)

    def hawkes_intensity(self, mu, alpha, points, t):
        """Find the hawkes intensity:
        mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )
        """
        p = np.array(points)
        p = p[p <= t]
        p = np.exp(p - t) * alpha
        return mu + np.sum(p)

    def simulate_hawkes(self, mu, alpha, num_instances):
        t = 0
        points = []
        while len(points) < num_instances:
            m = self.hawkes_intensity(mu, alpha, points, t)
            s = np.random.exponential(scale = 1/m)
            ratio = hawkes_intensity(mu, alpha, points, t + s) / m
            if ratio >= np.random.uniform():
                points.append(t + s)
            t = t + s
        return np.array(points)

    def makeIrregular(self, x, y, num_instances):
        timesteps = []
        values = []
        for i in range(len(x)):
            if y[i] == 0:
                t_i = self.simulate_hawkes(self.mu, self.a_neg, num_instances)
            else:
                t_i = self.simulate_hawkes(self.mu, self.a_pos, num_instances)
            x_i = self.getInterpolations(x[0].squeeze(), t_i)
            timesteps.append(t_i)
            values.append(x_i)
        timesteps = np.stack(timesteps)
        timesteps = (timesteps-timesteps.min(1)[:, None])/(timesteps.max(1)-timesteps.min(1))[:, None]
        values = np.stack(values)
        return timesteps, values

    def loadData(self, name):
        x_train = np.loadtxt(self._load_path+"{}/{}_TRAIN.txt".format(self._ucr_name, self._ucr_name))
        x_test = np.loadtxt(self._load_path+"{}/{}_TEST.txt".format(self._ucr_name, self._ucr_name))
        x = np.concatenate((x_train, x_test), 0)

        # Separate labels and time series
        y = x[:, 0] - 1
        x = x[:, 1:]

        # Shuffle
        ix = np.random.choice(len(x), len(x), replace=False)
        x = x[ix]
        y = y[ix]

        # Make irregular
        t, v = self.makeIrregular(x, y, self.num_timesteps)
        V, L = self.irregularize(t, v, self.num_timesteps)
        t = torch.tensor(t, dtype=torch.float)
        V = torch.tensor(V, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        print(t.shape)
        print(V.shape)
        print(L.shape)
        print(y.shape)
        return V, y, L.float()

class Computers(IrregularTimeSeries):
    def __init__(self):
        super(Computers, self).__init__()
        self.NAME = "Computers"
        self._load_path = self._data_path + "UCR/Computers/"
        self.values, self.labels, self.intensities = self.loadData()
        #self.timesteps, self.values, self.labels, self.intensities = self.loadData()
        self.data_setting["N_FEATURES"] = 1
        self.data_setting["N_CLASSES"] = 2

    def loadData(self):
        values = torch.load(self._load_path+"values.pt")
        labels = torch.load(self._load_path+"labels.pt")
        intensities = torch.load(self._load_path+"intensities.pt")
        all_ix = np.arange(len(values))
        np.random.shuffle(all_ix)
        self.train_ix = all_ix[:int(len(values)*0.8)]
        self.val_ix = all_ix[int(len(values)*0.8):]
        self.test_ix = all_ix[int(len(values)*0.8):]
        return values.float(), labels.long(), intensities.float()

class ISTS(Dataset):
    def __init__(self, nref):
        super(ISTS, self).__init__()
        self.nref = nref
        #self.nsteps_interp = ninterp
        #self.nsteps_impute = nimpute

    def __getitem__(self, ix):
        return (self._imputed[ix], self._interpolated[ix], self._raw[ix]), self.labels[ix]

    def getIntensity(self, t, r, alpha=0.1):
        r = r.unsqueeze(1).repeat(1, t.shape[0]) # Add a column of all r values for each of t timesteps
        dist = torch.exp(torch.mul(-alpha, 1*torch.sub(r, t).pow(2)))    
        return dist.sum(1)

    def resample(self, t, v, R):
        new_r = torch.linspace(t.min(), t.max(), R)
        new_v = []
        lam = []
        f = interpolate.interp1d(t, v, fill_value=(v[0], v[-1]), bounds_error=False)
        new_v = f(new_r)
        lam = self.getIntensity(t, new_r)
        return new_r, torch.tensor(new_v), lam

    def irregularize(self, timesteps, values, nref):
        interpolated = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            t, v, lam = self.resample(torch.tensor(t), torch.tensor(v), nref)
            v = v.unsqueeze(1)
            lam = torch.tensor(lam, dtype=torch.float).unsqueeze(1)
            interpolated.append((v, lam))
        return interpolated

    def valMaskPad(self, timesteps, values, nsteps):
        # Goal: Bin into nsteps evenly-spaced bins and then collect masks/delta
        imputed = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            v = values[i]
            bins = np.round(np.linspace(np.min(timesteps[i]), np.max(timesteps[i]), nsteps)[:, None], 3)
            t = np.array(t)[None, :]
            v = np.array(v)[None, :]
            buckets = np.abs((t-bins)).argmin(0)
            val_means = []
            timestep_means = []
            for n in range(nsteps):
                ix = np.where(buckets==n)
                val_means.append(np.nanmean(np.take(v, ix)))
                timestep_means.append(np.nanmean(np.take(t, ix)))
            val_means = np.array(val_means)
            timestep_means = np.array(timestep_means)
            masks = np.zeros_like(val_means)
            masks[np.isnan(val_means)] = 1
            deltas = []
            curr_val = 0
            for t in range(len(timestep_means)):
                if np.isnan(timestep_means[t]): # increase counter by the amount of time that has passed
                    curr_val += np.abs(np.float(bins[t]))
                else:
                    curr_val = 0
                deltas.append(curr_val)
            deltas = np.round(np.array(deltas), 3)
            val_means[np.isnan(val_means)] = np.nanmean(val_means)
            val_means = torch.tensor(val_means, dtype=torch.float)
            masks = torch.tensor(masks, dtype=torch.float)
            deltas = torch.tensor(deltas, dtype=torch.float)
            imputed.append((val_means.unsqueeze(1), masks.unsqueeze(1), deltas.unsqueeze(1)))
        return imputed

    def prepISTS(self, timesteps, values, nref):
        # Assume timesteps/values are numpy arrays (same # steps per series)

        # Masked
        imputed = self.valMaskPad(timesteps, values, nref)

        # Interpolated
        interpolated = self.irregularize(timesteps, values, nref=500) # For CAT
        #interpolated = (V, L)
        #interpolated = tuple([torch.tensor(i, dtype=torch.float) for i in interpolated])

        timesteps = [torch.tensor(i, dtype=torch.float) for i in timesteps]
        #lengths = []
        #for item in timesteps:
        #    lengths.append(len(item))
        #lengths = torch.tensor(np.array(lengths))
        timesteps = pad_sequence(timesteps).T.unsqueeze(2)
        values = [torch.tensor(i, dtype=torch.float) for i in values]
        values = pad_sequence(values).T.unsqueeze(2)
        raw = []
        for i in range(len(timesteps)):
            raw.append((timesteps[i], values[i]))#, lengths[i]))

        return imputed, interpolated, raw

class SeqLength(ISTS):
    def __init__(self, T, N, nref):
        #self.NAME = "SyntheticSeqLength"
        super(SeqLength, self).__init__(nref)
        self._load_path = self._data_path + "SeqLength/{}/".format(self.NAME)
        self.T = T
        self.N = N
        self.signal_length = .2
        self.nsamples_on_signal = 20
        self.nsamples_off_signal = T-self.nsamples_on_signal
        self.signal_start = np.ones(self.N)*0.4
        self.signal_end = self.signal_start + self.signal_length
        self._imputed, self._interpolated, self._raw, self.labels = self.loadData(self.NAME)

    def dome(self, t):
        """A dome defined in [0, 1]"""
        return .5*np.sin(np.pi*t/self.signal_length)

    def spike(self, t):
        """A spike defined in [0, 1] with a peak of 1.0"""
        if t < .5:
            return 1.0*t
        else:
            return -1.0*t + 1.0

    def createSignal(self, timesteps, signal_start, dome=True):
        """Piecewise function - sampling points from signals at different locations in the time series."""
        values = np.zeros(len(timesteps))
        for i in range(len(timesteps)):
            if timesteps[i] < signal_start: # Left of signal
                values[i] = np.random.normal(0, 0.01)
            elif ((timesteps[i] >= signal_start) & (timesteps[i] < (signal_start+self.signal_length))): # On signal
                if dome:
                    values[i] = self.dome(timesteps[i]-signal_start)
                else:
                    values[i] = self.spike((timesteps[i]-signal_start)/self.signal_length)
                values[i] += np.random.normal(0, .0, 1)
            else: # Right of signal
                values[i] = np.random.normal(0, 0.01)
        return values
    
    @abstractmethod
    def getTimesteps(self, signal_start, signal_end):
        #timesteps = np.random.uniform(signal_start, signal_end, self.nsamples_on_signal)
        #nsamples_off_signal = self.T - self.nsamples_on_signal
        #n_from_left = np.random.choice(self.T - self.nsamples_on_signal, 1).astype(np.int32)
        #n_from_right = (self.T - self.nsamples_on_signal) - n_from_left
        #left_samples = np.random.uniform(0, signal_start, (n_from_left))#[:, None]
        #right_samples = np.random.uniform(signal_end, 1.0, (n_from_right))#[:, None]
        #timesteps = np.concatenate((timesteps, left_samples, right_samples), 0)
        #timesteps = np.sort(timesteps)    
        return timesteps
    
    def getTimestepsValuesLabels(self):
        timesteps = np.empty((self.N, self.T))
        values = np.empty((self.N, self.T))
        labels = np.empty((self.N))
        for i in range(self.N):
            t = self.getTimesteps(self.signal_start[i], self.signal_end[i])
            if i <= int(self.N/2): # Dome
                values[i, :] = self.createSignal(t, self.signal_start[i], dome=True)
                labels[i] = 0
            else: # Table
                values[i, :] = self.createSignal(t, self.signal_start[i], dome=False)
                labels[i] = 1
            timesteps[i, :] = t
        return timesteps, values, labels
        #return torch.tensor(timesteps, dtype=torch.float).unsqueeze(2), torch.tensor(values, dtype=torch.float).unsqueeze(2), torch.tensor(labels, dtype=torch.long)

    def loadData(self, name):
        t, v, y = self.getTimestepsValuesLabels()

        # Shuffle
        ix = np.random.choice(self.N, self.N, replace=False)
        t = t[ix]
        v = v[ix]
        y = y[ix]

        # Make irregular
        imputed, interpolated, raw = self.prepISTS(t, v, self.nref)
        y = torch.tensor(y, dtype=torch.long)
        return imputed, interpolated, raw, y

class SeqLengthUniform(SeqLength):
    def __init__(self, T=50, N=500, nref=50):
        self.NAME = "SyntheticSeqLengthUniform"
        super(SeqLengthUniform, self).__init__(T=T, N=N, nref=nref)

    # First, I want to get timesteps THIS way
    def getTimesteps(self, signal_start, signal_end):
        timesteps = np.random.uniform(signal_start, signal_end, self.nsamples_on_signal)
        nsamples_off_signal = self.T - self.nsamples_on_signal
        n_from_left = np.random.choice(self.T - self.nsamples_on_signal, 1).astype(np.int32)
        n_from_right = (self.T - self.nsamples_on_signal) - n_from_left
        left_samples = np.random.uniform(0, signal_start, (n_from_left))#[:, None]
        right_samples = np.random.uniform(signal_end, 1.0, (n_from_right))#[:, None]
        timesteps = np.concatenate((timesteps, left_samples, right_samples), 0)
        timesteps = np.sort(timesteps)    
        return timesteps
