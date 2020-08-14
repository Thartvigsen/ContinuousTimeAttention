#from torch.utils.data import Dataset
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
        self.NAME = "InHospitalMortality"
        self._load_path = self._data_path + "MIMIC/in-hospital-mortality/"
        self._train_path = self._load_path + "train/"
        self._test_path = self._load_path + "test/"
        #self.timesteps, self.values, self.labels = self.loadData()
        self.train_reader = InHospitalMortalityReader(self._train_path, period_length=48.0)
        self.test_reader = InHospitalMortalityReader(self._test_path, period_length=48.0)
        self.data, self.labels = self.loadData()
        self.data_setting["N_FEATURES"] = 17
        self.data_setting["N_CLASSES"] = 2
        end = time.time()
        print("preprocessing took {} minutes.".format(np.round((end-start)/60., 3)))
    
    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]
    
    def __len__(self):
        return len(self.labels)

    def collectData(self, load_path, reader):
        #indices = self.getFilenames(listfile_path)
        indices = range(len(os.listdir(load_path))-1)
        count = 0
        X = []
        y = []
        for i in indices:
            x = reader.read_example(i)
            try:
                df = np.array(x).astype(np.float32)
            except:
                df = pd.DataFrame(x["X"])
                for name in df.columns:
                    if df[name].dtype != np.number:
                        df[name] = df[name].str.extract('(\d+)')
                df = np.array(df).astype(np.float32)
            X.append(df)
            y.append(x["y"])
        #X = np.stack(X).squeeze()
        #y = np.stack(y).squeeze()
        return X, y, len(indices)

    def loadData(self):
        # Load all train files, load all test files, then concatenate and index
        # Train
        if os.path.exists(self._load_path+"data.pt"):
            X = torch.load(self._load_path+"data.pt")
            y = torch.load(self._load_path+"labels.pt")
            self.train_ix = np.load(self._load_path+"train_ix.npy")
            self.val_ix = np.load(self._load_path+"val_ix.npy")
            self.test_ix = np.load(self._load_path+"test_ix.npy")

        else: # Regenerate everything, then save it
            X_train, y_train, count_train = self.collectData(self._train_path, self.train_reader) 
            self.train_ix = np.random.choice(np.arange(count_train), int(count_train*0.8), replace=False)
            self.val_ix = list(set(np.arange(count_train)) - set(self.train_ix))

            # Test
            X_test, y_test, count_test = self.collectData(self._test_path, self.test_reader)
            self.test_ix = np.arange(count_test)+count_train

            X = X_train + X_test
            y = y_train + y_test

            # Save files
            torch.save(X, self._load_path+"data.pt")
            torch.save(y, self._load_path+"labels.pt")
            np.save(self._load_path+"train_ix.npy", self.train_ix)
            np.save(self._load_path+"val_ix.npy", self.val_ix)
            np.save(self._load_path+"test_ix.npy", self.test_ix)
        return X, y
