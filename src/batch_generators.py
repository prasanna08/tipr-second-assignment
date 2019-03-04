import os
from sklearn.preprocessing import OneHotEncoder
import skimage
import numpy as np
import reader

class BatchGeneratorDisk(object):
    def __init__(self, path_to_dataset, batch_size, grayscale=True):
        self.path = path_to_dataset
        self.batch_size = batch_size
        self.setup_data(path_to_dataset)
        self.encoder = OneHotEncoder(categories='auto')
        self.encoder.fit(self.labels)
        self.grayscale = grayscale
        self.splitted = False
        self._cursor = 0
        self._reader = skimage.io.imread_collection_wrapper(self._imread_wrapper)
    
    def setup_data(self, path_to_dataset):
        labels = os.listdir(path_to_dataset)
        instances = []
        for label in labels:
            class_path = os.path.join(path_to_dataset, label)
            class_instances = [os.path.join(class_path, i) for i in os.listdir(class_path)]
            instances.extend(class_instances)
        self.data = instances
        self.labels = np.array(labels).reshape(-1, 1)
    
    def shuffle_data(self):
        np.random.shuffle(self.data)
    
    def split(self):
        self.shuffle_data()
        self.splitted = True
        instances = len(self.data)
        i = int(0.6*instances)
        j = int(0.8*instances)
        self.train_data = self.data[:i]
        self.valid_data = self.data[i:j]
        self.test_data = self.data[j:]
    
    def _imread_wrapper(self, path):
        return skimage.io.imread(path, as_gray=self.grayscale)
    
    def _read_from_disk(self, paths):
        x = self._reader(paths)
        return np.array(x).astype(np.uint8)
    
    def _get_labels_from_paths(self, paths):
        y = np.array([path.split('/')[-2] for path in paths]).reshape(-1, 1)
        return np.squeeze(np.asarray(self.encoder.transform(y).todense())).astype(np.uint8)
    
    def get_validation_data(self):
        if self.splitted:
            return self._read_from_disk(self.valid_data), self._get_labels_from_paths(self.valid_data)
        else:
            return None
    
    def get_test_data(self):
        if self.splitted:
            return self._read_from_disk(self.test_data), self._get_labels_from_paths(self.test_data)
        else:
            return None
    
    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self):
        X = self.data if not self.splitted else self.train_data
        epoch = False
        if self._cursor + self.batch_size >= len(X):
            rem = self._cursor + self.batch_size - len(X)
            paths = X[self._cursor:] + X[:rem]
            self._cursor = rem
            epoch = True
        else:
            paths = X[self._cursor: self._cursor+self.batch_size]
            _pcursor = self._cursor
            self._cursor = (self._cursor + self.batch_size) % (len(X))
            epoch = True if _pcursor >= self._cursor else False

        batch_output = self._get_labels_from_paths(paths)
        batch_input = self._read_from_disk(paths)
        return reader.rescale(batch_input.reshape(self.batch_size, -1)), batch_output, epoch

class BatchGenerator(object):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.splitted = False
        self._cursor = 0
    
    def shuffle_data(self):
        idcs = np.arange(self.data.shape[0])
        np.random.shuffle(idcs)
        self.data = self.data[idcs]
        self.labels = self.labels[idcs]
    
    def split(self):
        self.shuffle_data()
        self.splitted = True
        instances = self.data.shape[0]
        i = int(0.6*instances)
        j = int(0.8*instances)
        self.train_data = self.data[:i]
        self.train_labels = self.labels[:i]
        self.valid_data = self.data[i:j]
        self.valid_labels = self.labels[i:j]
        self.test_data = self.data[j:]
        self.test_labels = self.labels[j:]
    
    def get_validation_data(self):
        if self.splitted:
            return self.valid_data, self.valid_labels
        else:
            return None
    
    def get_test_data(self):
        if self.splitted:
            return self.test_data, self.test_labels
        else:
            return None
    
    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self):
        X = self.data if not self.splitted else self.train_data
        Y = self.labels if not self.splitted else self.train_labels

        if self._cursor + self.batch_size > X.shape[0]:
            rem = self._cursor + self.batch_size - X.shape[0]
            batch_input = np.concatenate([X[self._cursor:], X[:rem]], axis=0)
            batch_output = np.concatenate([Y[self._cursor:], Y[:rem]], axis=0)
            self._cursor = rem
            return batch_input, batch_output, True

        batch_input = X[self._cursor: self._cursor+self.batch_size]
        batch_output = Y[self._cursor: self._cursor+self.batch_size]
        _pcursor = self._cursor
        self._cursor = (self._cursor + self.batch_size) % (X.shape[0])

        epoch = False
        if _pcursor >= self._cursor:
            epoch = True
        return batch_input, batch_output, epoch
