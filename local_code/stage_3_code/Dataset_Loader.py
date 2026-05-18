from local_code.base_class.dataset import dataset
import numpy as np
import os
import pickle


class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None, folder_path='', file_name=''):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = folder_path
        self.dataset_source_file_name = file_name

    def load(self):
        print(f'loading data from {self.dataset_source_file_name}...')

        file_path = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        )

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        X_train, y_train = self.process_split(data['train'])
        X_test, y_test = self.process_split(data['test'])

        DEBUG = False

        if DEBUG:
            X_train = X_train[:500]
            y_train = y_train[:500]
            X_test = X_test[:100]
            y_test = y_test[:100]

        return {
            'train': {
                'X': X_train,
                'y': y_train
            },
            'test': {
                'X': X_test,
                'y': y_test
            }
        }

    def process_split(self, split_data):
        X = []
        y = []

        dataset_name = self.dataset_source_file_name.upper()

        for instance in split_data:
            img = np.array(instance['image'], dtype=np.float32)
            label = instance['label']

            #ORL labels are 1-40, so convert to 0-39 for PyTorch
            if 'ORL' in dataset_name:

                #(height, width, channels)
                #ORL is grayscale but stored as 3 identical RGB channels (R = G = B)
                #Use only one channel: (112, 92, 3) -> (112, 92)
                if img.ndim == 3:
                    img = img[:, :, 0]

            #Normalize pixel values from 0-255 to 0-1
            img = img / 255.0

            #MNIST/ORL grayscale: (H, W) -> (1, H, W)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)

            #CIFAR color: (H, W, C) -> (C, H, W)
            elif img.ndim == 3:
                img = img.transpose(2, 0, 1)

            X.append(img)
            y.append(label)

        X = np.stack(X)
        y = np.array(y, dtype=np.int64)

        print("Before fix label min:", y.min(), "label max:", y.max())

        if y.min() == 1:
            y = y - 1

        print("After fix label min:", y.min(), "label max:", y.max())

        return X, y