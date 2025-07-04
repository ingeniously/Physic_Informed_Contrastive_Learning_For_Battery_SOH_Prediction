import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class SimpleLoader:
    def __init__(self, csv_path, batch_size=256, normalization=True, normalization_method='z-score'):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.normalization = normalization
        self.normalization_method = normalization_method

    def _3_sigma(self, ser):
        rule = (ser.mean() - 3 * ser.std() > ser) | (ser.mean() + 3 * ser.std() < ser)
        index = np.arange(ser.shape[0])[rule]
        return index

    def delete_3_sigma(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_csv(self):
        df = pd.read_csv(self.csv_path)
        # Remove 3-sigma outliers
        df = self.delete_3_sigma(df)
        # Sanity check: print info about time column before normalization
       #  print("Before normalization, time min:", df.iloc[:, -2].min(), "max:", df.iloc[:, -2].max(), "unique:", df.iloc[:, -2].unique()[:5])
        # Normalize features except SoH (last column)
        f_df = df.iloc[:, :-1]
        if self.normalization_method == 'min-max':
            f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
        elif self.normalization_method == 'z-score':
            f_df = (f_df - f_df.mean()) / f_df.std()
        df.iloc[:, :-1] = f_df
        # After normalization, check that time feature is not constant
        # print("After normalization, time min:", df.iloc[:, -2].min(), "max:", df.iloc[:, -2].max(), "unique:", df.iloc[:, -2].unique()[:5])
        # If time is constant, raise error!
        if np.allclose(df.iloc[:, -2].std(), 0):
            raise ValueError("The 'Time_norm' feature (second to last column) is constant after normalization! Check your data or normalization method.")
        return df

    def load(self):
        df = self.read_csv()
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]

        tensor_X1 = torch.from_numpy(x1).float()
        tensor_X2 = torch.from_numpy(x2).float()
        tensor_Y1 = torch.from_numpy(y1).float().view(-1, 1)
        tensor_Y2 = torch.from_numpy(y2).float().view(-1, 1)

        # 80% train, 20% test, 20% of train as valid
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
        from sklearn.model_selection import train_test_split
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.batch_size,
                                 shuffle=False)
        all_loader = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                 batch_size=self.batch_size,
                                 shuffle=False)

        return {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader,
            'all': all_loader
        }

if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
        return parser.parse_args()

    args = get_args()

    loader = SimpleLoader(
        csv_path=args.csv_file,
        batch_size=args.batch_size,
        normalization=True,
        normalization_method=args.normalization_method
    ).load()

    print('train_loader:', len(loader['train']),
          'test_loader:', len(loader['test']),
          'valid_loader:', len(loader['valid']),
          'all_loader:', len(loader['all']))

    for iter, (x1, x2, y1, y2) in enumerate(loader['train']):
        print('x1 shape:', x1.shape)
        print('x2 shape:', x2.shape)
        print('y1 shape:', y1.shape)
        print('y2 shape:', y2.shape)
        print('y1 max:', y1.max())
        # Print first 5 times for sanity check
        print('Time_norm x1[:5]:', x1[:5, -1])
        print('Time_norm x2[:5]:', x2[:5, -1])
        break