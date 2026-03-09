import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from src.utils.xuq_utils import numpy_to_torch
from datasets import Dataset
from sklearn.model_selection import KFold


class MNISTDataset(Dataset):
    name = "MNIST Dataset"

    def check_if_dataset_exists(self, dataset_path):
        data_root = os.getcwd()
        for data_object in [
            "df_train_input",
            "df_test_input",
            "df_train_output",
            "df_test_output",
        ]:
            object_exists = os.path.exists(
                os.path.join(data_root, dataset_path + "/" + data_object)
            )
            if not object_exists:
                return False
        return True

    def create_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_data = datasets.MNIST(
            root="./datasets", train=True, transform=transform, download=False
        )
        test_data = datasets.MNIST(
            root="./datasets", train=False, transform=transform, download=False
        )
        return train_data, test_data

    def serve_dataset(self, val_split: float = 0.1, val_set_required: bool = True) -> tuple:
        """serve dataset function

        Args:
            val_split (float, optional): validation split. Defaults to 0.1.
            val_set_required (bool, optional): validation set required flag. Defaults to True.

        Returns:
            tuple: tuple of data splits (X_train, X_test, X_val, y_train, y_test, y_val)
        """
        if val_set_required:
            assert (val_split is not None) and (
                0 < val_split < 1
            ), "val_split must be between 0 and 1"
        train_data, test_data = self.create_data()

        X_train, y_train = train_data.data.float().unsqueeze(1), train_data.targets
        X_test, y_test = test_data.data.float().unsqueeze(1), test_data.targets

        if val_set_required:
            n_train = int(X_train.shape[0] * (1 - val_split))
            X_train, X_val = X_train[:n_train], X_train[n_train:]
            y_train, y_val = y_train[:n_train], y_train[n_train:]
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            return X_train, X_test, torch.tensor([]), y_train, y_test, torch.tensor([])

    def serve_dataset_as_folds(
        self,
        k_folds: int,
        val_split: float = 0.1,
        val_set_required: bool = True,
        random_state: int = 42,
    ) -> dict:
        """serves the dataset as k folds for cross-validation, each fold containing a train set and a test set

        Args:
            k_folds (int): Number of folds for cross-validation
            val_split (float, optional): Percentage of training data to use for validation. Defaults to 0.1.
            val_set_required (bool, optional): Whether a validation set is required. Defaults to True.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.

        Returns:
            dict: A dictionary containing the folds with train and test sets {'fold_1': {'X_train': Tensor, 'X_val': Tensor|None, 'X_test': Tensor, 'y_train': Tensor,
                                'y_val': Tensor|None, 'y_test': Tensor}, ...}
        """
        if val_set_required:
            assert (val_split is not None) and (
                0 < val_split < 1
            ), "val_split must be between 0 and 1"
        train_data, test_data = self.create_data()
        X_train, y_train = train_data.data.float(), train_data.targets
        X_test, y_test = test_data.data.float(), test_data.targets

        full_feature_set = torch.cat([X_train, X_test], dim=0).numpy()
        full_target_set = np.concatenate(
            [np.asarray(y_train).reshape(-1), np.asarray(y_test).reshape(-1)]
        )

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        folds_dict = {}
        fold_number = 1

        for train_index, test_index in kf.split(full_feature_set):
            X_train_fold, X_test_fold = numpy_to_torch(
                (full_feature_set[train_index], full_feature_set[test_index])
            )
            y_train_fold, y_test_fold = numpy_to_torch(
                (full_target_set[train_index], full_target_set[test_index])
            )
            # ensure labels are integer class indices (LongTensor) for loss functions
            y_train_fold = y_train_fold.long()
            y_test_fold = y_test_fold.long()

            n_train = int(X_train_fold.shape[0] * (1 - val_split))
            X_train_fold, X_val_fold = X_train_fold[:n_train], X_train_fold[n_train:]
            y_train_fold, y_val_fold = y_train_fold[:n_train], y_train_fold[n_train:]

            folds_dict[f"fold_{fold_number}"] = {
                "X_train": X_train_fold.unsqueeze(1),
                "X_val": X_val_fold.unsqueeze(1) if val_set_required else None,
                "X_test": X_test_fold.unsqueeze(1),
                "y_train": y_train_fold,
                "y_val": y_val_fold if val_set_required else None,
                "y_test": y_test_fold,
            }
            fold_number += 1

        return folds_dict

    def serve_dataset_as_folds_for_calibration(
        self, val_split: float = 0.1, k_folds: int = 5
    ) -> dict:
        """preparing dataset for mc calibration

        Args:
            val_split (float, optional): validation split. Defaults to 0.1.
            k_folds (int, optional): number of fold splits. Defaults to 5.

        Returns:
            dict : A dictionary containing the folds with train and test sets {'fold_1': {'X_train': Tensor, 'X_val': Tensor|None, 'X_test': Tensor, 'y_train': Tensor,
                                'y_val': Tensor|None, 'y_test': Tensor}, ...}
        """
        train_data, test_data = self.create_data()
        X_train, y_train = train_data.data.float().unsqueeze(1), train_data.targets
        X_test, y_test = test_data.data.float().unsqueeze(1), test_data.targets

        # Initialize KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        folds_dict = {}
        fold_number = 1

        for train_index, test_index in kf.split(X_train.numpy()):
            X_train_fold, X_val_fold = (X_train[train_index], X_train[test_index])
            y_train_fold, y_val_fold = (y_train[train_index], y_train[test_index])

            n_train = int(X_train_fold.shape[0] * (1 - val_split))
            X_train_fold, X_val_fold = X_train_fold[:n_train], X_train_fold[n_train:]
            y_train_fold, y_val_fold = y_train_fold[:n_train], y_train_fold[n_train:]
            folds_dict[f"fold_{fold_number}"] = {
                "X_train": X_train_fold,
                "X_val": X_val_fold,
                "y_train": y_train_fold,
                "y_val": y_val_fold,
                "X_test": X_test,
                "y_test": y_test,
            }
            fold_number += 1
        return folds_dict

    def serve_dataset_as_dataloader(self, val_split: float = 0.1, batch_size: int = 32):
        """load nist as dataloader

        Args:
            val_split (float, optional): validation split. Defaults to 0.1.
            batch_size (int, optional): dataloader batch size. Defaults to 32.

        Returns:
            tuple: tuple of train test and val loaders (train_loader, val_loader, test_loader)
        """
        X_train, X_test, X_val, y_train, y_test, y_val = self.serve_dataset(
            val_split, val_set_required=True
        )

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
