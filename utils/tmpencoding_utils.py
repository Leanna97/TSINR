import pickle
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

prefix = "processed"


def save_z(z, filename="z"):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + "_" + str(i) + ".txt", "w") as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write("%f " % (z[j][i][k]))
                file.write("\n")
    i = z.shape[1] - 1
    with open(filename + "_" + str(i) + ".txt", "w") as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write("%f " % (z[j][i][k]))
            file.write("\n")


def get_data_dim(dataset):
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_data(
    prefix,
    dataset,
    max_train_size=None,
    max_test_size=None,
    print_log=True,
    do_preprocess=False,
    train_start=0,
    test_start=0,
):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    # print("load data of:", dataset)
    # print("train: ", train_start, train_end)
    # print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    # print(dataset)
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    # print("train set shape: ", train_data.shape)
    # print("test set shape: ", test_data.shape)
    # print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """returns normalized and standardized data."""

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be a 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains null values. Will be replaced with 0")
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    # print("Data normalized")

    return df


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        #             self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #             self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model=None):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        #         torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss