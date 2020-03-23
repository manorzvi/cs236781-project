from typing import NamedTuple, List
import pickle

class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss.
    """
    loss: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch (train or test).
    """
    losses: List[float]


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch.
    """
    num_epochs: int
    train_loss: List[float]
    test_loss:  List[float]

    def save(self, fname):
        with open(fname+'.pkl', "wb") as fp:  # Pickling
            pickle.dump(self, fp)
    def load(self,fname):
        with open(fname+'.pkl', "rb") as fp:  # Unpickling
            return pickle.load(fp)