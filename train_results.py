from typing import NamedTuple, List

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
