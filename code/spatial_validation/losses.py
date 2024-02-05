import numpy as np
from numpy.typing import ArrayLike


class Loss:
    def __call__(self, targets: ArrayLike, predictions: ArrayLike) -> ArrayLike:
        """
        Compute a per element loss between targets and predictions

        Args:
            targets (ArrayLike): Values of field to be predicted, [n, 1].
            predictions (ArrayLike): Output of trained model at sites, [n, 1].

        Returns:
            ArrayLike: loss(targets, predictions). Generally [n, 1].
        """


class SquaredLoss(Loss):
    """
    Computes the squared loss, loss(x, y) = (x-y)^2
    """

    def __call__(self, targets: ArrayLike, predictions: ArrayLike) -> ArrayLike:
        return np.square(targets - predictions)


class AbsoluteError(Loss):
    """
    Computes the absolute error, loss(x, y) = |x-y|
    """

    def __call__(self, targets: ArrayLike, predictions: ArrayLike) -> ArrayLike:
        return np.abs(targets - predictions)


class TruncatedSquaredLoss(SquaredLoss):
    """
    Computes a clipped squared loss, i.e loss(x, y) = min((x-y)^2, max_value)
    """

    def __init__(self, max_value: float) -> None:
        assert max_value >= 0, "The maximum value of the loss must be non-negative"
        self.max_value = max_value

    def __call__(self, targets: ArrayLike, predictions: ArrayLike) -> ArrayLike:
        sq_losses = super().__call__(targets, predictions)
        return np.minimum(sq_losses, self.max_value)
