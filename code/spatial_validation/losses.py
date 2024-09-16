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

    @property
    def upper_bound(self) -> float:
        """
        Returns the upper bound of the loss function, returns 1 if no upper bound exists.
        """
        return 1.0


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


class AbsoluteExpError(Loss):
    """
    Computes the absolute error, loss(x, y) = |x-y|
    """

    def __call__(self, targets: ArrayLike, predictions: ArrayLike) -> ArrayLike:
        return np.abs(10**targets - 10**predictions)


class TruncatedLossMixin(Loss):
    """
    Mixin class for losses that have a truncation value
    """

    def __init__(self, max_value: float, scale_factor: float = 1.0) -> None:
        assert max_value >= 0, "The maximum value of the loss must be non-negative"
        self.max_value = max_value
        self.scale_factor = scale_factor

    def __call__(self, targets: ArrayLike, predictions: ArrayLike) -> ArrayLike:
        unclipped_losses = super().__call__(targets, predictions)
        return np.minimum(unclipped_losses, self.max_value)

    @property
    def upper_bound(self) -> float:
        return self.max_value / self.scale_factor


class TruncatedSquaredLoss(TruncatedLossMixin, SquaredLoss):
    """
    Computes a clipped squared loss, i.e loss(x, y) = min((x-y)^2, max_value)
    """


class TruncatedAbsoluteError(TruncatedLossMixin, AbsoluteError):
    """
    Computes a clipped absolute error, i.e loss(x, y) = min(|x-y|, max_value)
    """


class TruncatedAbsoluteExpError(TruncatedLossMixin, AbsoluteExpError):
    """
    Computes a clipped absolute error, i.e loss(x, y) = min(|x-y|, max_value)
    """
