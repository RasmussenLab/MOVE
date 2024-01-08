__all__ = []


class CudaIsNotAvailable(RuntimeError):
    """CUDA is not available."""

    def __init__(self) -> None:
        super().__init__(self.__class__.__doc__)


class ShapeAndWeightMismatch(ValueError):
    def __init__(self, num_shapes, num_weights) -> None:
        message = (
            f"Mismatch between supplied number of dataset shapes ({num_shapes})"
            f" and number of dataset weights ({num_weights})."
        )
        super().__init__(message)


class UnsetProperty(ValueError):
    def __init__(self, property_name: str) -> None:
        super().__init__(f"{property_name} has not been set")
