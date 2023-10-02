__all__ = []

class ShapeAndWeightMismatch(ValueError):
    def __init__(self, num_shapes, num_weights) -> None:
        message = (
            f"Mismatch between supplied number of dataset shapes ({num_shapes})"
            f" and number of dataset weights ({num_weights})."
        )
        super().__init__(message)
