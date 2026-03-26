from enum import Enum


class LayerPurpose(Enum):
    """Tags regarding layer purpose.

    This is used to filter layers, when choosing a subset to serialize and
    deserialize. Note that a layer may have multiple purposes, and layer is
    included if the intersection of its purpose tags and the filter set is not
    empty.
    
    If no categorical or specific filter is provided to the (de)serialization
    method, IModel has a default presumption to only include INFERENCE layers.

    By default all layers are tagged with the INFERENCE purpose, unless
    explicitly overwritten upon layer instantiation.

    Purposes
    ---
    INFERENCE = 0:
        Layer necessary for inference in typical operation. Always serialized in
        default filters.
    TRAINING = 1:
        Layer necessary in the training and/or fine-tuning process.
        - Checkpoint filter: Serialized.
        - Debugging filter: Not serialized.
        - Inference filter: Not serialized.
    DEBUGGING = 2:
        Layer necessary for debugging and profiling tasks.
        - Checkpoint filter: Serialized.
        - Debugging filter: Serialized.
        - Inference filter: Not serialized.
    VALIDATION = 3:
        Layer necessary for validation, not only training-related.
        - Checkpoint filter: Serialized.
        - Debugging filter: Serialized.
        - Inference filter: Serialized.

    """
    INFERENCE = 0
    TRAINING = 1
    DEBUGGING = 2


class LayerFilter(Enum):
    INFERENCE_ONLY = {LayerPurpose.INFERENCE}
    DEBUGGING = (
        LayerPurpose.INFERENCE,
        LayerPurpose.DEBUGGING
    )
    FULL_CHECKPOINT = (
        LayerPurpose.INFERENCE,
        LayerPurpose.TRAINING,
        LayerPurpose.DEBUGGING
    )

    @property
    def as_set(self):
        return set(self.value)


class LayerType(Enum):
    """Enumeration of all implemented layer classes.

    Any layer using the ILayer abstract interface contract must be included
    here for the purposes of identification.
    """
    LINEAR = 0
    RELU = 1
    SOFTMAX = 2
    SIGMOID = 3
    AVERAGINGLINEAR = 4


class LossFunctionType(Enum):
    """Enumeration of all implemented loss function classes.

    Any loss funciton using the ILossFunction abstract interface contract must
    be included here for the purposes of identification.
    """
    CROSSENTROPY = 0
    CATEGORICALCROSSENTROPY = 1


class PerformanceMetric(Enum):
    """Enumeration of all possible scheduler metrics.

    Currently specifically for PlateauScheduler, however, any criterion-driven
    schedulers using the IScheduler abstract interface contract can conceivable
    make use of this enumeration.

    TODO Rewrite into more generic maximze/minimize metrics.

    Metrics:
    ---
    Accuracy (float):
        Represents a fraction of correct matches in the current epoch (maximized
        criterion).
    Loss (float):
        Represents loss function value (minimized criterion).

    """
    ACCURACY = 0
    LOSS = 1
