from enum import Enum
import numpy as np


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
    DEBUGGING = 2:
        Layer necessary for debugging and profiling tasks.
    """
    INFERENCE = 0
    TRAINING = 1
    DEBUGGING = 2


class LayerFilter(Enum):
    """Presets for purpose filtering.

    All presets are given as tuples to maintain immutability, but can be turned
    to sets as required by the project using the as_set property.

    Properties
    ---
    as_set(self) -> set[LayerPurpose]:
        Turn enum member from immutable tuple into a set.

    Presets
    ---
    INFERENCE_ONLY = 0:
        Only (de)serialize layers needed for forward pass at inference time.
        Used for deployment-ready checkpoints or whenever all the data needed to
        continue training/fine-tune can be discarded.
    FULL_CHECKPOINT = 1:
        (De)serialize all known layer purposes defined within the model. Used
        when fine-tuning or training continuation is considered a possibility.
    """
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
    def as_set(self) -> set[LayerPurpose]:
        return set(self.value)


class LayerType(Enum):
    """Enumeration of all implemented layer classes.

    Any layer using the ILayer abstract interface contract must be included
    here for the purposes of identification. Newly implemented layers musy be
    appended at the end of the enumeration, preferably in an order following the
    position in layer.py.
    """
    LINEAR = 0
    RELU = 1
    SOFTMAX = 2
    SIGMOID = 3
    AVERAGINGLINEAR = 4


class LayerPrecision(Enum):
    """Enumeration of types preferred for ML.
    
    This is currently a simple wrapper for numpy types. Entries should always
    return numpy-compatible types.
    
    There exist projects to extend the numpy data type list, which might be used
    in the future. No plans to do this exist at this time, however.

    There also exist exotic and GPU-specific data types, such as packed balanced
    ternary etc. There are no plans to support those at this time.

    Precisions
    ---
    FP64 = np.float64
        Double precision (8 byte) floating point numbers. Useful for scientific
        tasks requiring very high accuracy. Bit layout:
        `1B: sign|11B: exponent|52B: mantissa`

    FP32 = np.float32
        Single precision (4 byte) floating point numbers. Default training
        precision in machine learning tasks. Bit layout:
        `1B: sign|8B: exponent|23B: mantissa`

    ~~BF16~~
        Half-precision (2 byte) floating point numbers with expanded exponent
        allocation, at the cost of having fewer mantissa bits. Bit layout:
        `1B: sign|8B: exponent|7B: mantissa`
        Not supported.

    FP16 = np.float16
        Half-precision (2 byte) floating point numbers. Not preferred for
        machine learning due to numerical instability. Bit layout:
        `1B: sign|5B: exponent|10B: mantissa`

    INT8 = np.int8
        1 byte integer numbers, used in quantized arithmetic.

    ~~INT4~~
        4 bit integer numbers, used in quantized arithmetic. Not supported.
    
    ~~K-quants/GGUF quants~~
        Number representations quantized to N bits. Examples include Q8, Q4 and
        down to Q2. Not supported.

    ~~BitNet~~
        Also known as b1.58 or sometimes Q1 to Q2 in dynamic quants (where it is
        not the sole precision applied). Not supported.

    Notes
    ---
    - BF16 is not supported by numpy and as such is not supported by the engine.
    - INT4 is not supported by numpy and as such is not supported by the engine.
    - Q1B is not supported by numpy and as such is not supported by
    the engine.
    """
    FP64 = np.float64
    FP32 = np.float32
    # BF16 = None # Currently not implemented.
    FP16 = np.float16
    INT8 = np.int8
    # INT4 = None # Currently not implemented.
    # Q1BT = None # Currently not implemented.


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
