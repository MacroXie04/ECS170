"""CNN for CIFAR-10: 32x32 RGB objects, 10 classes."""
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    dataset_key = 'cifar'
    input_shape = (3, 32, 32)
    num_classes = 10
