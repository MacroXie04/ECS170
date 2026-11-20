"""CNN for MNIST: 28x28 grayscale digits, 10 classes."""
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    dataset_key = 'mnist'
    input_shape = (1, 28, 28)
    num_classes = 10
