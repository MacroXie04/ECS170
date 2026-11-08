"""CNN entry point for the MNIST dataset."""
from .Method_CNN import Method_CNN
from .Configurations import configurations


class Method_CNN_MNIST(Method_CNN):
    def __init__(self, num_classes=10, config=None):
        super().__init__((1, 28, 28), num_classes, config or configurations('MNIST')['baseline'])
