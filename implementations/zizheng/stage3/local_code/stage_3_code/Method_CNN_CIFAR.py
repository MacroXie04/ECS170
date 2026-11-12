"""CNN entry point for the CIFAR dataset."""
from .Method_CNN import Method_CNN
from .Configurations import configurations


class Method_CNN_CIFAR(Method_CNN):
    def __init__(self, num_classes=10, config=None):
        super().__init__((3, 32, 32), num_classes, config or configurations('CIFAR')['baseline'])
