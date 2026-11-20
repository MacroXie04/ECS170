"""CNN entry point for the ORL dataset."""
from .Method_CNN import Method_CNN
from .Configurations import configurations


class Method_CNN_ORL(Method_CNN):
    def __init__(self, num_classes=40, config=None):
        super().__init__((1, 112, 92), num_classes, config or configurations('ORL')['baseline'])
