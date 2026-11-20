from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    def __init__(self, channels=(16, 32), **kwargs):
        super().__init__('MNIST', (1, 28, 28), 10, channels, **kwargs)
