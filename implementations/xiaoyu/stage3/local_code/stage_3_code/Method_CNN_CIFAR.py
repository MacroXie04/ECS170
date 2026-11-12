from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    def __init__(self, channels=(32, 64), **kwargs):
        super().__init__('CIFAR', (3, 32, 32), 10, channels, **kwargs)
