from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    def __init__(self, channels=(16, 32), **kwargs):
        super().__init__('ORL', (1, 112, 92), 40, channels, **kwargs)
