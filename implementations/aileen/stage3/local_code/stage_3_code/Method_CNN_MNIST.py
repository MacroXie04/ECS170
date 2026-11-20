'''
CNN for MNIST digits (1 x 28 x 28, 10 classes)
'''
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST(Method_CNN):
    in_channels = 1
    image_size = (28, 28)
    num_classes = 10
    conv_channels = [32, 64]
    kernel_size = 3
    padding = 1
    pool = 'max'
    hidden_dim = 128
    dropout = 0.3
    batch_norm = True
    optimizer_name = 'adam'
    learning_rate = 1e-3
    batch_size = 128
    max_epoch = 10
