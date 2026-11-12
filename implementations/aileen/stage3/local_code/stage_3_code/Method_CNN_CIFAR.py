'''
CNN for CIFAR-10 colour objects (3 x 32 x 32, 10 classes)
'''
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    in_channels = 3
    image_size = (32, 32)
    num_classes = 10
    conv_channels = [32, 64, 128]
    kernel_size = 3
    padding = 1
    pool = 'max'
    hidden_dim = 256
    dropout = 0.3
    batch_norm = True
    optimizer_name = 'adam'
    learning_rate = 1e-3
    batch_size = 128
    max_epoch = 20
    augment = True           # flip + random crop helps a lot on CIFAR
