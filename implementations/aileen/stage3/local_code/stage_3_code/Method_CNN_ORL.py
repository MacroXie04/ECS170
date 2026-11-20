'''
CNN for ORL faces (1 x 112 x 92 after keeping one gray channel, 40 people, only 360 training images)
'''
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    in_channels = 1
    image_size = (112, 92)
    num_classes = 40
    conv_channels = [16, 32, 64]
    kernel_size = 3
    padding = 1
    pool = 'max'
    hidden_dim = 128
    dropout = 0.5            # tiny training set -> strong dropout
    batch_norm = True
    optimizer_name = 'adam'
    learning_rate = 1e-3
    batch_size = 32
    max_epoch = 30
