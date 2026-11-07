"""CNN for ORL: 112x92 faces on one channel, 40 people."""
from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    dataset_key = 'orl'
    input_shape = (1, 112, 92)
    num_classes = 40
