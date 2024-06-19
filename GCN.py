import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self,
                 units,
                 support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,):
        super(GraphConvolution, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.supports_masking = True

        self.support = support
        assert support >= 1


def build(self, input_shapes):
    features_shape = input_shapes[0]
    assert len(features_shape) == 2
    input_dim = features_shape[1]

    self.kernel = self.add_weight(shape=(input_dim * self.support, self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer)
    if self.use_bias:
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.kernel_regularizer)
    else:
        self.bias = None

    self.built = True


def call(self, inputs, mask=None):
    features = inputs[0]
    basis = inputs[1:]

    supports = list()
    for i in range(self.support):
        supports.append(K.dot(basis[i], features))
    supports = K.concatenate(supports, axis=1)
    output = K.dot(supports, self.kernel)

    if self.bias:
        output += self.bias

    return self.activation(output)
