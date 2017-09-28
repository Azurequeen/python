import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *






class ManyLayer_BN_ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    [conv -sbn - relu]*3 - [conv - sbn - relu - pool]*1 - [affine - vbn - relu]*2 - [affine - vbn]*1 - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 use_batchnorm=False,dropout=0, dtype=np.float32,seed=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.filter_size= filter_size
        self.conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        C, H, W = input_dim
        F = num_filters
        HH = filter_size
        WW = filter_size

        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params['b1'] = np.zeros(F)

        self.params['gamma1'] = np.ones(F)
        self.params['beta1'] = np.zeros(F)

        self.params['W2'] = weight_scale * np.random.randn(F, F, HH, WW)
        self.params['b2'] = np.zeros(F)

        self.params['gamma2'] = np.ones(F)
        self.params['beta2'] = np.zeros(F)

        self.params['W3'] = weight_scale * np.random.randn(F, F, HH, WW)
        self.params['b3'] = np.zeros(F)

        self.params['gamma3'] = np.ones(F)
        self.params['beta3'] = np.zeros(F)

        self.params['W4'] = weight_scale * np.random.randn(F, F, HH, WW)
        self.params['b4'] = np.zeros(F )

        self.params['gamma4'] = np.ones(F)
        self.params['beta4'] = np.zeros(F)

        self.params['W5'] = weight_scale * np.random.randn(F * H * W / 4, hidden_dim)
        self.params['b5'] = np.zeros(hidden_dim)

        self.params['gamma5'] = np.ones(hidden_dim)
        self.params['beta5'] = np.zeros(hidden_dim)

        self.params['W6'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
        self.params['b6'] = np.zeros(hidden_dim)

        self.params['gamma6'] = np.ones(hidden_dim)
        self.params['beta6'] = np.zeros(hidden_dim)

        self.params['W7'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b7'] = np.zeros(num_classes)

     #   self.params['gamma7'] = np.ones(num_classes)
     #   self.params['beta7'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed


        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(7)]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        W7, b7 = self.params['W7'], self.params['b7']

        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        gamma4, beta4 = self.params['gamma4'], self.params['beta4']
        gamma5, beta5 = self.params['gamma5'], self.params['beta5']
        gamma6, beta6 = self.params['gamma6'], self.params['beta6']
#        gamma7, beta7 = self.params['gamma7'], self.params['beta7']



        conv_param = self.conv_param
        pool_param = self.pool_param
        bn_params = self.bn_params

        mode = 'test' if y is None else 'train'
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode




        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        if self.use_batchnorm:
            out1, cache1 = conv_sbn_relu_forward(X, W1, b1, conv_param, gamma1, beta1, bn_params[0])
            out1, cache11 = dropout_forward(out1, self.dropout_param)

            out2, cache2 = conv_sbn_relu_forward(out1, W2, b2, conv_param, gamma2, beta2, bn_params[1])
            out2, cache22 = dropout_forward(out2, self.dropout_param)

            out3, cache3 = conv_sbn_relu_forward(out2, W3, b3, conv_param, gamma3, beta3, bn_params[2])
            out3, cache33 = dropout_forward(out3, self.dropout_param)

            out4, cache4 = conv_sbn_relu_pool_forward(out3, W4, b4, conv_param, pool_param, gamma4, beta4, bn_params[3])
            out4, cache44 = dropout_forward(out4, self.dropout_param)

            out5, cache5 = affine_vbn_relu_forward(out4, W5, b5, gamma5, beta5, bn_params[4])
            out5, cache55 = dropout_forward(out5, self.dropout_param)

            out6, cache6 = affine_vbn_relu_forward(out5, W6, b6, gamma6, beta6, bn_params[5])
            out6, cache66 = dropout_forward(out6, self.dropout_param)

            out7, cache7 = affine_forward(out6, W7, b7)
            out7, cache77 = dropout_forward(out7, self.dropout_param)

        else:
            out1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
            out2, cache2 = conv_relu_forward(out1, W2, b2, conv_param)
            out3, cache3 = conv_relu_forward(out2, W3, b3, conv_param)
            out4, cache4 = conv_relu_pool_forward(out3, W4, b4, conv_param, pool_param)
            out5, cache5 = affine_relu_forward(out4, W5, b5)
            out6, cache6 = affine_relu_forward(out5, W6, b6)
            out7, cache7 = affine_forward(out6, W7, b7)

        scores = out7
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        reg = self.reg
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)
                             + np.sum(W4 * W4) + np.sum(W5 * W5)
                             + np.sum(W6 * W6) + np.sum(W7 * W7)
                             )

        if self.use_batchnorm:
            dout7 = dropout_backward(dx, cache77)
            dout6, dW7, db7 = affine_backward(dout7, cache7)

            dout6 = dropout_backward(dout6, cache66)
            dout5, dW6, db6, dgamma6, dbeta6 = affine_vbn_relu_backward(dout6, cache6)

            dout5 = dropout_backward(dout5, cache55)
            dout4, dW5, db5, dgamma5, dbeta5 = affine_vbn_relu_backward(dout5, cache5)

            dout4 = dropout_backward(dout4, cache44)
            dout3, dW4, db4, dgamma4, dbeta4 = conv_sbn_relu_pool_backward(dout4, cache4)

            dout3 = dropout_backward(dout3, cache33)
            dout2, dW3, db3, dgamma3, dbeta3 = conv_sbn_relu_backward(dout3, cache3)

            dout2 = dropout_backward(dout2, cache22)
            dout1, dW2, db2, dgamma2, dbeta2 = conv_sbn_relu_backward(dout2, cache2)

            dout1 = dropout_backward(dout1, cache11)
            dX, dW1, db1, dgamma1, dbeta1 = conv_sbn_relu_backward(dout1, cache1)
        else:
            dout6, dW7, db7 = affine_backward(dx, cache7)
            dout5, dW6, db6 = affine_relu_backward(dout6, cache6)
            dout4, dW5, db5 = affine_relu_backward(dout5, cache5)
            dout3, dW4, db4 = conv_relu_pool_backward(dout4, cache4)
            dout2, dW3, db3 = conv_relu_backward(dout3, cache3)
            dout1, dW2, db2 = conv_relu_backward(dout2, cache2)
            dX, dW1, db1 = conv_relu_backward(dout1, cache1)


        dW1 += reg * W1
        dW2 += reg * W2
        dW3 += reg * W3
        dW4 += reg * W4
        dW5 += reg * W5
        dW6 += reg * W6
        dW7 += reg * W7

        if self.use_batchnorm:
            grads['W1'], grads['b1'] = dW1, db1
            grads['W2'], grads['b2'] = dW2, db2
            grads['W3'], grads['b3'] = dW3, db3
            grads['W4'], grads['b4'] = dW4, db4
            grads['W5'], grads['b5'] = dW5, db5
            grads['W6'], grads['b6'] = dW6, db6
            grads['W7'], grads['b7'] = dW7, db7

            grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
            grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
            grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
            grads['gamma4'], grads['beta4'] = dgamma4, dbeta4
            grads['gamma5'], grads['beta5'] = dgamma5, dbeta5
            grads['gamma6'], grads['beta6'] = dgamma6, dbeta6
   #         grads['gamma7'], grads['beta7'] = dgamma7, dbeta7
        else:
            grads['W1'], grads['b1'] = dW1, db1
            grads['W2'], grads['b2'] = dW2, db2
            grads['W3'], grads['b3'] = dW3, db3
            grads['W4'], grads['b4'] = dW4, db4
            grads['W5'], grads['b5'] = dW5, db5
            grads['W6'], grads['b6'] = dW6, db6
            grads['W7'], grads['b7'] = dW7, db7


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def conv_sbn_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  aa,sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(aa)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache,sbn_cache)
  return out, cache


def conv_sbn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache,sbn_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  daa,dgamma,dbeta = spatial_batchnorm_backward(da,sbn_cache)
  dx, dw, db = conv_backward_fast(daa, conv_cache)
  return dx, dw, db ,dgamma,dbeta


def affine_vbn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  aa, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(aa)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_vbn_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  daa, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(daa, fc_cache)
  return dx, dw, db, dgamma, dbeta


def conv_sbn_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    aa, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(aa)
    cache = (conv_cache, relu_cache, sbn_cache)
    return out, cache


def conv_sbn_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache, sbn_cache = cache
    da = relu_backward(dout, relu_cache)
    daa, dgamma, dbeta = spatial_batchnorm_backward(da, sbn_cache)
    dx, dw, db = conv_backward_fast(daa, conv_cache)
    return dx, dw, db, dgamma, dbeta





pass








