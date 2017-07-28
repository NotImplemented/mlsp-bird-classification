import tensorflow
import numpy

def weight_variable(shape):

    initial = tensorflow.contrib.layers.xavier_initializer_conv2d()
    variable = tensorflow.Variable(initial(shape))

    #initial = tensorflow.truncated_normal(shape, stddev = 0.4)
    #variable = tensorflow.Variable(initial)

    return variable

def gaussian_filter(kernel_shape):
    filter = numpy.zeros(kernel_shape)

    def gauss(x, y, sigma = 2.0):
        Z = 2 * numpy.pi * sigma ** 2
        return  1.0 / Z * numpy.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))

    mid_x = numpy.floor(kernel_shape[0] / 2.0)
    mid_y = numpy.floor(kernel_shape[1] / 2.0)

    for i in range(kernel_shape[0]):
        for j in range(kernel_shape[1]):
            filter[i, j] = gauss(i - mid_x, j - mid_y)

    return filter / numpy.sum(filter)


def bias_variable(shape):
    initial = tensorflow.constant(0.06, shape = shape)
    return tensorflow.Variable(initial)

def convolution_layer(x, W):
    return tensorflow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pooling_layer(x, height = 2, width = 2):
    return tensorflow.nn.max_pool(x, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='VALID')

def avg_pooling_layer(x, height = 2, width = 2):
    return tensorflow.nn.avg_pool(x, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='VALID')

def tanh_layer(x):
    return tensorflow.nn.tanh(x)

def variable_summaries(var):

    with tensorflow.name_scope('summaries'):

        mean = tensorflow.reduce_mean(var)
        tensorflow.summary.scalar('mean', mean)

        with tensorflow.name_scope('stddev'):
            stddev = tensorflow.sqrt(tensorflow.reduce_mean(tensorflow.square(var - mean)))

        tensorflow.summary.scalar('stddev', stddev)
        tensorflow.summary.scalar('max', tensorflow.reduce_max(var))
        tensorflow.summary.scalar('min', tensorflow.reduce_min(var))
        tensorflow.summary.histogram('histogram', var)

def create_convolution_layer(layer_name, input_tensor, width, height, input_features, output_features):

    # TODO: Calculate input features count
    # TODO: Introduce window size argument

    with tensorflow.name_scope(layer_name):

        with tensorflow.name_scope('weights'):
            Weight_convolution = weight_variable([width, height, input_features, output_features])
            variable_summaries(Weight_convolution)

        with tensorflow.name_scope('biases'):
            bias_convolution = bias_variable([output_features])
            variable_summaries(bias_convolution)

        h_convolution = tensorflow.nn.relu(convolution_layer(input_tensor, Weight_convolution) + bias_convolution)
        print('h_convolution size = {}'.format(h_convolution.get_shape()))

        tensorflow.summary.histogram('activations-convolution', h_convolution)

        return h_convolution

def create_avg_pooling_layer(layer_name, input_tensor, width, height):

    with tensorflow.name_scope(layer_name):
        h_pooling = avg_pooling_layer(input_tensor)
        print('h_pooling size = {}'.format(h_pooling.get_shape()))

        tensorflow.summary.histogram('subsampling', h_pooling)

        return h_pooling

def create_tanh_layer(layer_name, input_tensor):

    with tensorflow.name_scope(layer_name):
        h_tanh = tanh_layer(input_tensor)
        print('tahn size = {}'.format(h_tanh.get_shape()))

        tensorflow.summary.histogram('subsampling', h_tanh)

        return h_tanh

def create_max_pooling_layer(layer_name, input_tensor, width = 2, height = 2):

    with tensorflow.name_scope(layer_name):
        h_pooling = max_pooling_layer(input_tensor)
        print('h_pooling size = {}'.format(h_pooling.get_shape()))

        tensorflow.summary.histogram('activations-pooling', h_pooling)

        return h_pooling

def create_contrast_layer(x, shape, layer_name):

    with tensorflow.name_scope(layer_name):
        gaussian = gaussian_filter(shape)
        filter = tensorflow.constant(gaussian, tensorflow.float32)
        reshaped_filter = tensorflow.reshape(filter, [13, 13, 1, 1])
        weights = tensorflow.Variable(reshaped_filter)
        h_convolution = tensorflow.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

        print('h_convolution size = {}'.format(h_convolution.get_shape()))
        tensorflow.summary.histogram('activations-convolution', h_convolution)

        return h_convolution

def create_convolution_layers(input_image):

    pool_0th = create_max_pooling_layer('pool-0th', input_image)

    conv_1st = create_convolution_layer('conv-1st', pool_0th, 5, 5, 1, 16)
    pool_1st = create_max_pooling_layer('pool-1st', conv_1st)

    conv_2nd = create_convolution_layer('conv-2nd', pool_1st, 5, 5, 16, 32)
    pool_2nd = create_max_pooling_layer('pool-2nd', conv_2nd)

    conv_3rd = create_convolution_layer('conv-3rd', pool_2nd, 5, 5, 32, 64)
    pool_3rd = create_max_pooling_layer('pool-3rd', conv_3rd)

    conv_4th = create_convolution_layer('conv-4th', pool_3rd, 5, 5, 64, 128)
    pool_4th = create_max_pooling_layer('pool-4th', conv_4th)
    
    conv_5th = create_convolution_layer('conv-5th', pool_4th, 4, 4, 128, output_classes)
    tanh = create_tanh_layer('tanh', conv_5th)

    return pool_5th

def create_fully_connected_layer(layer_name, input_layer, input_features, output_features):

    with tensorflow.name_scope(layer_name):

        Weight_fully_connected = weight_variable([input_features, output_features])
        variable_summaries(Weight_fully_connected)

        bias_fully_connected = bias_variable([output_features])
        variable_summaries(bias_fully_connected)

        input_flat = tensorflow.reshape(input_layer, [-1, input_features])
        h_fully_connected = tensorflow.nn.relu(tensorflow.matmul(input_flat, Weight_fully_connected) + bias_fully_connected)

        return h_fully_connected

def create_fully_connected_layers(input_layer, output_classes, keep_probability):

    _, num_rows, num_columns, num_features = input_layer.get_shape().as_list()

    fc_1st = create_fully_connected_layer('fc-1st', input_layer, num_rows * num_columns * num_features, 1024)
    fc1_drop = tensorflow.nn.dropout(fc_1st, keep_probability)

    fc_2nd = create_fully_connected_layer('fc-2nd', fc1_drop, 1024, output_classes)

    return fc_2nd

def create_schema(x_image, output_classes, keep_probability):

    contrast = create_contrast_layer(x_image, [13, 13], 'contrast-0th')

    pool_0th = create_max_pooling_layer('pool-0th', contrast)

    conv_1st = create_convolution_layer('conv-1st', pool_0th, 5, 5, 1, 16)
    pool_1st = create_max_pooling_layer('pool-1st', conv_1st)

    conv_2nd = create_convolution_layer('conv-2nd', pool_1st, 5, 5, 16, 32)
    pool_2nd = create_max_pooling_layer('pool-2nd', conv_2nd)

    conv_3rd = create_convolution_layer('conv-3rd', pool_2nd, 5, 5, 32, 64)
    pool_3rd = create_max_pooling_layer('pool-3rd', conv_3rd)

    conv_4th = create_convolution_layer('conv-4th', pool_3rd, 5, 5, 64, 128)
    pool_4th = create_max_pooling_layer('pool-4th', conv_4th)
    
    conv_5th = create_convolution_layer('conv-5th', pool_4th, 4, 4, 128, output_classes)
    tanh = create_tanh_layer('tanh', conv_5th)

    output = tensorflow.nn.max_pool(tanh, ksize=[1, 1, 32, 1], strides=[1, 1, 1, 1], padding='VALID', name='predictions')
    print('output size = {}'.format(output.get_shape()))


    return output
