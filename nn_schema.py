import tensorflow

def weight_variable(shape):
    initial = tensorflow.truncated_normal(shape, stddev=0.12)
    return tensorflow.Variable(initial)

def bias_variable(shape):
    initial = tensorflow.constant(0.12, shape = shape)
    return tensorflow.Variable(initial)

def convolution_layer(x, W):
    return tensorflow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pooling_layer(x, height = 2, width = 2):
    return tensorflow.nn.max_pool(x, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding='SAME')


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


def create_convolution_layer(layer_name, input_tensor, input_features, output_features):

    # TODO: Calculate input features count
    # TODO: Introduce window size argument

    with tensorflow.name_scope(layer_name):

        with tensorflow.name_scope('weights'):
            Weight_convolution = weight_variable([6, 6, input_features, output_features])
            variable_summaries(Weight_convolution)

        with tensorflow.name_scope('biases'):
            bias_convolution = bias_variable([output_features])
            variable_summaries(bias_convolution)

        h_convolution = tensorflow.nn.relu(convolution_layer(input_tensor, Weight_convolution) + bias_convolution)
        print('h_convolution size = {}'.format(h_convolution.get_shape()))

        tensorflow.summary.histogram('activations-convolution', h_convolution)

        return h_convolution


def create_max_pooling_layer(layer_name, input_tensor):

    with tensorflow.name_scope(layer_name):
        h_pooling = max_pooling_layer(input_tensor)
        print('h_pooling size = {}'.format(h_pooling.get_shape()))

        tensorflow.summary.histogram('activations-pooling', h_pooling)

        return h_pooling

def create_convolution_layers(input_image):

    conv_1st = create_convolution_layer('conv-1st', input_image, 1, 16)
    pool_1st = create_max_pooling_layer('pool-1st', conv_1st)

    conv_2nd = create_convolution_layer('conv-2nd', pool_1st, 16, 32)
    pool_2nd = create_max_pooling_layer('pool-2nd', conv_2nd)

    conv_3rd = create_convolution_layer('conv-3rd', pool_2nd, 32, 64)
    pool_3rd = create_max_pooling_layer('pool-3rd', conv_3rd)

    conv_4th = create_convolution_layer('conv-4th', pool_3rd, 64, 128)
    pool_4th = create_max_pooling_layer('pool-4th', conv_4th)

    conv_5th = create_convolution_layer('conv-5th', pool_4th, 128, 256)
    pool_5th = create_max_pooling_layer('pool-5th', conv_5th)

    conv_6th = create_convolution_layer('conv-6th', pool_5th, 256, 512)
    pool_6th = create_max_pooling_layer('pool-6th', conv_6th)

    return pool_6th

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

    fc_1st = create_fully_connected_layer('fc-1st', input_layer, num_rows * num_columns * num_features, 256)
    fc1_drop = tensorflow.nn.dropout(fc_1st, keep_probability)

    fc_2nd = create_fully_connected_layer('fc-2nd', fc1_drop, 256, output_classes)

    return fc_2nd


def create_schema(x_image, output_classes, keep_probability):

    last_convolution = create_convolution_layers(x_image)
    output = create_fully_connected_layers(last_convolution, output_classes, keep_probability)

    return output
