import os
import sys
import numpy
import tensorflow
from skimage.measure.tests.test_pnpoly import test_npnpoly
from tensorflow.contrib.slim.python.slim.model_analyzer import tensor_description

import prepare_data
import nn_schema

batch_size = 16
learning_epochs = 48
output_classes = 87
learning_rate = 0.000002

summaries_directory = (os.path.join(os.getcwd(), 'summary'))

input_size_height = prepare_data.image_rows
input_size_width = prepare_data.image_columns

keep_probability = tensorflow.placeholder(tensorflow.float32)

x_place = tensorflow.placeholder(tensorflow.float32, shape=[None, input_size_height, input_size_width])
print('Input tensor size = {}'.format(x_place.get_shape()))

y_place = tensorflow.placeholder(tensorflow.float32, shape=[None, output_classes])
print('Output tensor size = {}'.format(y_place.get_shape()))

x_image = tensorflow.reshape(x_place, [-1, input_size_height, input_size_width, 1])
print('Image tensor size = {}'.format(x_image.get_shape()))

y_output = nn_schema.create_schema(x_image, output_classes, keep_probability)
y_output_sigmoid = tensorflow.nn.sigmoid(y_output)


with tensorflow.name_scope('cross_entropy'):
    with tensorflow.name_scope('difference'):
        sigmoid_cross_entropy_with_logits = tensorflow.nn.sigmoid_cross_entropy_with_logits(y_output, y_place)
        tensorflow.summary.histogram('sigmoid_cross_entropy_with_logits', sigmoid_cross_entropy_with_logits)

    with tensorflow.name_scope('total'):
        cross_entropy = tensorflow.reduce_mean(sigmoid_cross_entropy_with_logits)

tensorflow.summary.scalar('cross_entropy', cross_entropy)

with tensorflow.name_scope('train'):
    train_step = tensorflow.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

session = tensorflow.InteractiveSession()

merged = tensorflow.summary.merge_all()
train_writer = tensorflow.summary.FileWriter(os.path.join(summaries_directory, 'train'), session.graph)
test_writer = tensorflow.summary.FileWriter(os.path.join(summaries_directory, 'test'))

tensorflow.global_variables_initializer().run()

index = 0

(images, labels) = prepare_data.prepare_train_dataset()
print('Train data preparing is complete.')

print('Input image height = {} width = {}'.format(input_size_height, input_size_width))
print('Output classes = {}'.format(output_classes))

print('Batch size = {}'.format(batch_size))
print('Learning rate = {}'.format(learning_rate))

print('Train data size = {}'.format(len(images)))
print('Batch size = {}'.format(batch_size))
print('Learning epochs = {}'.format(learning_epochs))


batch = numpy.ndarray((batch_size, input_size_height, input_size_width))
label = numpy.ndarray((batch_size, output_classes))

step = 0
while(True):

    step += 1
    for j in range(batch_size):

        batch[j, :] = images[index % len(images)]
        label[j, :] = labels[index % len(images)]

        index += 1

    epoch = index / len(images) + 1
    summary, _, cross_entropy_batch = session.run([merged, train_step, cross_entropy], feed_dict = {x_place: batch, y_place: label, keep_probability: 0.5})

    print("Step #%d Epoch #%d: cross-entropy = %g, images = %d" % (step, epoch, cross_entropy_batch, index))
    test_writer.add_summary(summary, step)

    if epoch >= learning_epochs:
        break

print('Training is completed.\n')

(test_files, test_images) = prepare_data.prepare_test_dataset()
print('Test data preparing is complete.')
print('Input image height = {} width = {}'.format(input_size_height, input_size_width))
print('Output classes = {}'.format(output_classes))
print('Test data size = {}'.format(len(test_images)))
print('Starting evaluating predictions.\n')

with open('test_predictions.csv', 'w') as test_predictions_file:

    test_predictions_file.write('ID,Probability\n')

    for i in range(len(test_files)):
        file_name = test_files[i]
        test_image = test_images[i]
        output = y_output_sigmoid.eval(feed_dict = {x_place: test_image.reshape((1, input_size_height, input_size_width)), keep_probability:1.0})
        for j in range(output_classes):
            test_predictions_file.write('{}_classnumber_{}, {}\n'.format(file_name, j + 1, output[(0, j)]) )

with open('train_predictions.csv', 'w') as test_predictions_file:
    test_predictions_file.write('ID,Probability\n')

    for i in range(len(images)):
        train_image = images[i]
        output = y_output_sigmoid.eval(
            feed_dict={x_place: train_image.reshape((1, input_size_height, input_size_width)), keep_probability:1.0})
        for j in range(output_classes):
            test_predictions_file.write('train_{}_classnumber_{}, {}\n'.format(i + 1, j + 1, output[(0, j)]))

print('Prediction is completed.')