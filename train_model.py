import os
import sys
import numpy
import shutil
import random
import tensorflow
from skimage.measure.tests.test_pnpoly import test_npnpoly
from tensorflow.contrib.slim.python.slim.model_analyzer import tensor_description

import prepare_data
import nn_schema

batch_size = 8
learning_epochs = 100
output_classes = 19
learning_rate = 0.00001

def shuffle(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)

    return (a, b)

def normalize(x):
    mx = numpy.max(x)
    mn = numpy.min(x)

    if mn != mx:
        x = (x - mn) / (mx - mn)
    else:
        x -= x
    
    return x

summaries_directory = (os.path.join('tensorflow', 'summary'))

input_size_height = prepare_data.image_rows
input_size_width = prepare_data.image_columns

keep_probability = tensorflow.placeholder(tensorflow.float32)

x_place = tensorflow.placeholder(tensorflow.float32, shape=[None, input_size_height, input_size_width], name="input")
print('Input tensor size = {}'.format(x_place.get_shape()))

y_place = tensorflow.placeholder(tensorflow.float32, shape=[None, output_classes])
print('Output tensor size = {}'.format(y_place.get_shape()))

x_image = tensorflow.reshape(x_place, [-1, input_size_height, input_size_width, 1])
print('Image tensor size = {}'.format(x_image.get_shape()))

y_output = nn_schema.create_schema(x_image, output_classes, keep_probability)
y_output_sigmoid = tensorflow.nn.sigmoid(y_output, name="predictions")

print('Cleaning summary folder = {}'.format(summaries_directory))
shutil.rmtree(summaries_directory, ignore_errors=True)

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
saver = tensorflow.train.Saver()

merged = tensorflow.summary.merge_all()
train_writer = tensorflow.summary.FileWriter(os.path.join(summaries_directory, 'train'), session.graph)
test_writer = tensorflow.summary.FileWriter(os.path.join(summaries_directory, 'test'))

tensorflow.global_variables_initializer().run()

(train_images, train_labels, test_images, test_ids) = prepare_data.prepare_dataset()

print('Test data preparing is complete.')
print('Input image height = {} width = {}'.format(input_size_height, input_size_width))
print('Output classes = {}'.format(output_classes))

print('Input image height = {} width = {}'.format(input_size_height, input_size_width))
print('Output classes = {}'.format(output_classes))

print('Batch size = {}'.format(batch_size))
print('Learning rate = {}'.format(learning_rate))

print('Train data size = {}'.format(len(train_images)))
print('Batch size = {}'.format(batch_size))
print('Learning epochs = {}'.format(learning_epochs))
print('Summary folder = {}'.format(summaries_directory))


batch = numpy.ndarray((batch_size, input_size_height, input_size_width))
label = numpy.ndarray((batch_size, output_classes))

step = 0
index = 0
epoch = 0
train_size = len(train_images)

while(True):

    if epoch >= learning_epochs:
        break

    step += 1
    for j in range(batch_size):
        batch[j, :] = train_images[index % train_size]
        label[j, :] = train_labels[index % train_size]
        index += 1

    epoch = int((index + train_size - 1) / train_size)
    previous_epoch = int((index - batch_size + train_size - 1) / train_size)

    summary, _, cross_entropy_batch = session.run([merged, train_step, cross_entropy], feed_dict = {x_place: batch, y_place: label, keep_probability: 0.5})

    if previous_epoch > 0 and epoch != previous_epoch:
        (batch, label) = shuffle(batch, label)
        train_writer.add_summary(summary, epoch)
        print("Step #%d Epoch #%d: shuffle train data" % (step, epoch))
        print("Step #%d Epoch #%d: writing summary" % (step, epoch))

    output = y_output_sigmoid.eval(feed_dict = {x_place: batch, keep_probability: 1.0})
    print('Labels = {}'.format(label))
    print('Predictions = {}'.format( output))

    print("Step #%d Epoch #%d: cross-entropy = %g, images = %d" % (step, epoch, cross_entropy_batch, index))

print('Training is completed.\n')

for i in range(len(train_images)):
    image = numpy.ndarray((1, input_size_height, input_size_width))
    image[0,:] = train_images[i]

    output = y_output_sigmoid.eval(feed_dict = {x_place: image.reshape((1, input_size_height, input_size_width)), keep_probability: 1.0})
    print('Train image {}: labels = {}, predictions = {}'.format(i, train_labels[i], output))

print('Starting evaluating predictions.\n')

with open('test_predictions.csv', 'w') as test_predictions_file:
    test_predictions_file.write('Id,probability\n')

    for i in range(len(test_ids)):
        id = test_ids[i]
        test_image = test_images[i]
        output = y_output_sigmoid.eval(feed_dict = {x_place: test_image.reshape((1, input_size_height, input_size_width)), keep_probability: 1.0})
        output = normalize(output)
        for j in range(output_classes):
            combined_id = int(id) * 100 + j
            probability = output[(0, j)]
            test_predictions_file.write('{}, {}\n'.format(combined_id , probability) )


model_path = saver.save(session, ".\mlsp_classification_model")
print("Model saved in file: {}\n".format(model_path))


print('Prediction is completed.')