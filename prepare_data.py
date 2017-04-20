import os
import numpy
import numpy.fft
import struct
import matplotlib.image
import matplotlib.pyplot
import tensorflow
import fourier
from time import ctime
import time as time_module


output_classes = 87

sample_size = 512
time_shift = int(sample_size * 0.75)
max_song_length = 6.5
max_frame_rate = 44100
max_spectrogram_length = int((max_frame_rate * max_song_length - sample_size) / time_shift + 1)
rows = int(sample_size / 2)

image_rows = rows
image_columns = max_spectrogram_length
dataset_size = 1024

wav_path_train = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', 'train')
wav_path_test = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', 'test')

labels_path = os.path.join('NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS', 'nips4b_birdchallenge_train_labels.csv')

def cook_spectrogram(file_path):

    _, extension = os.path.splitext(file_path)

    # read raw sound and build spectrogram
    sound = wave.open(file_path, 'r')
    frames_count = sound.getnframes()
    frame_rate = sound.getframerate()
    sample_width = sound.getsampwidth()
    sound_channels = sound.getnchannels()

    print('Processing file {}: channels = {}, frames count = {}, frame rate = {}, sample width = {}, duration = {:.4f} seconds'.format(
            file_path, sound_channels, frames_count, frame_rate, sample_width, float(frames_count) / frame_rate))

    raw_sound = sound.readframes(frames_count)
    time = 0

    spectrogram = numpy.ndarray((rows, max_spectrogram_length))

    index = 0
    while time + sample_size < frames_count and index < max_spectrogram_length:

        raw_bytes = raw_sound[time * 2: (time + sample_size) * 2]
        converted_data = numpy.fromstring(raw_bytes, dtype=numpy.int16)
        fourier = numpy.fft.fft(converted_data)

        # get only half of fourier coefficients
        fourier_normalized_converted = numpy.ndarray((1, rows))
        fourier_normalized_absolute = numpy.ndarray((1, rows))

        epsilon = 0.000002
        minimal_greater_than_zero = epsilon
        for i in range(rows):

            value = numpy.abs(fourier[i])
            if value > 0.0 and minimal_greater_than_zero > value:
                minimal_greater_than_zero = value
            fourier_normalized_absolute[(0, i)] = value + 1.0

        # take logarithm and check for infinity
        for i in range(rows):
            fourier_normalized_converted[(0, i)] = 10*numpy.log10(max(fourier_normalized_absolute[(0, i)], minimal_greater_than_zero))

        #for i in range(rows):
            #fourier_normalized_converted[(0, i)] = fourier_normalized_absolute[(0, i)]

        #mx = numpy.max(fourier_normalized_converted)
        #mn = numpy.min(fourier_normalized_converted)

        #fourier_normalized_converted = (fourier_normalized_converted - mn) / (mx - mn)

        spectrogram[:, index] = fourier_normalized_converted

        #if numpy.max(fourier_normalized_converted) == float('inf'):
        #    print("!")

        time += time_shift
        index += 1

    # append if necessary
    start_index = 0
    while index < max_spectrogram_length:
        spectrogram[:, index] = spectrogram[:, start_index]
        start_index += 1
        index += 1

    mx = numpy.max(spectrogram)
    mn = numpy.min(spectrogram)

    spectrogram = (spectrogram - mn) / (mx - mn)

    print('Test file {}: max = {}, min = {}'.format(file_path, mx, mn))

    show_spectrogram(spectrogram, 'sample')
    return spectrogram

def show_spectrogram(spectrogram, description):

    display_count = 1
    figure = matplotlib.pyplot.figure()

    subplot = matplotlib.pyplot.subplot(display_count, 1, 1)

    if description is not None:
        matplotlib.pyplot.title(description.replace('_', ' '), fontsize = 10)

    subplot.set_yticklabels([])
    subplot.set_xticklabels([])

    matplotlib.pyplot.imshow(spectrogram)
    matplotlib.pyplot.show()

def prepare_train_dataset():

    print('[' + ctime() + ']: Train data preparation has started.')
    start_time = time_module.time()

    class_offset = 4

    pixel_type = numpy.dtype(numpy.uint16)
    factor = 1.0 / (numpy.iinfo(pixel_type).max - numpy.iinfo(pixel_type).min)

    labels = []
    spectrograms = []

    file_labels = open(labels_path)
    file_labels.readline()
    species = file_labels.readline()
    auxiliary = file_labels.readline()

    print(species)
    print(auxiliary)

    class_names = species.split(',')[class_offset:]
    print(class_names)

    file_index = 0
    for file in os.listdir(wav_path_train):

        # read labels and set up classes
        line = file_labels.readline()
        classes = line.split(',')
        label = numpy.zeros((1, output_classes))
        description = ''

        for i in range(1, len(classes)):
            if classes[i]:
                label[(0, i - class_offset)] = 1.0
                if description:
                    description += ', '
                description += class_names[i-class_offset]

        if description is None:
            continue

        print('Description = {}'.format(description))

        file_path = os.path.abspath(os.path.join(wav_path_train, file))
        spectrogram = cook_spectrogram(file_path)
        if spectrogram is None:
            continue

        spectrograms.append(spectrogram)
        labels.append(label)

        file_index += 1
        if file_index >= dataset_size:
            break

    print('[' + ctime() + ']: Data preparation is complete.')
    end_time = time_module.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = {} minutes and {} seconds'.format(int(elapsed_time / 60), int(elapsed_time % 60)))

    return (spectrograms, labels)

def prepare_test_dataset():

    print('[' + ctime() + ']: Test data preparation has started.')
    start_time = time_module.time()

    files = []
    spectrograms = []

    file_index = 0
    for file in os.listdir(wav_path_test):

        label = numpy.zeros((1, output_classes))

        file_path = os.path.abspath(os.path.join(wav_path_test, file))
        spectrogram = cook_spectrogram(file_path)
        if spectrogram is None:
            continue

        spectrograms.append(spectrogram)
        files.append(os.path.basename(file_path))

        file_index += 1
        if file_index >= dataset_size:
            break

    print('[' + ctime() + ']: Test data preparation is complete.')
    end_time = time_module.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = {} minutes and {} seconds'.format(int(elapsed_time / 60), int(elapsed_time % 60)))

    return (files, spectrograms)
