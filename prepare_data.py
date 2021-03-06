import os
import numpy
import numpy.fft
import struct
import matplotlib.image
import matplotlib.pyplot
import tensorflow
import wave
import math
from time import ctime
import time as time_module


output_classes = 19

sample_size = 512
time_shift = int(sample_size * 0.25)
max_song_length = 10
max_frame_rate = 16000
max_spectrogram_length = int((max_frame_rate * max_song_length - sample_size) / time_shift + 1)
rows = int(sample_size / 2)

image_rows = rows
image_columns = max_spectrogram_length
dataset_size = 1024

wav_path = os.path.join('mlsp_contest_dataset', 'essential_data', 'src_wavs')
rec_id2filename = os.path.join('mlsp_contest_dataset', 'essential_data', 'rec_id2filename.txt')
rec_id2test = os.path.join('mlsp_contest_dataset', 'essential_data', 'CVfolds_2.txt')
rec_id2label = os.path.join('mlsp_contest_dataset', 'essential_data', 'rec_labels_test_hidden.txt')
rec_id2label_hidden = os.path.join('mlsp_contest_dataset', 'essential_data', 'rec_labels_test.txt')

filter = [] #['PC4_20100705_050000_0010', 'PC4_20100705_050000_0020']
show_image = False
use_hidden = True

def parse_int(value):
    try:
        int(value)
        return True
    except:
        return False

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

def local_contrast_subtractive_transform(spectrogram, rows, columns):

    result = numpy.copy(spectrogram)
    filter = gaussian_filter( [13, 13] )

    mid_x = int(numpy.floor(filter.shape[0] / 2.0))
    mid_y = int(numpy.floor(filter.shape[1] / 2.0))
    P = filter.shape[0]
    Q = filter.shape[1]

    for i in range(rows):
        for j in range(columns):
            x = 0.0
            for p in range(P):
                for q in range(Q):
                    i_from = i - mid_x + p
                    j_from = j - mid_y + q
                    if i_from >= 0 and i_from < rows and j_from >= 0 and j_from < columns:
                        x += spectrogram[i_from, j_from] * filter[p, q]
            result[i, j] = result[i, j] - x

    return result

def subsample_max(spectrogram, rows, columns):
    for i in range(0, rows-1, 2):
        for j in range(0, columns-1, 2):
            mx1 = max(spectrogram[i, j], spectrogram[i, j+1])
            mx2 = max(spectrogram[i+1, j], spectrogram[i+1, j+1])
            mx3 = max(mx1, mx2)

            spectrogram[i, j] = mx3
            spectrogram[i+1, j] = mx3
            spectrogram[i, j+1] = mx3
            spectrogram[i+1, j+1] = mx3

def erode(spectrogram, rows, columns):
    for i in range(rows):
        for j in range(columns):
            count = 0
            direction = [ [0,1], [1,0], [-1,0], [0,-1] ]
            for k in range(len(direction)):
                x = i + direction[k][0]
                y = j + direction[k][1]
                if x >= 0 and x < rows and y >= 0 and y < columns:
                    if spectrogram[x,y] == 1.0 or spectrogram[x,y] == 2.0:
                        count += 1
            if count != 4:
                spectrogram[i,j] = 2.0

    for i in range(rows):
        for j in range(columns):
            if spectrogram[i,j] == 2.0:
                spectrogram[i,j] = 0.0

def dilate(spectrogram, rows, columns):
    for i in range(rows):
        for j in range(columns):
            count = 0

            for a in range(2):
                for b in range(2):
                    x = i + a
                    y = j + b
                    if x >= 0 and x < rows and y >= 0 and y < columns:
                        if spectrogram[x,y] == 1.0:
                            count += 1
            if count > 0:
                spectrogram[i,j] = 2.0

    for i in range(rows):
        for j in range(columns):
            if spectrogram[i,j] == 2.0:
                spectrogram[i,j] = 1.0

def median_filtering(spectrogram, rows, columns):
    row_sum = numpy.ndarray(rows)
    column_sum = numpy.ndarray(columns)

    for i in range(rows - (first_cut + last_cut)):
        for j in range(columns):
            column_sum[j] += spectrogram[i, j]
            row_sum[i] += spectrogram[i, j]
    
    row_sum /= columns
    column_sum /= (rows - (first_cut + last_cut))

    for i in range(rows - (first_cut + last_cut)):
        for j in range(columns):
            if spectrogram[i, j] > column_sum[j] * 2.5 or spectrogram[i, j] > row_sum[i] * 2.5:
                spectrogram[i, j] = 1.0
            else:
                spectrogram[i, j] = 0.0;

def print_statistics(spectrogram, file_path):

    mx = numpy.max(spectrogram)
    mn = numpy.min(spectrogram)

    mean = numpy.mean(spectrogram)
    std = numpy.std(spectrogram)

    print('{}: max = {:.4f}, min = {:.4f}, mean = {:.4f}, stddev = {:.4f}'.format(file_path, mx, mn, mean, std))

def cook_spectrogram(file_path):

    _, extension = os.path.splitext(file_path)

    # read raw sound and build spectrogram
    sound = wave.open(file_path, 'r')
    frames_count = sound.getnframes()
    frame_rate = sound.getframerate()
    sample_width = sound.getsampwidth()
    sound_channels = sound.getnchannels()

    print('{}: channels = {}, frames = {}, frame rate = {}, sample width = {}, duration = {:.4f} seconds'.format(
            file_path, sound_channels, frames_count, frame_rate, sample_width, float(frames_count) / frame_rate))

    raw_sound = sound.readframes(frames_count)
    time = 0

    first_cut = 0
    last_cut = 0

    spectrogram = numpy.ndarray((rows - (first_cut + last_cut) , max_spectrogram_length))
    print('Raw sound data length = {}'.format(len(raw_sound)))

    hann = numpy.ndarray((1, sample_size))
    for n in range(sample_size):
        hann[(0, n)] = 0.5 * (1 - numpy.cos(2 * math.pi * n / (sample_size-1)))

    print('{}: raw sound length = {}'.format(file_path, len(raw_sound)))
    print('{}: spectrogram columns = {}'.format(file_path, max_spectrogram_length))

    index = 0
    while time + sample_size * 2 <= len(raw_sound):

        raw_bytes = raw_sound[time : time + sample_size * 2]
        converted_data = numpy.fromstring(raw_bytes, dtype = numpy.int16)

        windowed_data = numpy.multiply(converted_data, hann)
        fourier = numpy.fft.fft(windowed_data)

        # get only half of fourier coefficients
        fourier_normalized_converted = numpy.ndarray((1, rows))
        fourier_normalized_absolute = numpy.ndarray((1, rows))

        for i in range(rows):
            value = numpy.abs(fourier[(0, i)])
            fourier_normalized_absolute[(0, i)] = value

        #for i in range(rows):
        #    fourier_normalized_converted[(0, i)] = 20 * numpy.log10(fourier_normalized_absolute[(0, i)] + 1.0)

        for i in range(rows):
            fourier_normalized_converted[(0, i)] = numpy.sqrt(fourier_normalized_absolute[(0, i)] + 1.0)

        #for i in range(rows):
        #    fourier_normalized_converted[(0, i)] = fourier_normalized_absolute[(0, i)]

        # strip first and last frequences
        spectrogram[:, index] = fourier_normalized_converted[0, first_cut : rows - last_cut]

        time += time_shift * 2
        index += 1

    print('{}: columns = {}'.format(file_path, index))

    columns = max_spectrogram_length

    #erode(spectrogram, rows, columns)
    #dilate(spectrogram, rows, columns)

    print_statistics(spectrogram, file_path)
    show_spectrogram(spectrogram, file_path)

    #spectrogram = local_contrast_subtractive_transform(spectrogram, rows, columns)
    #print_statistics(spectrogram, file_path)

    mean = numpy.mean(spectrogram)
    std = numpy.std(spectrogram)

    spectrogram = spectrogram - mean
    spectrogram = spectrogram / std
    print_statistics(spectrogram, file_path)
    show_spectrogram(spectrogram, file_path)

    return spectrogram

def show_spectrogram(spectrogram, description):

    if show_image == False:
        return

    display_count = 1
    figure = matplotlib.pyplot.figure()

    subplot = matplotlib.pyplot.subplot(display_count, 1, 1)

    if description is not None:
        matplotlib.pyplot.title(description.replace('_', ' '), fontsize = 10)

    subplot.set_yticklabels([])
    subplot.set_xticklabels([])

    matplotlib.pyplot.imshow(spectrogram, cmap=matplotlib.pyplot.get_cmap('gray'))
    matplotlib.pyplot.show()

def prepare_dataset():

    print('[' + ctime() + ']: Train data preparation has started.')
    start_time = time_module.time()

    train_labels = []
    train_spectrograms = []

    test_ids = []
    test_spectrograms = []

    rec_id2filenames = []
    with open(rec_id2filename, 'r') as rec_id2_filename:
        with open(rec_id2test, 'r') as rec_id2_test:
            with open(rec_id2label, 'r') as rec_ic2_label:
                with open(rec_id2label_hidden, 'r') as rec_ic2_label_hidden:
                    rec_id2_filename_str = rec_id2_filename.readline()
                    rec_id2_test_str = rec_id2_test.readline()
                    rec_ic2_label_str = rec_ic2_label.readline()
                    rec_ic2_label_hidden_str = rec_ic2_label_hidden.readline()

                    while True:
                        rec_id2_filename_str = rec_id2_filename.readline()
                        rec_id2_test_str = rec_id2_test.readline()
                        rec_ic2_label_str = rec_ic2_label.readline()
                        rec_ic2_label_hidden_str = rec_ic2_label_hidden.readline()
                        if rec_id2_filename_str == '':
                            break

                        [rec_id, filename] = rec_id2_filename_str.split(',')
                        filename = filename.strip()
                        filepath = os.path.abspath(os.path.join(wav_path, filename + '.wav'))
                        if len(filter) > 0:
                            valid = False
                            for i in range(len(filter)):
                                if filepath.find(filter[i]) != -1:
                                    valid = True
                            if not valid:
                                continue

                        spectrogram = cook_spectrogram(filepath)

                        [rec_id_test, is_test] = rec_id2_test_str.split(',')
                        if rec_id_test != rec_id:
                            print('Record id mismatch.')

                        if int(is_test) == 1:
                            test_ids.append(rec_id)
                            test_spectrograms.append(spectrogram)
                            print('Test sample: {}, {}'.format(rec_id, filename))

                            if use_hidden:
                                classes = rec_ic2_label_hidden_str.split(',')
                                train_spectrograms.append(spectrogram)

                                label = numpy.zeros((1, output_classes))

                                for i in range(1, len(classes)):
                                    if parse_int(classes[i]):
                                        label[(0, int(classes[i]))] = 1.0

                                train_labels.append(label)

                                print('Test sample hidden: {}, {}, {}'.format(rec_id, filename, label))

                        else:
                            classes = rec_ic2_label_str.split(',')

                            train_spectrograms.append(spectrogram)
                            label = numpy.zeros((1, output_classes))

                            for i in range(1, len(classes)):
                                    if parse_int(classes[i]):
                                        label[(0, int(classes[i]))] = 1.0
                            train_labels.append(label)

                            print('Train sample: {}, {}, {}'.format(rec_id, filename, label))


    print('Train size: {} '.format(len(train_spectrograms)))
    print('Test size: {}'.format(len(test_spectrograms)))

    print('[' + ctime() + ']: Data preparation is complete.')
    end_time = time_module.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = {} minutes and {} seconds'.format(int(elapsed_time / 60), int(elapsed_time % 60)))
    print('Train data: {} Train labels: {} Test data: {} Test ids: {}'.format(len(train_spectrograms), len(train_labels), len(test_spectrograms), len(test_ids)))

    return (train_spectrograms, train_labels, test_spectrograms, test_ids)
