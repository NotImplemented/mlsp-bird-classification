# mlsp-bird-classification

AUC-ROC = 0.72

NN-schema:
  contrast subtractive layer 13x13</br>
  max-pool 2x2, 
  conv 5x5 64 filters,
  max-pool 2x2,
  conv 5x5 128 filters,
  max-pool 2x2,
  conv 5x5 128 filters,
  max-pool 2x2,
  conv 5x5 256 filters,
  max-pool 2x2,
  conv 5x5 512 filters,
  max-pool 2x2,
  fully-connected 1024 nodes,
  output 19 nodes with sigmoid activation function

Weights initialization: 
  - xavier initialization for convolution filters
  - 0.06 for bias

Batch size = 8, epochs = 100

Spectrograms are build using 512 size sliding Hann window with 3/4 overlap and are normalized to zero mean and unit variance.
