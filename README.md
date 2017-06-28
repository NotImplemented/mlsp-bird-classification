# mlsp-bird-classification

AUC-ROC = 0.72

NN-schema:</br>
  </t>contrast subtractive layer 13x13</br>
  max-pool 2x2</br>
  conv 5x5 64 filters</br>
  max-pool 2x2</br>
  conv 5x5 128 filters</br>
  max-pool 2x2</br>
  conv 5x5 128 filters</br>
  max-pool 2x2</br>
  conv 5x5 256 filters</br>
  max-pool 2x2</br>
  conv 5x5 512 filters</br>
  max-pool 2x2</br>
  fully-connected 1024 nodes</br>
  output 19 nodes with sigmoid activation function</br>

Weights initialization: 
  - xavier initialization for convolution filters
  - 0.06 for bias

Batch size = 8, epochs = 100</br>
Spectrograms are build using 512 size sliding Hann window with 3/4 overlap and are normalized to zero mean and unit variance.
