# absolutely_detailed_mnist_classifier_implementation
An absolutely detailed classifier neural network built for th torchvision MNIST dataset. Contains comprehensive documentation.

<img src="./asset.png" alt="Code flow" width="620"/>

## MNIST Linear Neural Network
A minimal PyTorch implementation of a 3-layer fully connected neural network to classify handwritten digits from the MNIST dataset.

## Architecture
- Input Layer: 784 units (28x28 flattened image pixels)
- Hidden Layer 1: 196 units with ReLU activation
- Hidden Layer 2: 49 units with ReLU activation
- Output Layer: 10 units (class logits for digits 0-9)

## Pipeline Flow
1. Data Ingestion: Downloads MNIST; normalizes images to PyTorch tensors.
2. Batching: DataLoader splits data into mini-batches of 150 and shuffles training data.
3. Forward Pass: Flattens 2D images to 1D vectors, passes them through linear layers and ReLU functions.
4. Loss Computation: Calculates Cross-Entropy loss against ground truth labels.
5. Optimization: Adam optimizer computes gradients via backpropagation and updates weights over 15 epochs.
6. Evaluation: Disables gradient tracking to compute training and testing accuracy.
