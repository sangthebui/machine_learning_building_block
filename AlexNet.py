import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, LocalResponseNorm, Dropout, Linear




class AlexNet(Module):
    """Pytorch implementation of AlexNet
    ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
    (dataset)
    trained: 1.2milllion images (1000 classes)
    256x256x3
    50,000 validations
    150,000 testing
    (model)
    60 millinos parameters
    650,000 neurons
    5 convolution layers
    3 fully connect layers
    1000-way softmax
    Local Reponse Normalization (LRN)

    """
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # Part 1: Feature Extractor (Convolutional base)
        self.features = Sequential(
            # Layer 1: Convolution -> ReLU -> Max Pooling
            # Input: 3x224x224
            # Paper specifies 96 kernels of size 11x11 with stride 4.
            # Output size: (224 - 11 + 2*2) / 4 + 1 = 55.25 -> we'll stick to standard implementation
            # which results in a 55x55 feature map.
            # (In PyTorch's torchvision, the first conv layer is Conv2d(3, 64, ...), but the original paper uses 96 filters)
            Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            # Paper specifies overlapping pooling. Here, kernel_size > stride.
            LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            MaxPool2d(kernel_size=3, stride=2),

            # Layer 2: Convolution -> ReLU -> Max Pooling
            # Input: 96x27x27 (from (55-3)/2+1 = 27)
            # Paper specifies 256 kernels of size 5x5 with padding 2.
            Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            ReLU(inplace=True),
            LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            MaxPool2d(kernel_size=3, stride=2),

            # Layer 3: Convolution -> ReLU
            # Input: 256x13x13 (from (27-3)/2+1 = 13)
            # Paper specifies 384 kernels of size 3x3 with padding 1.
            Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            ReLU(inplace=True),

            # Layer 4: Convolution -> ReLU
            # Input: 384x13x13
            # Paper specifies 384 kernels of size 3x3 with padding 1.
            Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            ReLU(inplace=True),

            # Layer 5: Convolution -> ReLU -> Max Pooling
            # Input: 384x13x13
            # Paper specifies 256 kernels of size 3x3 with padding 1.
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )

        # This layer is used to adapt the pooling output to the classifier input.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Part 2: Classifier (Fully-connected layers)
        self.classifier = Sequential(
            # Dropout is used to prevent overfitting, as mentioned in the paper.
            # p=0.5 is the probability of an element to be zeroed.
            Dropout(p=0.5),
            # Layer 6: Fully-connected -> ReLU
            # Input features: 256 channels * 6 * 6 grid size = 9216
            Linear(in_features=(256 * 6 * 6), out_features=4096),
            ReLU(inplace=True),

            Dropout(p=0.5),
            # Layer 7: Fully-connected -> ReLU
            Linear(in_features=4096, out_features=4096),
            ReLU(inplace=True),

            # Layer 8: Output layer (Fully-connected)
            Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 224, 224).
        Returns:
            torch.Tensor: The output tensor (logits) of shape (batch_size, num_classes).
        """
        # Pass input through the feature extractor
        x = self.features(x)
        x = self.avgpool(x)

        # Flatten the output from the convolutional part to feed into the classifier
        # The shape becomes (batch_size, 256*6*6)
        x = torch.flatten(x, 1)

        # Pass the flattened tensor through the classifier
        logits = self.classifier(x)
        return logits

