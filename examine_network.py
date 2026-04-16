# Divya - Examine the trained MNIST network: analyze filters and their effects

# import statements
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from train_mnist import MyNetwork


# prints the model structure and layer names
def print_model(model):
    print('Model Structure:')
    print(model)
    print()


# extracts and prints the first layer (conv1) filter weights and shape
def analyze_first_layer(model):
    weights = model.conv1.weight
    print(f'Conv1 weight shape: {weights.shape}')
    print(f'  (10 filters, 1 input channel, 5x5 filter size)\n')
    for i in range(weights.shape[0]):
        print(f'Filter {i}:')
        print(weights[i, 0])
        print()
    return weights


# visualizes the 10 conv1 filters in a 3x4 grid using pyplot
def plot_filters(weights):
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    with torch.no_grad():
        for i in range(10):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            ax.imshow(weights[i, 0].numpy(), cmap='gray')
            ax.set_title(f'Filter {i}')
            ax.set_xticks([])
            ax.set_yticks([])
        # hide the two unused subplots in the 3x4 grid
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
    plt.suptitle('Conv1 Filters (10 x 5x5)')
    plt.tight_layout()
    plt.savefig('conv1_filters.png')
    plt.show()


# applies each of the 10 filters to the first training image using OpenCV filter2D
# shows filters and their effects side by side in a 5x4 grid
def show_filter_effects(weights, train_set):
    # get the first training image
    image, label = train_set[0]
    image_np = image.squeeze().numpy()

    # 5 rows x 4 columns: each row has [filter, result, filter, result]
    fig, axes = plt.subplots(5, 4, figsize=(8, 10))

    with torch.no_grad():
        for i in range(10):
            row = i // 2
            col = (i % 2) * 2  # 0 or 2
            # show the filter
            kernel = weights[i, 0].numpy()
            axes[row, col].imshow(kernel, cmap='gray')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            # show the filtered image
            filtered = cv2.filter2D(image_np, -1, kernel)
            axes[row, col + 1].imshow(filtered, cmap='gray')
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])

    plt.suptitle(f'Conv1 Filters and Their Effects on First Training Image (digit: {label})')
    plt.tight_layout()
    plt.savefig('conv1_filterResults.png')
    plt.show()


# main function - loads model and examines the first layer
def main(argv):
    # load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    model.eval()

    # print model structure
    print_model(model)

    # analyze first layer weights
    weights = analyze_first_layer(model)

    # visualize the 10 filters
    plot_filters(weights)

    # load training data (un-normalized for display)
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # show the effect of filters on the first training image
    show_filter_effects(weights, train_set)


if __name__ == "__main__":
    main(sys.argv)
