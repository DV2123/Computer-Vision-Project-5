# Divya - Extension 1: Pre-trained Network Layer Analysis
# Loads a pre-trained VGG16 network and visualizes its first two
# convolutional layers, then applies filters to a sample MNIST image

# import statements
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import cv2


# prints the structure of the pre-trained model (first few layers)
def print_model_structure(model):
    print("VGG16 Feature Layers:")
    print("=" * 60)
    for i, layer in enumerate(model.features):
        print(f"  Layer {i}: {layer}")
        if i > 10:
            print("  ...")
            break
    print("=" * 60)


# extracts and visualizes filters from a given convolutional layer
def visualize_filters(weights, layer_name, filename):
    num_filters = weights.shape[0]
    num_channels = weights.shape[1]

    # determine grid size
    cols = 8
    rows = (num_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.8))
    fig.suptitle(f'{layer_name} Filters ({num_filters} filters, '
                 f'{weights.shape[2]}x{weights.shape[3]}, '
                 f'{num_channels} channels)', fontsize=14)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < num_filters:
            # for multi-channel filters, average across channels for display
            if num_channels == 3:
                # show as RGB image (normalize to 0-1)
                filt = weights[i].permute(1, 2, 0).numpy()
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                ax.imshow(filt)
            else:
                # single channel or averaged
                filt = weights[i].mean(dim=0).numpy()
                ax.imshow(filt, cmap='viridis')
            ax.set_title(f'F{i}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {layer_name} filter visualization to {filename}")


# applies conv1 filters to a sample image and shows the results
def apply_filters_to_image(weights, image, layer_name, filename):
    num_filters = weights.shape[0]
    cols = 8
    rows = (num_filters + cols - 1) // cols + 1  # extra row for original

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.8))
    fig.suptitle(f'{layer_name} Filter Results on Sample Image', fontsize=14)

    # show original image in first position
    axes[0][0].imshow(image, cmap='gray')
    axes[0][0].set_title('Original', fontsize=8)
    for j in range(1, cols):
        axes[0][j].axis('off')

    # apply each filter and show result
    for i in range(num_filters):
        row = (i // cols) + 1
        col = i % cols
        ax = axes[row][col]

        # average filter across input channels to get a 2D kernel
        kernel = weights[i].mean(dim=0).numpy()

        # apply filter using OpenCV
        filtered = cv2.filter2D(image, -1, kernel)
        ax.imshow(filtered, cmap='gray')
        ax.set_title(f'F{i}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused axes
    for i in range(num_filters, (rows - 1) * cols):
        row = (i // cols) + 1
        col = i % cols
        if row < rows:
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {layer_name} filter results to {filename}")


# loads MNIST sample image for filter application
def load_sample_image():
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    image, label = test_set[0]
    print(f"Sample image label: {label}")
    return image.squeeze().numpy()


# compares MNIST CNN filters with VGG16 filters side by side
def compare_with_mnist(vgg_weights, mnist_model_path, filename):
    from train_mnist import MyNetwork

    # load MNIST model
    mnist_model = MyNetwork()
    mnist_model.load_state_dict(torch.load(mnist_model_path, map_location='cpu',
                                           weights_only=True))
    mnist_weights = mnist_model.conv1.weight.data

    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    fig.suptitle('Filter Comparison: MNIST CNN (top) vs VGG16 (bottom)', fontsize=14)

    # MNIST conv1 filters (10 filters, 1 channel, 5x5)
    for i in range(10):
        ax = axes[0][i]
        ax.imshow(mnist_weights[i, 0].numpy(), cmap='viridis')
        ax.set_title(f'MNIST F{i}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # VGG16 conv1 filters (first 10, shown as RGB)
    for i in range(10):
        ax = axes[1][i]
        filt = vgg_weights[i].permute(1, 2, 0).numpy()
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        ax.imshow(filt)
        ax.set_title(f'VGG F{i}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved filter comparison to {filename}")


# main function - loads VGG16, visualizes layers, applies filters
def main(argv):
    print("Loading pre-trained VGG16 model...")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()

    # print model structure
    print_model_structure(model)

    # get first two convolutional layers
    # VGG16 structure: features[0] = Conv2d(3, 64, 3, padding=1)
    #                  features[2] = Conv2d(64, 128, 3, padding=1)
    conv1 = model.features[0]
    conv2 = model.features[2]

    print(f"\nConv1: {conv1}")
    print(f"Conv1 weight shape: {conv1.weight.shape}")
    print(f"\nConv2: {conv2}")
    print(f"Conv2 weight shape: {conv2.weight.shape}")

    # extract weights (no gradient tracking needed)
    with torch.no_grad():
        conv1_weights = conv1.weight.data.cpu()
        conv2_weights = conv2.weight.data.cpu()

    # visualize conv1 filters (64 filters, 3 channels, 3x3)
    print("\nVisualizing Conv1 filters...")
    visualize_filters(conv1_weights, 'VGG16 Conv1', 'ext_vgg16_conv1_filters.png')

    # visualize conv2 filters (64 filters, 64 channels, 3x3)
    print("Visualizing Conv2 filters...")
    visualize_filters(conv2_weights, 'VGG16 Conv2', 'ext_vgg16_conv2_filters.png')

    # load sample MNIST image and apply conv1 filters
    print("\nApplying Conv1 filters to MNIST sample...")
    sample_image = load_sample_image()
    apply_filters_to_image(conv1_weights, sample_image, 'VGG16 Conv1',
                           'ext_vgg16_conv1_results.png')

    # compare MNIST CNN filters with VGG16 filters
    print("\nComparing MNIST CNN filters with VGG16 filters...")
    compare_with_mnist(conv1_weights, 'mnist_model.pth',
                       'ext_filter_comparison.png')

    print("\nDone! All visualizations saved.")


if __name__ == "__main__":
    main(sys.argv)
