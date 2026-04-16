# Divya - MNIST Digit Recognition: Load model and test on test set + custom images

# import statements
import sys
import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from train_mnist import MyNetwork


# runs the model on the first 10 test examples and prints results
def test_first_ten(model, test_loader):
    model.eval()
    test_set = test_loader.dataset
    print('\n--- First 10 Test Examples ---')
    print(f'{"Index":<6} {"Output Values":<65} {"Predicted":<10} {"Label":<6}')
    print('-' * 90)

    with torch.no_grad():
        for i in range(10):
            image, label = test_set[i]
            output = model(image.unsqueeze(0))
            values = output.squeeze().tolist()
            pred = output.argmax(dim=1).item()
            values_str = ' '.join(f'{v:6.2f}' for v in values)
            print(f'{i:<6} [{values_str}]  {pred:<10} {label:<6}')


# plots the first 9 test digits in a 3x3 grid with predictions
def plot_predictions(model, display_test_set, norm_test_set):
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    with torch.no_grad():
        for i, ax in enumerate(axes.flatten()):
            # use normalized data for prediction
            image_norm, label = norm_test_set[i]
            output = model(image_norm.unsqueeze(0))
            pred = output.argmax(dim=1).item()

            # use un-normalized data for display
            image_disp, _ = display_test_set[i]
            ax.imshow(image_disp.squeeze(), cmap='gray')
            ax.set_title(f'Pred: {pred}')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('First 9 Test Digits with Predictions')
    plt.tight_layout()
    plt.savefig('plot_predictions.png')
    plt.show()


# loads and preprocesses a custom handwritten digit image to match MNIST format
def load_custom_image(filepath):
    # read image and convert to greyscale
    img = Image.open(filepath).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # MNIST digits are white on black, so invert (photos are dark on light)
    img_array = 255.0 - img_array

    # apply threshold to remove background noise and make digit cleaner
    # background after inversion is typically below 130, digit strokes are above
    threshold = 130
    img_array[img_array < threshold] = 0

    # crop to bounding box of the digit with some padding
    coords = np.argwhere(img_array > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # add padding around the digit
        pad = 20
        y_min = max(0, y_min - pad)
        y_max = min(img_array.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(img_array.shape[1], x_max + pad)
        img_array = img_array[y_min:y_max+1, x_min:x_max+1]

    # make the image square by padding the shorter side
    h, w = img_array.shape
    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        img_array = np.pad(img_array, ((0, 0), (left, right)), mode='constant')
    elif w > h:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        img_array = np.pad(img_array, ((top, bottom), (0, 0)), mode='constant')

    # resize to 20x20 (MNIST digits are 20x20 centered in 28x28)
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    img_pil = img_pil.resize((20, 20), Image.LANCZOS)
    img_array = np.array(img_pil, dtype=np.float32)

    # center in a 28x28 image with 4px padding on each side (like MNIST)
    padded = np.zeros((28, 28), dtype=np.float32)
    padded[4:24, 4:24] = img_array

    # normalize to [0, 1] then apply MNIST normalization
    img_tensor = torch.tensor(padded / 255.0).unsqueeze(0)
    img_normalized = torchvision.transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
    return img_normalized, padded


# tests the network on custom handwritten digit images
def test_custom_digits(model, image_dir):
    model.eval()
    if not os.path.exists(image_dir):
        print(f'\nCustom image directory "{image_dir}" not found. Skipping custom digit test.')
        print('Create a folder called "custom_digits" with images named 0.png, 1.png, ..., 9.png')
        return

    images = []
    labels = []
    predictions = []
    raw_images = []

    for digit in range(10):
        # try common image extensions
        found = False
        for ext in ['png', 'jpg', 'jpeg', 'bmp']:
            filepath = os.path.join(image_dir, f'{digit}.{ext}')
            if os.path.exists(filepath):
                img_normalized, img_raw = load_custom_image(filepath)
                with torch.no_grad():
                    output = model(img_normalized.unsqueeze(0))
                    pred = output.argmax(dim=1).item()
                images.append(img_normalized)
                raw_images.append(img_raw)
                labels.append(digit)
                predictions.append(pred)
                found = True
                break
        if not found:
            print(f'Warning: No image found for digit {digit}')

    if not images:
        print('No custom images found.')
        return

    # print results
    print('\n--- Custom Handwritten Digit Results ---')
    print(f'{"True Label":<12} {"Predicted":<12} {"Correct":<8}')
    print('-' * 32)
    correct = 0
    for label, pred in zip(labels, predictions):
        is_correct = 'Yes' if label == pred else 'No'
        if label == pred:
            correct += 1
        print(f'{label:<12} {pred:<12} {is_correct:<8}')
    print(f'\nAccuracy: {correct}/{len(labels)} ({100.0 * correct / len(labels):.1f}%)')

    # plot results
    n = len(raw_images)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(raw_images[i], cmap='gray')
            color = 'green' if labels[i] == predictions[i] else 'red'
            axes[i].set_title(f'True: {labels[i]}, Pred: {predictions[i]}', color=color)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle('Custom Handwritten Digit Recognition')
    plt.tight_layout()
    plt.savefig('plot_custom_digits.png')
    plt.show()


# main function - loads trained model and runs tests
def main(argv):
    # load the trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    model.eval()
    print('Model loaded from mnist_model.pth')
    print(model)

    # load test data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    norm_test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    display_test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        norm_test_set, batch_size=64, shuffle=False
    )

    # test on first 10 examples
    test_first_ten(model, test_loader)

    # plot first 9 predictions
    plot_predictions(model, display_test_set, norm_test_set)

    # test on custom handwritten digits
    custom_dir = 'custom_digits'
    if len(argv) > 1:
        custom_dir = argv[1]
    test_custom_digits(model, custom_dir)


if __name__ == "__main__":
    main(sys.argv)
