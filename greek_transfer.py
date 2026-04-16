# Divya - Transfer Learning: Re-use MNIST network to recognize Greek letters

# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from train_mnist import MyNetwork


# greek data set transform - converts RGB to grayscale, scales, crops, and inverts
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# trains the network for one epoch on the greek dataset
def train_greek(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# evaluates the network on the greek training set
def test_greek(model, train_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


# plots training error over epochs
def plot_training_error(losses, epochs):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, 'b-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Greek Letters Transfer Learning - Training Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('greek_training_error.png')
    plt.show()


# loads and preprocesses a custom greek letter image for classification
def load_custom_greek_image(filepath):
    img = Image.open(filepath).convert('RGB')
    # resize to ~128x128 to match the greek letter dataset
    img = img.resize((128, 128), Image.LANCZOS)
    # apply same transforms as training data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img)
    return img_tensor


# tests the network on custom greek letter images
def test_custom_greek(model, image_dir):
    model.eval()
    if not os.path.exists(image_dir):
        print(f'\nCustom greek image directory "{image_dir}" not found. Skipping.')
        print('Create a folder called "custom_greek" with subfolders alpha/, beta/, gamma/')
        return

    label_names = ['alpha', 'beta', 'gamma']
    images = []
    true_labels = []
    predictions = []

    for label_idx, name in enumerate(label_names):
        folder = os.path.join(image_dir, name)
        if not os.path.exists(folder):
            print(f'Warning: folder {folder} not found')
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                filepath = os.path.join(folder, fname)
                img_tensor = load_custom_greek_image(filepath)
                with torch.no_grad():
                    output = model(img_tensor.unsqueeze(0))
                    pred = output.argmax(dim=1).item()
                images.append(img_tensor)
                true_labels.append(label_idx)
                predictions.append(pred)

    if not images:
        print('No custom greek images found.')
        return

    # print results
    print('\n--- Custom Greek Letter Results ---')
    print(f'{"True Label":<12} {"Predicted":<12} {"Correct":<8}')
    print('-' * 32)
    correct = 0
    for true_l, pred_l in zip(true_labels, predictions):
        true_name = label_names[true_l]
        pred_name = label_names[pred_l]
        is_correct = 'Yes' if true_l == pred_l else 'No'
        if true_l == pred_l:
            correct += 1
        print(f'{true_name:<12} {pred_name:<12} {is_correct:<8}')
    print(f'\nAccuracy: {correct}/{len(images)} ({100.0 * correct / len(images):.1f}%)')

    # plot results
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        if i < n:
            # denormalize for display
            img_display = images[i].squeeze().numpy()
            img_display = img_display * 0.3081 + 0.1307
            axes[i].imshow(img_display, cmap='gray')
            true_name = label_names[true_labels[i]]
            pred_name = label_names[predictions[i]]
            color = 'green' if true_labels[i] == predictions[i] else 'red'
            axes[i].set_title(f'True: {true_name}\nPred: {pred_name}', color=color)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle('Custom Greek Letter Recognition')
    plt.tight_layout()
    plt.savefig('greek_custom_results.png')
    plt.show()


# main function - loads pretrained MNIST model, modifies for greek letters, trains
def main(argv):
    # set random seed
    torch.manual_seed(42)

    # (1) generate the MNIST network
    model = MyNetwork()

    # (2) load pre-trained weights
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    print('Loaded pre-trained MNIST model:')
    print(model)

    # (3) freeze all network weights
    for param in model.parameters():
        param.requires_grad = False

    # (4) replace the last layer with a new Linear layer with 3 nodes
    model.fc2 = nn.Linear(50, 3)
    print('\nModified network for Greek letters (3 classes):')
    print(model)

    # set up the greek letter data loader
    training_set_path = 'greek_letters'

    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )

    # optimizer - only train the new last layer
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)

    # train until near-perfect accuracy
    n_epochs = 100
    losses = []
    print('\nTraining...')
    for epoch in range(1, n_epochs + 1):
        loss, acc = train_greek(model, greek_train, optimizer, epoch)
        losses.append(loss)
        if epoch % 10 == 0 or acc == 100.0:
            print(f'Epoch {epoch}: Loss: {loss:.4f}, Accuracy: {acc:.1f}%')
        if acc == 100.0:
            print(f'\nPerfect accuracy reached at epoch {epoch}!')
            break

    # final accuracy check
    final_acc = test_greek(model, greek_train)
    print(f'Final training accuracy: {final_acc:.1f}%')

    # plot training error
    plot_training_error(losses, len(losses))

    # save the greek model
    torch.save(model.state_dict(), 'greek_model.pth')
    print('Greek model saved to greek_model.pth')

    # test on custom greek letter images
    custom_dir = 'custom_greek'
    if len(argv) > 1:
        custom_dir = argv[1]
    test_custom_greek(model, custom_dir)


if __name__ == "__main__":
    main(sys.argv)
