# Divya - MNIST Digit Recognition: Build, Train, and Save a CNN

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


# CNN network model for MNIST digit recognition
class MyNetwork(nn.Module):
    # initializes the network layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        # convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # dropout layer with 0.5 dropout rate
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # fully connected layer with 50 nodes
        self.fc1 = nn.Linear(320, 50)
        # final fully connected layer with 10 nodes (one per digit)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass through the network
    def forward(self, x):
        # conv1 -> max pool 2x2 -> relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 -> dropout -> max pool 2x2 -> relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten
        x = x.view(-1, 320)
        # fully connected layer with relu
        x = F.relu(self.fc1(x))
        # final layer with log softmax
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# plots the first six example digits from the test set
def plot_first_six(test_set):
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axes.flatten()):
        image, label = test_set[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('First 6 Test Set Digits')
    plt.tight_layout()
    plt.savefig('plot_first_six.png')
    plt.show()


# trains the network for one epoch and returns the average loss
def train_network(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
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
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


# evaluates the network on a data loader and returns loss and accuracy
def test_network(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(f'         Test Loss:  {avg_loss:.4f}, Test Accuracy:  {accuracy:.2f}%')
    return avg_loss, accuracy


# plots training and testing loss and accuracy over epochs
def plot_training_results(train_losses, test_losses, train_accs, test_accs, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, epochs + 1), train_losses, 'b-o', label='Training Loss')
    ax1.plot(range(1, epochs + 1), test_losses, 'r-o', label='Testing Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Testing Error')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), train_accs, 'b-o', label='Training Accuracy')
    ax2.plot(range(1, epochs + 1), test_accs, 'r-o', label='Testing Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Testing Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('plot_training_results.png')
    plt.show()


# main function - loads data, builds network, trains, and saves
def main(argv):
    # hyperparameters
    n_epochs = 5
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5

    # set random seed for reproducibility
    torch.manual_seed(42)

    # load MNIST training and test datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    # plot first six test digits (use un-normalized data for display)
    display_test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    plot_first_six(display_test_set)

    # build network
    model = MyNetwork()
    print(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # train and evaluate for each epoch
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_network(model, train_loader, optimizer, epoch)
        test_loss, test_acc = test_network(model, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # plot training results
    plot_training_results(train_losses, test_losses, train_accs, test_accs, n_epochs)

    # save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')


if __name__ == "__main__":
    main(sys.argv)
