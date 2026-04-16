# Divya - MNIST Digit Recognition using Transformer Network

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


class NetConfig:

    def __init__(self,
                 name = 'vit_base',
                 dataset = 'mnist',
                 patch_size = 4,
                 stride = 2,
                 embed_dim = 48,
                 depth = 4,
                 num_heads = 8,
                 mlp_dim = 128,
                 dropout = 0.1,
                 use_cls_token = False,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 weight_decay = 1e-4,
                 seed = 0,
                 optimizer = 'adamw',
                 device = 'cpu',
                 ):

        # data set fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # variable things
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name,
            self.dataset,
            self.patch_size,
            self.stride,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_dim,
            self.dropout,
            self.use_cls_token,
            self.epochs,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.seed,
            self.optimizer
            )
        self.config_string = s

        return


# Patch Embedding class - splits image into patches and projects to embedding space
class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Input:
        x of shape (B, C, H, W)

    Output:
        tokens of shape (B, N, D)

    where:
        B = batch size
        N = number of patches (tokens)
        D = embedding dimension
    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            stride: int,
            in_channels: int,
            embed_dim: int,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # - non-overlapping patches  (stride == patch_size)
        # - overlapping patches      (stride < patch_size)
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )

        # Each extracted patch is flattened into one vector
        self.patch_dim = in_channels * patch_size * patch_size

        # After flattening a patch, project it into embedding space.
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute how many patches will be produced for this image setup
        self.num_patches = self._compute_num_patches()

    # computes how many patches are extracted in total
    def _compute_num_patches(self) -> int:
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    # extracts patches and converts them to embeddings
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: extract patches using nn.Unfold, shape becomes (B, patch_dim, N)
        x = self.unfold(x)

        # Step 2: move dimensions so each patch becomes one row/token.
        # Shape becomes: (B, N, patch_dim)
        x = x.transpose(1, 2)

        # Step 3: project each flattened patch into embedding space.
        # Shape becomes: (B, N, embed_dim)
        x = self.proj(x)

        return x


# Transformer Network class for MNIST digit recognition
class NetTransformer(nn.Module):

    # initializes the transformer network layers
    def __init__(self, config):
        super(NetTransformer, self).__init__()

        # make the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # how many tokens are there?
        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens))

        # does it use a classifier token or a global average token?
        self.use_cls_token = config.use_cls_token

        # if it uses a classifier node, create a source for the node
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        # need to include a learned positional embedding, one for each token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer Encoder Layer with multi-head self attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        # Create a stack of transformer layers to build an encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        # final normalization layer prior to classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )

        return

    # initialize special parameters
    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # executes a forward pass through the transformer network
    def forward(self, x):
        # execute the patch embedding layer
        x = self.patch_embed(x)

        # get the batch size (0 dimension of x)
        batch_size = x.size(0)

        # add the optional CLS token to the set
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # add the learnable positional embedding to each token
        x = x + self.pos_embed

        # run the dropout layer right after the patch embedding
        x = self.pos_dropout(x)

        # run the transformer encoder
        x = self.encoder(x)

        # either pool the tokens or use the cls token (first token)
        if self.use_cls_token:
            x = x[:, 0]  # classify based on the cls token
        else:
            x = x.mean(dim=1)  # classify using the mean token vector

        # final normalization of the token to classify
        x = self.norm(x)

        # call the classification MLP
        x = self.classifier(x)

        # return the softmax of the output layer
        return F.log_softmax(x, dim=1)


# trains the transformer for one epoch
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
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%', flush=True)
    return avg_loss, accuracy


# evaluates the transformer on a data loader
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
    print(f'         Test Loss:  {avg_loss:.4f}, Test Accuracy:  {accuracy:.2f}%', flush=True)
    return avg_loss, accuracy


# plots training and testing loss and accuracy over epochs
def plot_training_results(train_losses, test_losses, train_accs, test_accs, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, epochs + 1), train_losses, 'b-o', label='Training Loss')
    ax1.plot(range(1, epochs + 1), test_losses, 'r-o', label='Testing Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Transformer - Training and Testing Error')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), train_accs, 'b-o', label='Training Accuracy')
    ax2.plot(range(1, epochs + 1), test_accs, 'r-o', label='Testing Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Transformer - Training and Testing Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('transformer_training_results.png')
    plt.show()


# main function - builds transformer, trains on MNIST, saves model
def main(argv):
    # create default config
    config = NetConfig()
    print('Config:')
    print(config.config_string)

    # set random seed
    torch.manual_seed(config.seed)

    # load MNIST datasets
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
        train_set, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False
    )

    # build transformer model
    model = NetTransformer(config)
    print('\nModel:')
    print(model)

    # optimizer
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # train and evaluate for each epoch
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_network(model, train_loader, optimizer, epoch)
        test_loss, test_acc = test_network(model, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # plot results
    plot_training_results(train_losses, test_losses, train_accs, test_accs, config.epochs)

    # save model
    torch.save(model.state_dict(), 'transformer_model.pth')
    print('Transformer model saved to transformer_model.pth')


if __name__ == "__main__":
    main(sys.argv)
