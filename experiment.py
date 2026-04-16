# Divya - Task 5: Design Your Own Experiment
# Evaluates the effect of changing transformer network architecture along 3 dimensions
# using Fashion MNIST dataset with a linear search strategy (round-robin optimization)

# import statements
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from train_transformer import NetConfig, NetTransformer


# trains the model for given epochs and returns final metrics (runs on GPU)
def run_experiment(config, train_loader, test_loader, device):
    torch.manual_seed(config.seed)
    model = NetTransformer(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_test_acc = 0
    best_epoch = 0
    train_loss_final = 0
    train_acc_final = 0

    for epoch in range(1, config.epochs + 1):
        # training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
        train_loss_final = total_loss / total
        train_acc_final = 100.0 * correct / total

        # testing phase
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += data.size(0)
        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

    return {
        'train_loss': train_loss_final,
        'train_acc': train_acc_final,
        'test_acc': best_test_acc,
        'best_epoch': best_epoch,
    }


# loads Fashion MNIST dataset and returns train and test data loaders
def load_fashion_mnist(batch_size=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_set = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return train_loader, test_loader


# plots test accuracy vs parameter value for one dimension sweep
def plot_dimension_results(param_values, test_accs, dim_name, round_num, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, test_accs, 'b-o', linewidth=2, markersize=8)
    plt.xlabel(dim_name, fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'Round {round_num}: Test Accuracy vs {dim_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# generates a summary plot showing all three dimensions across all rounds
def plot_summary(all_results, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dim_names = ['Depth (Num Layers)', 'Num Attention Heads', 'Dropout Rate']
    colors = ['b', 'r', 'g']

    for dim_idx in range(3):
        ax = axes[dim_idx]
        for round_num in range(1, 3):
            key = f'round{round_num}_dim{dim_idx}'
            if key in all_results:
                data = all_results[key]
                ax.plot(data['params'], data['accs'],
                        f'{colors[round_num-1]}-o',
                        label=f'Round {round_num}',
                        linewidth=2, markersize=6)
        ax.set_xlabel(dim_names[dim_idx], fontsize=11)
        ax.set_ylabel('Test Accuracy (%)', fontsize=11)
        ax.set_title(f'Effect of {dim_names[dim_idx]}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment Results: Fashion MNIST Transformer', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# sweeps one dimension, returns list of test accuracies and the best parameter value
def sweep_dimension(dim_name, param_values, make_config_fn, train_loader,
                    test_loader, device, round_num, run_count, total_runs, all_rows):
    dim_accs = []
    for val in param_values:
        run_count += 1
        config = make_config_fn(val)
        start_time = time.time()
        results = run_experiment(config, train_loader, test_loader, device)
        elapsed = time.time() - start_time

        dim_accs.append(results['test_acc'])
        row = {
            'run': run_count,
            'round': round_num,
            'dimension': dim_name,
            'param_value': val,
            'depth': config.depth,
            'heads': config.num_heads,
            'dropout': config.dropout,
            'train_loss': round(results['train_loss'], 4),
            'train_acc': round(results['train_acc'], 2),
            'test_acc': round(results['test_acc'], 2),
            'best_epoch': results['best_epoch'],
            'time_sec': round(elapsed, 1),
        }
        all_rows.append(row)
        print(f"  Run {run_count}/{total_runs}: {dim_name}={val}, "
              f"test_acc={results['test_acc']:.2f}%, time={elapsed:.1f}s")

    best_idx = dim_accs.index(max(dim_accs))
    best_val = param_values[best_idx]
    print(f"  >> Best {dim_name}: {best_val} ({max(dim_accs):.2f}%)")
    return dim_accs, best_val, run_count


# main function - runs the full experiment with linear search across 3 dimensions
def main(argv):
    # ========================================================================
    # EXPERIMENT PLAN
    # ========================================================================
    # Dataset: Fashion MNIST (more challenging than MNIST digits)
    # Model: Vision Transformer (from Task 4)
    # Epochs per run: 3 (enough to see trends while keeping runs fast on GPU)
    #
    # Three dimensions to explore:
    #   Dim 1 - Depth (number of transformer layers): [1, 2, 3, 4]
    #   Dim 2 - Number of attention heads:            [1, 2, 4, 8]
    #   Dim 3 - Dropout rate:                         [0.0, 0.1, 0.2, 0.5]
    #
    # Strategy: Linear search (round-robin)
    #   - Hold 2 dims constant, vary the 3rd, pick best
    #   - Repeat for each dim, 2 full rounds
    #   - Total runs: 2 rounds x (4 + 4 + 4) = 24 runs
    #
    # HYPOTHESES:
    #   H1 (Depth): Increasing depth will improve accuracy up to a point
    #       (around 3-4 layers), then plateau or slightly decrease due to
    #       overfitting on this relatively simple task.
    #
    #   H2 (Heads): More attention heads should help up to a moderate number
    #       (4-6), but too many heads with small embed_dim (48) will hurt
    #       because each head gets very few dimensions to work with.
    #
    #   H3 (Dropout): A small amount of dropout (0.1-0.2) should improve
    #       generalization. Too much dropout (>0.3) will hurt accuracy by
    #       removing too much information during training.
    # ========================================================================

    # search space for each dimension (trimmed for speed)
    depth_values = [1, 2, 3, 4]
    heads_values = [1, 2, 4, 8]
    dropout_values = [0.0, 0.1, 0.2, 0.5]

    # starting best values (defaults from Task 4)
    best_depth = 2
    best_heads = 4
    best_dropout = 0.1

    # epochs per experiment run
    epochs_per_run = 2

    # set up device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # load Fashion MNIST once
    print("Loading Fashion MNIST dataset...")
    train_loader, test_loader = load_fashion_mnist(batch_size=128)

    # storage for results
    all_rows = []
    all_plot_data = {}
    run_count = 0
    total_runs = 2 * (len(depth_values) + len(heads_values) + len(dropout_values))

    print(f"\nTotal planned runs: {total_runs}")
    print("=" * 70)
    print("STARTING EXPERIMENT: 2 Rounds of Round-Robin Linear Search")
    print("=" * 70)

    experiment_start = time.time()

    # run 2 rounds of linear search (round-robin optimization)
    for round_num in range(1, 3):
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num}")
        print(f"Current best: depth={best_depth}, heads={best_heads}, dropout={best_dropout}")
        print(f"{'=' * 70}")

        # --- Dimension 1: Vary Depth ---
        print(f"\n--- Round {round_num}, Dim 1: Varying Depth ---")
        accs, best_depth, run_count = sweep_dimension(
            'depth', depth_values,
            lambda d: NetConfig(
                name=f'r{round_num}_depth{d}', dataset='fashion_mnist',
                depth=d, num_heads=best_heads, dropout=best_dropout,
                epochs=epochs_per_run, batch_size=128,
            ),
            train_loader, test_loader, device, round_num, run_count, total_runs, all_rows
        )
        all_plot_data[f'round{round_num}_dim0'] = {
            'params': depth_values[:], 'accs': accs[:]
        }
        plot_dimension_results(depth_values, accs, 'Depth (Num Layers)',
                               round_num, f'exp_round{round_num}_depth.png')

        # --- Dimension 2: Vary Heads ---
        print(f"\n--- Round {round_num}, Dim 2: Varying Heads ---")
        accs, best_heads, run_count = sweep_dimension(
            'heads', heads_values,
            lambda h: NetConfig(
                name=f'r{round_num}_heads{h}', dataset='fashion_mnist',
                depth=best_depth, num_heads=h, dropout=best_dropout,
                epochs=epochs_per_run, batch_size=128,
            ),
            train_loader, test_loader, device, round_num, run_count, total_runs, all_rows
        )
        all_plot_data[f'round{round_num}_dim1'] = {
            'params': heads_values[:], 'accs': accs[:]
        }
        plot_dimension_results(heads_values, accs, 'Num Attention Heads',
                               round_num, f'exp_round{round_num}_heads.png')

        # --- Dimension 3: Vary Dropout ---
        print(f"\n--- Round {round_num}, Dim 3: Varying Dropout ---")
        accs, best_dropout, run_count = sweep_dimension(
            'dropout', dropout_values,
            lambda dr: NetConfig(
                name=f'r{round_num}_drop{dr}', dataset='fashion_mnist',
                depth=best_depth, num_heads=best_heads, dropout=dr,
                epochs=epochs_per_run, batch_size=128,
            ),
            train_loader, test_loader, device, round_num, run_count, total_runs, all_rows
        )
        all_plot_data[f'round{round_num}_dim2'] = {
            'params': dropout_values[:], 'accs': accs[:]
        }
        plot_dimension_results(dropout_values, accs, 'Dropout Rate',
                               round_num, f'exp_round{round_num}_dropout.png')

    total_time = time.time() - experiment_start

    # save all results to CSV
    csv_filename = 'experiment_results.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'run', 'round', 'dimension', 'param_value', 'depth', 'heads',
            'dropout', 'train_loss', 'train_acc', 'test_acc', 'best_epoch', 'time_sec'
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to {csv_filename}")

    # generate summary plot
    plot_summary(all_plot_data, 'experiment_summary.png')
    print("Summary plot saved to experiment_summary.png")

    # print final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Total runs: {run_count}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Optimized parameters: depth={best_depth}, heads={best_heads}, dropout={best_dropout}")
    print("=" * 70)

    # find overall best run
    best_run = max(all_rows, key=lambda r: r['test_acc'])
    print(f"\nBest single run: Run #{best_run['run']}")
    print(f"  Config: depth={best_run['depth']}, heads={best_run['heads']}, "
          f"dropout={best_run['dropout']}")
    print(f"  Test Accuracy: {best_run['test_acc']:.2f}%")
    print(f"  Train Accuracy: {best_run['train_acc']:.2f}%")


if __name__ == "__main__":
    main(sys.argv)
