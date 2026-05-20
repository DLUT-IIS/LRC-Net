import argparse
import numpy as np
import random, os
import torch
import matplotlib.pyplot as plt
import pandas as pd 
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from tqdm import tqdm
from lrcnet.lutnn import LUTNN
from utils.mnist import load_mnist_dataset
from utils.cifar10 import load_cifar10_dataset
from utils.uci_datasets import AdultDataset, BreastCancerDataset
from utils.jsc import JetSubstructureDataset
from hdl.vhdl.convert2vhdl import get_model_params, gen_vhdl_code
from hdl.sv.convert2sv import gen_sv_code
from hdl.verilog.convert2verilog import gen_verilog_code

def get_args():
    parser = argparse.ArgumentParser(description='Train LUTNN.')

    parser.add_argument('--dataset', type=str, choices=['mnist', 'mnist20x20', 'cifar10-3', 'cifar10-31', 'adult', 'breast', 'jsc'], default="mnist20x20", help='Dataset to use')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature of the aggregation output')

    parser.add_argument('--num-iterations', '-ni', type=int, default=15_000, help='Number of training iterations (default: 15000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=1_000, help='Evaluation frequency (default: 1000)')
    parser.add_argument('--connections', type=str, default='unique', choices=['unique', 'GRC', 'LRC'])
    parser.add_argument('--init_type',   type=str, default='random', choices=['random', 'IMI'])
    parser.add_argument('--luts_per_layer', '-k', nargs='*', type=int, default=[2000], help='LUTs per layer (default: 2000)')
    parser.add_argument('--num_layers', '-l', type=int, default=2, help='Number of layers (default: 2)')
    parser.add_argument('--lut_size', '-s', nargs='*', type=int, default=[2], help='LUT input size (default: 2)')

    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--vhdl', action='store_true', help='Get VHDL code from net weights.')
    parser.add_argument('--sv', action='store_true', help='Get System Verilog code from net weights.')
    parser.add_argument('--v', action='store_true', help='Get Verilog code from net weights.')
    parser.add_argument('--save', action='store_true', help='Save model weights.')
    parser.add_argument('--train', action='store_true', help='Train model.')
    parser.add_argument('--load', action='store_true', help='Load model')
    parser.add_argument('--infer_time', action='store_true', help='Measure inference time')

    parser.add_argument('--paint_grad', action='store_true', help='Plot gradient analysis at the end')

    return parser.parse_args()


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


import matplotlib.pyplot as plt
import numpy as np

# Global academic style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "grid.alpha": 0.3,
    "mathtext.fontset": "stix" # 
})

def plot_iteration_gradient_trend(grad_track, png_dir):
    """Trend chart: Shows changes as training iterations progress."""
    w_names = [name for name in grad_track.keys() if name.endswith('.w')]
    w_names_sorted = sorted(w_names, key=lambda x: int(x.split('.')[1]))

    # Uniform size (8, 5)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # Color mapping
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(w_names_sorted)))

    for i, name in enumerate(w_names_sorted):
        layer_id = name.split('.')[1]
        data = grad_track[name]
        ax.plot(data, label=f'Layer {layer_id}', color=colors[i], 
                 alpha=0.85, linewidth=1.5)
    
    ax.set_yscale('log')
    ax.set_xlabel('Training Iterations', fontweight='bold')
    ax.set_ylabel('Gradient Norm (Log Scale)', fontweight='bold')
    
    # Legend settings: Display in 4 columns, placed at the top.
    ax.legend(loc='upper center', title="Hierarchy", title_fontsize=10, 
               frameon=True, framealpha=0.8, ncol=4, edgecolor='gray')

    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(png_dir, dpi=300, bbox_inches='tight')
    plt.show()


def plot_layer_gradient_mean_fixed(grad_track, png_dir):
    """Mean plot: Displays the average gradient of each layer."""
    w_names = [name for name in grad_track.keys() if name.endswith('.w')]
    w_names_sorted = sorted(w_names, key=lambda x: int(x.split('.')[1]))
    
    grad_means = [np.mean(grad_track[name]) for name in w_names_sorted]
    layers = [f'L{i+1}' for i in range(len(grad_means))]
    
    # Uniform size (8, 5)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # Draw a line chart
    ax.plot(layers, grad_means, marker='o', markersize=8, 
            linewidth=2, color='#1f77b4', markerfacecolor='white', 
            markeredgewidth=2)

    # Dynamic y-axis range
    y_min, y_max = min(grad_means), max(grad_means)
    ax.set_ylim(y_min * 0.8, y_max * 1.3) 

    # Label numerical values ​​(scientific notation)
    for i, val in enumerate(grad_means):
        ax.annotate(f'{val:.2e}', 
                    xy=(i, val), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color='#1f77b4',
                    fontweight='bold')

    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Mean Gradient Norm', fontweight='bold')
    
    # Unified Grid
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Force the y-axis to use scientific notation
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(png_dir, dpi=300, bbox_inches='tight')
    plt.show()


def train(model, train_loader, test_loader, args, device):
    # Initialize
    log_dir = getattr(args, 'log_dir', './logs') 
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f'{args.name}.csv')
    
    history = []
    grad_track = {}
    
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.8, cooldown=5, min_lr=0.0001)
    iterator = iter(train_loader)
    
    for i in tqdm(range(args.num_iterations), desc='Training', total=args.num_iterations):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        
        x = x.to(torch.float16).to(device).round()
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        class_loss = loss_fn(y_pred, y)
        scaler.scale(class_loss).backward()

        # Gradient Analysis Logic
        if getattr(args, 'paint_grad', False):
            scaler.unscale_(optimizer)
            for name, param in model.named_parameters():
                if param.grad is not None and ('.w' in name or 'weight' in name):
                    norm = param.grad.detach().data.norm(2).item()
                    if name not in grad_track:
                        grad_track[name] = []
                    grad_track[name].append(norm)
            
        scaler.step(optimizer)
        scaler.update()

        # Regularly Evaluate and Record Logs
        if (i + 1) % args.eval_freq == 0:
            train_acc = evaluation(model, train_loader, device, mode=False)
            test_acc_eval = evaluation(model, test_loader, device, mode=False)
            # test_acc_train = evaluation(model, test_loader, device, mode=True)
            current_lr = optimizer.param_groups[0]['lr']

            scheduler.step(test_acc_eval)
            
            log_entry = {
                'iteration': i + 1,
                'loss': f"{class_loss.item():.4f}",
                'train_acc': f"{train_acc:.4f}",
                'test_acc_eval': f"{test_acc_eval:.4f}",
                # 'test_acc_train': f"{test_acc_train:.4f}",
                'lr': f"{current_lr:.6f}"
            }
            history.append(log_entry)
            
            # Save in real-time as CSV (to prevent data loss in case of unexpected program interruption)
            df = pd.DataFrame(history)
            df.to_csv(csv_path, index=False)
            print(f'Iter {i+1}: LR={current_lr:.6f}, Loss={class_loss.item():.4f}, Test_Acc={test_acc_eval:.4f}')

    if getattr(args, 'paint_grad', False):
        grad_log_path = os.path.join(log_dir, 'gradient_track.json')
        with open(grad_log_path, 'w') as f:
            json.dump(grad_track, f)
        plot_iteration_gradient_trend(grad_track,f'./png/{args.name}_grad_norm.png')
        plot_layer_gradient_mean_fixed(grad_track,f'./png/{args.name}_mean_grad_norm.png')

    print(f"Training completed! Log has been saved to {log_dir}")
    return model



def evaluation(model, loader, device, mode=False):
    with torch.no_grad():
        model.train(mode=mode)
        result = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            result.append(((model(x).argmax(-1) == y).sum() / x.shape[0]).item())
        result = np.mean(result)
    model.train()
    return result.item()


def load_dataset(args):
    if "mnist20x20" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_mnist_dataset(args.batch_size, mnist20=True)
    elif "mnist" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_mnist_dataset(args.batch_size, mnist20=False)
    elif "adult" in args.dataset:
        train_set = AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
        input_dim_dataset = 116
        num_classes = 2
    elif "breast" in args.dataset:
        train_set = BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
        input_dim_dataset = 51
        num_classes = 2
    elif "cifar10-31" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_cifar10_dataset(args.batch_size, 31)
    elif "cifar10-3" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_cifar10_dataset(args.batch_size, 3)
    elif "jsc" in args.dataset:
        thresholds = 30
        train_set = JetSubstructureDataset("./data-jsc", thresholds=thresholds, split="train", download=True)
        test_set = JetSubstructureDataset("./data-jsc", thresholds=thresholds, split="test", download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        input_dim_dataset = 16*(thresholds-1)
        num_classes = 5
    return train_loader, test_loader, input_dim_dataset, num_classes


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seeds(args.seed)

    train_loader, test_loader, input_dim_dataset, num_classes = load_dataset(args)

    if args.load:
        # model = torch.load(f"models/{args.name}.pth")
        model = torch.load(f"models/{args.name}.pth", weights_only=False)
    else:
        model = LUTNN(args.luts_per_layer, args.num_layers, args.lut_size, input_dim_dataset, num_classes, args.tau,
                      device = device , dataset = args.dataset, connections = args.connections, init_type = args.init_type)
    
    if args.train:
        model = train(model, train_loader, test_loader, args, device)
    else:
        print(f"Inference accuracy: {evaluation(model, test_loader, device, False)}")
        if args.vhdl:
            number_of_layers, num_neurons, lut_size, number_of_inputs, number_of_classes = get_model_params(model)
            gen_vhdl_code(model, args.name, number_of_layers, number_of_classes, number_of_inputs, num_neurons,
                          lut_size)
        if args.sv:
            number_of_layers, num_neurons, lut_size, number_of_inputs, number_of_classes = get_model_params(model)
            gen_sv_code(model, args.name, number_of_layers, number_of_classes, number_of_inputs, num_neurons,
                          lut_size)
        if args.v:
            number_of_layers, num_neurons, lut_size, number_of_inputs, number_of_classes = get_model_params(model)
            gen_verilog_code(model, args.name, number_of_layers, number_of_classes, number_of_inputs, num_neurons,
                          lut_size)
                    
    if args.save:
        directory = "models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model, f"models/{args.name}.pth")
