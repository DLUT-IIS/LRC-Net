import torch
import torch.nn.functional as F
import string

def bin2dec(b, bits):
    """
    Convert binary tensor to decimal.
    :param b: Binary tensor
    :param bits: Number of bits
    :return: Decimal tensor
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=b.device, dtype=b.dtype)
    return torch.sum(mask * b, dim=-1)

def sample_gumbel(shape, eps=1e-20, device='cuda'):
    """
    Sample from Gumbel distribution.
    :param shape: Shape of the output tensor
    :param eps: Small value to avoid log(0)
    :param device: Device to place the tensor
    :return: Gumbel samples
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """
    Sample from Gumbel-Softmax distribution.
    :param logits: Logits tensor
    :param temperature: Temperature parameter
    :return: Gumbel-Softmax sample
    """
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    Apply Gumbel-Softmax with optional hard sampling.
    :param logits: Logits tensor
    :param temperature: Temperature parameter
    :param hard: Whether to return a one-hot vector
    :return: Gumbel-Softmax result
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y
    return y

class LUTLayer(torch.nn.Module):
    """
    LUT Layer.
    """
    def __init__(self, input_dim, lut_size=2, n_luts=1, radius = 2, 
                 device='cpu', dataset='mnist', connections='GRC',  init_type='random'):
        super().__init__()
        """
        :param input_dim: Number of input dimensions
        :param lut_size: Number of inputs per LUT
        :param n_luts: Number of LUTs
        :param connections: Method for initializing connectivity
        :param device: Device to place the tensors
        """
        self.n_luts = n_luts
        self.lut_size = lut_size
        self.w = torch.nn.Parameter(torch.empty(n_luts, 2 ** lut_size, device=device))
        self.w_comp = torch.nn.Parameter(torch.empty(n_luts, 2 ** lut_size, device=device))
        self.apply_initialization(init_type)
        self.indices = self.get_connections(input_dim, lut_size, n_luts, radius, device, dataset, connections)

    def apply_initialization(self, init_type):
        if init_type == 'random':
            torch.nn.init.normal_(self.w, mean=0.0, std=1.0)
            torch.nn.init.normal_(self.w_comp, mean=0.0, std=1.0)
            
        elif init_type == 'IMI':
            with torch.no_grad():
                torch.nn.init.normal_(self.w, mean=0.0, std=0.1)
                torch.nn.init.normal_(self.w_comp, mean=0.0, std=0.1)
                mask = torch.rand(self.n_luts) < 0.9
                logit_val = 1.0  
                for i in range(2 ** self.lut_size):
                    first_input_bit = (i >> (self.lut_size - 1)) & 1
                    if first_input_bit == 1:
                        self.w[mask, i] = logit_val
                        self.w_comp[mask, i] = -logit_val
                    else:
                        self.w[mask, i] = -logit_val
                        self.w_comp[mask, i] = logit_val
        else :
            raise Exception("init_type error")

    def forward(self, x):
        inputs = [x[..., idx] for idx in self.indices]
        x_reordered = torch.stack(inputs, dim=-1)
        logits = torch.stack((self.w, self.w_comp), dim=0)

        if self.training:
            noise = 0.0 * torch.randn_like(logits)
            # Normalize these two values ​​to probabilities P(1) and P(0).
            sigma_w_probs = F.softmax((logits + noise) / 1, dim=0)
            sigma_w = sigma_w_probs[0]
            probs = self.get_probs(x_reordered)
            output = (probs * sigma_w).sum(-1)
        else:
            idx = bin2dec(torch.stack(inputs, dim=-1).round().int(), self.lut_size)
            rounded_luts = torch.round(F.softmax(logits, dim=0)[0])
            output = rounded_luts[range(rounded_luts.shape[0]), idx]

        return output

    def get_connections(self, input_dim, lut_input, n_luts, radius, device, dataset, connections):
        """
        Get LUT connections.
        :param input_dim: Number of input dimensions
        :param lut_input: Number of inputs per LUT
        :param n_luts: Number of LUTs
        :param connections: Method for initializing connectivity
        :param device: Device to place the tensors
        :return: Connection indices
        """
        if connections == 'GRC':
            conn = torch.randperm(lut_input * n_luts) % input_dim
            conn = torch.randperm(input_dim)[conn]
            conn = conn.reshape(lut_input, n_luts).to(device)
            return conn

        elif connections == 'LRC':
            if (dataset == 'cifar10-3' or dataset == 'cifar10-31'):
                # CIFAR-10, 32x32
                H, W = 32, 32
                C_feat = input_dim // (H * W) 
            elif (dataset == 'mnist'):
                # Mnist
                H, W = 28, 28    
                C_feat = 1

            # 1. Randomly select a "center point" coordinate (row, col) for each LUT.
            center_h = torch.randint(0, H, (n_luts,))
            center_w = torch.randint(0, W, (n_luts,))
            
            # 2.Define the local window radius (e.g., a radius of 2 results in a 5x5 window).
            radius = radius 
            
            # 3. Generate local offsets for each input port (lut_input=4) of each LUT.
            # off_h and off_w are offsets relative to the center point.
            off_h = torch.randint(-radius, radius + 1, (lut_input, n_luts))
            off_w = torch.randint(-radius, radius + 1, (lut_input, n_luts))
            
            # 4. Calculate actual pixel coordinates and prevent out-of-bounds errors (Clamp)
            target_h = (center_h + off_h).clamp(0, H - 1)
            target_w = (center_w + off_w).clamp(0, W - 1)
            
            # 5. The channel/threshold dimension (C_feat) remains randomly selected, 
            # and each LUT can simultaneously see different threshold information at the same location.
            target_c = torch.randint(0, C_feat, (lut_input, n_luts))
            
            # 6. Transform the (c, h, w) coordinates back to a one-dimensional index,
            # index = c * (H*W) + h * W + w
            conn = target_c * (H * W) + target_h * W + target_w
            
            return conn.to(device)

        else:
            raise ValueError("Invalid connection method: {}".format(connections))

    @staticmethod
    def get_probs(x):
        """
        Compute probabilities from inputs.
        :param x: Input tensor
        :return: Probability tensor
        """
        x = x.unsqueeze(-1)
        x_ = torch.cat([x, 1 - x], dim=-1)
        b, neur, n_components, _ = x_.shape

        alphabet = string.ascii_letters
        indices = [alphabet[i] for i in range(n_components)]
        equation = f"{','.join(['...' + i for i in indices])}->...{''.join(indices)}"

        components = x_.unbind(dim=2)
        result = torch.einsum(equation, *components)

        return result.reshape(b, neur, -1).flip(-1)

    def extra_repr(self):
        return 'num_luts={}, lut_size={}'.format(self.n_luts, self.lut_size)

class Aggregation(torch.nn.Module):
    """
    Aggregation module to aggregate outputs.
    """
    def __init__(self, num_classes: int, tau: float = 1.):
        """
        :param num_classes: Number of intended real valued outputs
        :param tau: Softmax temperature
        """
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x):
        """
        Aggregate the output tensor.
        :param x: Input tensor
        :return: Aggregated tensor
        """
        return x.reshape(x.size(0), self.num_classes, -1).sum(-1) / self.tau

    def extra_repr(self):
        return 'num_classes={}, tau={}'.format(self.num_classes, self.tau)