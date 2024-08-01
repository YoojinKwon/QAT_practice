import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

"""
Straight-Through Estimator (STE) in Quantization-Aware Training (QAT)
"""
# STE 1 for [ICLR '20] Learned Step size Quantization (LSQ)
# Refer to https://github.com/hustzxd/LSQuantization/blob/master/lsq.py#L47
def identity_with_grad_scaling(x, scale):
    y_fp = x            # At forward prop
    y_bp = x * scale    # At back-prop 
    return y_fp.detach() - y_bp.detach() + y_bp  # detach(): requires_grad=False

# STE 2 for LSQ
# Refer to https://github.com/hustzxd/LSQuantization/blob/master/lsq.py#L53
def round_with_grad_pass(x):
    y_fp = x.round()    # At forward prop
    y_bp = x            # At back-prop
    return y_fp.detach() - y_bp.detach() + y_bp

"""
Fake-quantizer in Quantization-Aware Training (QAT)
"""
# Refer to https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L61 and https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L107
# per_channel is False
class LSQWeightFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        all_positive=False,
        symmetric=False):
        super(LSQWeightFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 33)]
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.n = 0
            self.p = 2 ** bits - 1
        else:
            if symmetric:
                self.n = - 2 ** (bits - 1) + 1
                self.p = 2 ** (bits - 1) - 1
            else:
                self.n = - 2 ** (bits - 1)
                self.p = 2 ** (bits - 1) - 1

        self.s = nn.Parameter(torch.tensor(-1.0))   # Dummy

    def forward(self, w):
        grad_scale = 1.0 / math.sqrt(w.numel() * self.p)

        if self.s == -1.0:
            self.s.data = 2.0 * w.abs().mean() / math.sqrt(self.p) # initialization

        s = identity_with_grad_scaling(self.s, grad_scale) # step size
        # w_q = round_with_grad_pass((w / s).clamp(self.n, self.p))
        w_fq = round_with_grad_pass((w / s).clamp(self.n, self.p)) * s
        return w_fq


# Refer to https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L136
class LSQActFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        symmetric=False):
        super(LSQActFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 33)]
        self.bits = bits
        self.symmetric = symmetric
        self.n = -1.0                               # Dummy
        self.p = -1.0                               # Dummy

        self.s = nn.Parameter(torch.tensor(-1.0))   # Dummy

    def forward(self, x):
        if self.p == -1.0:
            all_positive = x.min() >= -1e-5
            if all_positive:
                assert not self.symmetric, "Positive quantization cannot be symmetric"
                self.n = 0
                self.p = 2 ** self.bits - 1
            else:
                if self.symmetric:
                    self.n = - 2 ** (self.bits - 1) + 1
                    self.p = 2 ** (self.bits - 1) - 1
                else:
                    self.n = - 2 ** (self.bits - 1)
                    self.p = 2 ** (self.bits - 1) - 1

        grad_scale = 1.0 / math.sqrt(x.numel() * self.p)

        if self.s == -1.0:
            self.s.data = 2.0 * x.abs().mean() / math.sqrt(self.p)

        s = identity_with_grad_scaling(self.s, grad_scale)
        # x_q = round_with_grad_pass((x / s).clamp(self.n, self.p))
        x_fq = round_with_grad_pass((x / s).clamp(self.n, self.p)) * s
        return x_fq


"""
Etc.
"""
def get_extra_q_bits(scheme, bits_w, bits_a):
    if scheme == 'sat':
        bits_w_first = max(bits_w, 8) if bits_w is not None else None
        bits_a_first = None     # Practically 8 if bits_a is not None, since model input will be in uint8 after multiplied by 255 then rounded (as data augmentation)
        bits_w_last = max(bits_w, 8) if bits_w is not None else None
        bits_a_last = bits_a
    elif scheme == 'lsq':
        bits_w_first = max(bits_w, 8) if bits_w is not None else None
        bits_a_first = max(bits_a, 8) if bits_a is not None else None
        bits_w_last = max(bits_w, 8) if bits_w is not None else None
        bits_a_last = max(bits_a, 8) if bits_a is not None else None
    else:
        assert scheme is None
        bits_w_first = None
        bits_a_first = None
        bits_w_last = None
        bits_a_last = None

    return bits_w_first, bits_a_first, bits_w_last, bits_a_last
