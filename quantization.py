
import torch
import os

class QuantizerBase:
    pass

class RandomQuantizer(QuantizerBase):
    def __init__(self):
        self.set ='random'
        self.b = 16
        self.s = torch.pow(torch.tensor(2), self.b) - 1
    def quantize(self, g):
        ### g is input to be quantized
        ### b is # of bits
        s = self.s  ## number of quantization levels
        norm = torch.norm(g)
        g_normalized = torch.abs(g) / norm
        l = torch.floor(g_normalized * s)
        p = (s * g_normalized - l)
        xi = (l + torch.distributions.binomial.Binomial(1, p).sample())*2 + (torch.sign(g) + 1) / 2
        xi = xi.byte()
        return xi, norm
    def dequantize(self, xi, norm):
        sign = torch.fmod(xi, 2).float()
        
        sign = sign * 2 - 1
        xi = (xi / 2).float()
        return norm * sign * xi / self.s

class KContractionQuantizer(QuantizerBase):
    def __init__(self):
        self.set = 'topk'
        self.frac = 0.3
    def quantize(self, g):
        d = g.shape[-1]
        k = round(d * self.frac)
        return torch.topk(g, k)
        #mask = torch.zeros_like(g)
        #mask.scatter_(-1, topk_indices, 1)
        #comp = g * mask
        #comp = comp.byte()
        

#Uniform affine quantization
class Quantizer(QuantizerBase):
    def __init__(self):
        self.num_bits = 8
        pass
    def quantize(self, x):
        qmin = torch.tensor(0.).cuda()
        qmax = torch.tensor(2. ** self.num_bits - 1.).cuda()
        min_val, max_val = x.min(), x.max()

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale


        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = torch.floor(zero_point)
        q_x = zero_point + x / scale
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        return q_x, scale, zero_point

    def dequantize(self, q_x, scale, zero_point):
        return scale * (q_x.float() - zero_point)
        
        
class NaturalCompressionQuantizer(QuantizerBase):
    def __init__(self):
        self.set = 'natural'
        pass

    def quantize(self, g):
        mask_nonzero = g != 0
        abs_g = torch.abs(g)
        sign_g = torch.sign(g)

        log2_abs_g = torch.log2(abs_g[mask_nonzero])
        int_log2 = torch.floor(log2_abs_g)
        fraction = log2_abs_g - int_log2

        p_t = 1 - fraction 


        add_one = torch.bernoulli(p_t).to(torch.int)  
        log2_quantized = int_log2 + add_one

        #log2_quantized = torch.clamp(log2_quantized, -128, 127).to(torch.int8)  # -128~127 8bits
        #log2_quantized_full = torch.zeros_like(g, dtype=torch.int8)
        
        log2_quantized_full = torch.zeros_like(g)
        log2_quantized_full[mask_nonzero] = log2_quantized
        sign_g = torch.where(g == 0, torch.tensor(1.0, device=g.device), sign_g)  

        return sign_g, log2_quantized_full

    def dequantize(self, sign_g, log2_quantized):
        result = sign_g * (2 ** log2_quantized)
        return result

class NCQuantizer(QuantizerBase):
    def __init__(self):
        self.set = 'natural'
        pass

    def quantize(self, g):
        abs_g = torch.abs(g)
        sign_g = torch.sign(g)

        log2_abs_g = torch.where(abs_g > 0, torch.log2(abs_g), torch.tensor(0.0, device = g.device))
        log2_floor = torch.floor(log2_abs_g)
        log2_ceil = torch.ceil(log2_abs_g)
        p_t = 1 - (2 ** log2_ceil - abs_g) / (2 ** log2_floor)
        p_t[p_t >= 1] = 1
        
        c_nat = sign_g * (2 ** log2_floor) * (1 + torch.bernoulli(p_t))
        c_nat[g == 0] = 0

        return c_nat
