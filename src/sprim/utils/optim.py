import torch

import vector_quantize_pytorch as vq


def build_optimizer(optim_config, params):
    return getattr(torch.optim, optim_config["name"])(params, **optim_config["options"])


def build_quantizer(quantizer_config):
    return getattr(vq, quantizer_config["type"])(**quantizer_config["options"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


mse2psnr = (
    lambda x: -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], device=x.device))
)
