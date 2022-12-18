import torch


def get_target_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"
