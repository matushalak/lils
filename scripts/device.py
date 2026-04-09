import torch


def mps_available():
    return (hasattr(torch.backends, 'mps') and
            torch.backends.mps.is_built() and
            torch.backends.mps.is_available())


def resolve_device(no_cuda=False):
    if no_cuda:
        return torch.device('cpu')

    if torch.cuda.is_available():
        return torch.device('cuda')

    if mps_available():
        return torch.device('mps')

    return torch.device('cpu')
