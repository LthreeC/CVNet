import torch.nn as nn


def choose_lossfn(loss_name='cross_entropy'):
    losses_dict = {
        'mse': nn.MSELoss,
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCELoss,
        'nll': nn.NLLLoss
    }

    if loss_name not in losses_dict:
        raise ValueError('Invalid loss name: {}'.format(loss_name))

    return losses_dict[loss_name]()