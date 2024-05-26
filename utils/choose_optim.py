import torch.optim as optim

def choose_optimizer(Optimizer, parameters, **kwargs):
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        "AdamW": optim.AdamW,
        "Adamax": optim.Adamax,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "Rprop": optim.Rprop,
    }

    if Optimizer not in optimizers:
        raise ValueError(f"Invalid optimizer choice. Please choose from {list(optimizers.keys())}.")
    if optimizers[Optimizer] is None:
        raise ValueError(f"Missing required parameters for {Optimizer}.")

    return optimizers[Optimizer](params=parameters, **kwargs)
