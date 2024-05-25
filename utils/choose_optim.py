import torch.optim as optim

def choose_optimizer(Optimizer, parameters, lr, betas=(0.9, 0.999), weight_decay=0):
    optimizer_name = Optimizer.lower()

    available_optimizers = [
        "sgd", "adam", "rmsprop", "adagrad", "adadelta", "adamax", "asgd", "lbfgs"
    ]

    if optimizer_name not in available_optimizers:
        raise ValueError(f"Invalid optimizer choice. Please choose from {available_optimizers}.")

    optimizer_cls = getattr(optim, optimizer_name.capitalize())

    if optimizer_name in ["adam", "adamax"]:
        optimizer = optimizer_cls(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name in ["sgd", "rmsprop", "adagrad", "adadelta", "asgd"]:
        optimizer = optimizer_cls(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "lbfgs":
        optimizer = optimizer_cls(parameters)

    return optimizer
