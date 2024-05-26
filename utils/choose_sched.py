import torch.optim.lr_scheduler as lr_scheduler

def choose_schedular(Schedular, optimizer, **kwargs):

    schedulers = {
        "StepLR": lr_scheduler.StepLR,
        "MultiStepLR": lr_scheduler.MultiStepLR,
        "ExponentialLR": lr_scheduler.ExponentialLR,
        "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR
    }


    if Schedular not in schedulers:
        raise ValueError(f"Invalid scheduler choice. Please choose from {list(schedulers.keys())}.")
    if schedulers[Schedular] is None:
        raise ValueError(f"Missing required parameters for {Schedular}.")

    return schedulers[Schedular](optimizer=optimizer, **kwargs)