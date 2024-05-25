import torch.optim.lr_scheduler as lr_scheduler

def choose_schedular(Schedular, optimizer, step_size=None, milestones=None, gamma=0.1, T_max=None, eta_min=0):
    schedulers = {
        "StepLR": lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) if step_size else None,
        "MultiStepLR": lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) if milestones else None,
        "ExponentialLR": lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
        "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=10,
                                                            min_lr=eta_min),
        "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min) if T_max else None
    }


    if Schedular not in schedulers:
        raise ValueError(f"Invalid scheduler choice. Please choose from {list(schedulers.keys())}.")
    if schedulers[Schedular] is None:
        raise ValueError(f"Missing required parameters for {Schedular}.")

    return schedulers[Schedular]