from .net_OT import EfficientNetB0


def choose_model(ModelName, num_classes=1000, **kwargs):
    model_name = ModelName.lower()
    models_dict = {
        'efficientb0': EfficientNetB0,
        'resnet': EfficientNetB0,
        'densenet': EfficientNetB0,
        'vgg': EfficientNetB0
    }
    if model_name not in models_dict:
        raise ValueError('Invalid model name: {}'.format(model_name))

    return models_dict[model_name](num_classes=num_classes, **kwargs)