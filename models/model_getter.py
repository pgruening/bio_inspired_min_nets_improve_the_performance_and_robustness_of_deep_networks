from . import cifar_base_model, cifar_densenet
from torchvision.models import mobilenet_v2

EXP_10_MODELS = ['CifarMinFP-LS', 'CifarMinFP-ALL', 'CifarMinFP-LS-RNBasic']

EXP_11_MODELS = [
    'CifarDenseNet', 'CifarDenseMin'
]


def get_model(model_type, input_dim, output_dim, device, **kwargs):
    if model_type in EXP_10_MODELS:
        model = cifar_base_model.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    elif model_type in EXP_11_MODELS:
        model = cifar_densenet.get_model(
            model_type, input_dim, output_dim, device, **kwargs
        )

    else:
        raise ValueError(f"Unknown modeltype: {model_type}")

    return model.to(device).eval()
