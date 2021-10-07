from . import ds_cifar10
import warnings


def get_data_loaders(dataset, *, batch_size, num_workers, **kwargs):
    if dataset == 'cifar_10':
        use_val = kwargs.get('use_val', [True])[0]
        do_normalize = kwargs.get("do_normalize", [True])[0]
        if use_val:
            val_data_loader = ds_cifar10.get_data_loader(
                is_train=False, batch_size=batch_size,
                num_workers=num_workers
            )
        else:
            warnings.warn('No validation data-loader is used')
            val_data_loader = None

        return {
            "train": ds_cifar10.get_data_loader(
                is_train=True,
                batch_size=batch_size,
                num_workers=num_workers,
                do_normalize=do_normalize,
            ),
            'val': val_data_loader,
            'test': None
        }

    else:
        raise ValueError(f"Unkown dataset: {dataset}")
