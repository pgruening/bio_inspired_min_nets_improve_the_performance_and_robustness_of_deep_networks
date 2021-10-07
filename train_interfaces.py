from DLBio.train_interfaces import Classification


def get_interface(ti_type, model, device, printer, **kwargs):
    if ti_type == Classification.name:
        return Classification(model, device, printer)
