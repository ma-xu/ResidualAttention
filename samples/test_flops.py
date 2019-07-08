from flops_counter import get_model_complexity_info
import models as models
# SAResNet101   PreActResNet101
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
net = models.__dict__['RSResNet50'](num_classes=100)

flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
print('Flops: ' + flops)
print('Param: ' + params)