from flops_counter import get_model_complexity_info
import models as models
# SAResNet101   PreActResNet101
net = models.__dict__['SAResNet101'](num_classes=100)

flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
print('Flops: ' + flops)
print('Param: ' + params)