from mmcv.runner import Hook


class DatasetParamAdjustHook(Hook):
    """Adjust the param of a dataset (by iterations).
    """

    def __init__(self,
                 param_name_adjust_iter_value,
                 logger=None):
        self.param_name_adjust_iter_value = param_name_adjust_iter_value
        self.logger = logger

    def after_train_iter(self, runner):
        for name, adjust_iter, value in self.param_name_adjust_iter_value:
            if (runner.iter + 1) == adjust_iter:
                for loader in runner.data_loaders:
                    if hasattr(loader.dataset, 'dataset'):
                        # handle warpper
                        dataset = loader.dataset.dataset
                    else:
                        dataset = loader.dataset
                    old_value = getattr(dataset, name, 'Undefined')
                    setattr(dataset, name, value)
                    if self.logger:
                        self.logger.info(f'{name} for {dataset} is set '
                                         f'from {old_value} to {value}')
