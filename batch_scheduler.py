def get_batch_scheduler(type_, optimizer, **kwargs):
    if type_ == WarmUpScheduler.name:
        return WarmUpScheduler(optimizer, **kwargs)
    elif type_ == WarmUpSchedulerThenFixedSteps.name:
        return WarmUpSchedulerThenFixedSteps(optimizer, **kwargs)

    else:
        raise ValueError(type_)


class WarmUpScheduler():
    name = 'WarmUpScheduler'

    def __init__(self, optimizer, *, initial_lr, max_lr, max_wu_epochs):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr[0])
        self.max_lr = float(max_lr[0])
        self.max_wu_epochs = float(max_wu_epochs[0])

    def step(self, epoch, iteration, max_num_iterations):
        if epoch >= self.max_wu_epochs:
            return

        lr = self.optimizer.param_groups[0]['lr']
        l0 = self.initial_lr
        l1 = self.max_lr

        # at epoch zero -> only count iterations
        x = iteration + epoch * max_num_iterations
        x_max = max_num_iterations * self.max_wu_epochs

        # linear interpolation of start and max learning rate
        lr = l0 + (l1 - l0) * x / x_max

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class WarmUpSchedulerThenFixedSteps(WarmUpScheduler):
    name = 'WarmUpSchedulerThenFixedSteps'

    def __init__(self, optimizer, *, initial_lr, max_lr, max_wu_epochs, fixed_steps, gamma=.1):
        super(WarmUpSchedulerThenFixedSteps, self).__init__(
            optimizer, initial_lr=initial_lr, max_lr=max_lr,
            max_wu_epochs=max_wu_epochs
        )
        self.fixed_steps = [int(s) for s in fixed_steps]
        print(self.fixed_steps)
        self.gamma = gamma

    def step(self, epoch, iteration, max_num_iterations):
        # when warmup is done -> check if fixed steps scheduling is used
        if epoch >= self.max_wu_epochs and iteration == 0:
            return self.fixed_rate_step(epoch)
        else:
            # (maybe) run warm-up scheduling
            super(WarmUpSchedulerThenFixedSteps, self).step(
                epoch, iteration, max_num_iterations
            )

    def fixed_rate_step(self, epoch):
        # alter the learning rate precisely at the epochs in fixed rate
        if int(epoch) in self.fixed_steps:
            lr = self.optimizer.param_groups[0]['lr']
            lr *= self.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
