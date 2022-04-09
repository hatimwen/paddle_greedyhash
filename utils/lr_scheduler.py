from paddle.optimizer.lr import LRScheduler


class DecreaseLRScheduler(LRScheduler):
    def __init__(self,
                 learning_rate,
                 start_lr,
                 epoch_lr_decrease):
        """init DecreaseLRScheduler """
        self.start_lr = start_lr
        self.epoch_lr_decrease = epoch_lr_decrease
        super(DecreaseLRScheduler, self).__init__(learning_rate)
    
    def get_lr(self):
        t = self.last_epoch
        val = self.start_lr * (0.1 ** (t // self.epoch_lr_decrease))
        return val
