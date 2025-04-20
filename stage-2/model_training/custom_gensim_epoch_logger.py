from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    def __init__(self, tqdm_bar):
        self.epoch = 0
        self.tqdm_bar = tqdm_bar

    def on_epoch_end(self, model):
        self.epoch += 1
        self.tqdm_bar.update(1)