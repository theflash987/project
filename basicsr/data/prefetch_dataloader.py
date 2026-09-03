"""Iterator wrapper used by the NoRouter-K16 training loop."""


class CPUPrefetcher:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None

    def reset(self):
        self.iterator = iter(self.loader)

    def next(self):
        return next(self.iterator, None)
