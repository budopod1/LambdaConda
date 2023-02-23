from time import perf_counter


class Timer:
    def __init__(self):
        self.start()

    def start(self):
        self.started = perf_counter()

    def elapsed(self):
        return perf_counter() - self.started
        