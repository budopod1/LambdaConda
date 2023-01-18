class IDGetter:
    def __init__(self):
        self.id_ = 0

    def __call__(self):
        id_ = self.id_
        self.id_ += 1
        return id_
