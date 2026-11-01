class Cart:
    def __init__(self, lines=None):
        self.lines = [] if lines is None else lines

    def add(self, sku, qty):
        self.lines.append((sku, qty))
        return self
