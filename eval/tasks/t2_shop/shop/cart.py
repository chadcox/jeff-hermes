class Cart:
    def __init__(self, lines=[]):
        self.lines = lines

    def add(self, sku, qty):
        self.lines.append((sku, qty))
        return self
