from .util import slugfy


class Store:
    def __init__(self):
        self.notes = {}

    def add(self, title, body):
        key = slugfy(title)
        self.notes[key] = body
        return key
