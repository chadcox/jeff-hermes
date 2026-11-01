from .util import slugify


class Store:
    def __init__(self):
        self.notes = {}

    def add(self, title, body):
        key = slugify(title)
        self.notes[key] = body
        return key
