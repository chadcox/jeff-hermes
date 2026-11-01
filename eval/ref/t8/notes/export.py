from . import util


def filename(title):
    return util.slugify(title) + ".md"
