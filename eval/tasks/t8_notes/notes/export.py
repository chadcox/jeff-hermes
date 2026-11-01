from . import util


def filename(title):
    return util.slugfy(title) + ".md"
