import re


def slugfy(text):
    """'Hello, World!' -> 'hello-world'"""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
