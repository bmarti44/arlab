import re

class Config:

    def __init__(self):
        raise NotImplementedError()

    def get(self, section, key, default=None):
        raise NotImplementedError()

    def get_int(self, section, key, default=None):
        raise NotImplementedError()

    def get_bool(self, section, key, default=None):
        raise NotImplementedError()

def parse(text):
    raise NotImplementedError()
