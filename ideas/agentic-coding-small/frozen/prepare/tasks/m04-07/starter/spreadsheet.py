def normalize_cell(name):
    raise NotImplementedError


def parse_formula(formula):
    raise NotImplementedError


class Spreadsheet:
    def __init__(self):
        raise NotImplementedError

    def set(self, name, value):
        raise NotImplementedError

    def get(self, name):
        raise NotImplementedError

    def delete(self, name):
        raise NotImplementedError
