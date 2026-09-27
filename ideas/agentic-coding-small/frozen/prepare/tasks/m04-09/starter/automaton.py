from transition_table import parse_table


class Automaton:
    def __init__(self, table, start, accepting):
        raise NotImplementedError

    def run(self, text):
        raise NotImplementedError


def format_trace(result):
    raise NotImplementedError
