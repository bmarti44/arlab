from transition_table import parse_table


class Automaton:
    """A deterministic machine with partial transitions and explicit failures."""

    def __init__(self, table, start, accepting):
        symbols, transitions = parse_table(table)
        final_states = set(accepting)
        if start not in transitions:
            raise ValueError('unknown start state')
        if not final_states.issubset(transitions):
            raise ValueError('unknown accepting state')
        self._symbols = frozenset(symbols)
        self._transitions = transitions
        self._start = start
        self._accepting = frozenset(final_states)

    def run(self, text):
        state = self._start
        trace = [state]
        consumed = 0
        error = None
        for symbol in text:
            if symbol not in self._symbols:
                error = 'unknown symbol'
                break
            if symbol not in self._transitions[state]:
                error = 'missing transition'
                break
            state = self._transitions[state][symbol]
            trace.append(state)
            consumed += 1
        return {
            'accepted': error is None and state in self._accepting,
            'state': state,
            'trace': trace,
            'consumed': consumed,
            'error': error,
        }


def format_trace(result):
    chain = ' -> '.join(result['trace'])
    status = 'accepted' if result['accepted'] else 'rejected'
    if result['error'] is not None:
        status += f": {result['error']} at {result['consumed']}"
    return f'{chain} | {status}'
