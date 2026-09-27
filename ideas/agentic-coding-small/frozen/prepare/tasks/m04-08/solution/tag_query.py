import re


def normalize_tag(tag):
    """Canonicalize one tag while excluding query punctuation."""
    value = tag.strip().lower()
    if re.fullmatch(r'[a-z0-9][a-z0-9_-]*', value) is None:
        raise ValueError('invalid tag')
    return value


def parse_query(query):
    """Parse conjunctive groups and exclusions without consulting an index.

    Keep group order so callers can inspect the original query structure.
    """
    groups = []
    excluded = set()
    for token in query.split():
        if token.startswith('-'):
            tag = token[1:]
            if '|' in tag:
                raise ValueError('exclusions must contain one tag')
            excluded.add(normalize_tag(tag))
        else:
            alternatives = token.split('|')
            group = frozenset(normalize_tag(tag) for tag in alternatives)
            groups.append(group)
    return groups, frozenset(excluded)
