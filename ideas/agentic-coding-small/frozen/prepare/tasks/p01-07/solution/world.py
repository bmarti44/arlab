def make_world():
    return {
        'foyer': {'exits': {'north': 'hall'}, 'items': ['key']},
        'hall': {'exits': {'south': 'foyer', 'east': 'vault', 'west': 'garden'}, 'items': []},
        'garden': {'exits': {'east': 'hall'}, 'items': ['coin']},
        'vault': {'exits': {'west': 'hall'}, 'items': ['gem']},
    }
