import copy
import pytest
from world import make_world
from adventure import Game

def test_world_shape_and_independence():
    expected = {'foyer': {'exits': {'north': 'hall'}, 'items': ['key']}, 'hall': {'exits': {'south': 'foyer', 'east': 'vault', 'west': 'garden'}, 'items': []}, 'garden': {'exits': {'east': 'hall'}, 'items': ['coin']}, 'vault': {'exits': {'west': 'hall'}, 'items': ['gem']}}
    a, b = make_world(), make_world()
    assert a == b == expected
    a['foyer']['items'].clear()
    a['hall']['exits'].clear()
    assert b == expected

def test_initial_look_and_command_normalization():
    g = Game()
    assert g.look() == 'Room: foyer\nExits: north\nItems: key'
    assert g.command('  LOOK \t') == g.look()
    assert g.command('inventory') == 'Inventory: (empty)'
    assert g.command('GO   NORTH') == 'You enter hall.'
    assert g.look() == 'Room: hall\nExits: east, south, west\nItems: (none)'

def test_locked_route_and_unlock_rules():
    g = Game()
    assert g.command('unlock') == 'There is no lock here.'
    g.command('go north')
    before = g.snapshot()
    assert g.command('go east') == 'The vault is locked.'
    assert g.command('unlock') == 'You need a key.'
    assert g.snapshot() == before
    g.command('go south')
    assert g.command('take key') == 'Taken: key.'
    g.command('go north')
    assert g.command('unlock') == 'You unlock the vault.'
    assert g.command('inventory') == 'Inventory: key'
    assert g.command('unlock') == 'The vault is already unlocked.'
    assert g.command('go east') == 'You enter vault.'
    assert g.command('unlock') == 'There is no lock here.'
    assert g.command('go west') == 'You enter hall.'

def test_item_moves_and_sorted_inventory():
    g = Game()
    g.command('take key')
    assert g.command('take key') == 'No such item here.'
    g.command('go north')
    g.command('go west')
    assert g.command('take coin') == 'Taken: coin.'
    assert g.command('inventory') == 'Inventory: coin, key'
    assert g.command('drop key') == 'Dropped: key.'
    assert g.command('drop coin') == 'Dropped: coin.'
    assert g.look() == 'Room: garden\nExits: east\nItems: coin, key'
    assert g.command('drop key') == 'You are not carrying that.'

def test_invalid_commands_preserve_state():
    g = Game()
    before = g.snapshot()
    for text in ('', 'dance', 'go', 'go north now', 'look around', 'inventory x', 'unlock vault', 'take', 'eat key'):
        assert g.command(text) == 'Unknown command.'
        assert g.snapshot() == before
    assert g.command('go south') == 'You cannot go that way.'
    assert g.command('take gem') == 'No such item here.'
    assert g.command('drop gem') == 'You are not carrying that.'
    assert g.snapshot() == before

def test_snapshot_roundtrip_and_copies():
    g = Game()
    for command in ('take key', 'go north', 'unlock', 'go east', 'take gem'):
        g.command(command)
    saved = g.snapshot()
    assert saved == {'room': 'vault', 'inventory': ['gem', 'key'], 'unlocked': True, 'items': {'foyer': [], 'hall': [], 'garden': ['coin'], 'vault': []}}
    saved['inventory'].reverse()
    restored = Game.from_snapshot(saved)
    saved['items']['garden'].clear()
    saved['inventory'].clear()
    assert restored.snapshot() == g.snapshot()
    for command in ('go west', 'drop gem', 'go west', 'take coin', 'inventory'):
        assert restored.command(command) == g.command(command)
    assert restored.snapshot() == g.snapshot()
    assert Game().look().endswith('Items: key')

def test_reject_invalid_snapshots():
    good = Game().snapshot()
    invalid = []
    for field, value in [('room', 'unknown'), ('room', []), ('unlocked', 1), ('inventory', 'key'), ('inventory', ['key']), ('inventory', [1]), ('items', {}), ('items', []), ('room', 'vault')]:
        case = copy.deepcopy(good)
        case[field] = value
        invalid.append(case)
    case = copy.deepcopy(good)
    del case['items']['vault']
    invalid.append(case)
    case = copy.deepcopy(good)
    case['items']['vault'] = ['unknown']
    invalid.append(case)
    case = copy.deepcopy(good)
    case['items']['foyer'] = ('key',)
    invalid.append(case)
    invalid.extend([{}, dict(good, extra=1)])
    for data in invalid:
        before = copy.deepcopy(data)
        with pytest.raises(ValueError):
            Game.from_snapshot(data)
        assert data == before
