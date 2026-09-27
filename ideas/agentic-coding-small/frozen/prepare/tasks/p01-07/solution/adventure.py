from world import make_world

class Game:
    def __init__(self):
        self.world = make_world()
        self.room = 'foyer'
        self.inventory = []
        self.unlocked = False

    def look(self):
        room = self.world[self.room]
        exits = ', '.join(sorted(room['exits']))
        items = ', '.join(sorted(room['items'])) or '(none)'
        return f'Room: {self.room}\nExits: {exits}\nItems: {items}'

    def command(self, text):
        words = text.strip().lower().split()
        if words == ['look']:
            return self.look()
        if words == ['inventory']:
            return 'Inventory: ' + (', '.join(sorted(self.inventory)) or '(empty)')
        if words == ['unlock']:
            if self.room != 'hall':
                return 'There is no lock here.'
            if self.unlocked:
                return 'The vault is already unlocked.'
            if 'key' not in self.inventory:
                return 'You need a key.'
            self.unlocked = True
            return 'You unlock the vault.'
        if len(words) != 2:
            return 'Unknown command.'
        verb, noun = words
        room = self.world[self.room]
        if verb == 'go':
            if noun not in room['exits']:
                return 'You cannot go that way.'
            if self.room == 'hall' and noun == 'east' and not self.unlocked:
                return 'The vault is locked.'
            self.room = room['exits'][noun]
            return f'You enter {self.room}.'
        if verb == 'take':
            if noun not in room['items']:
                return 'No such item here.'
            room['items'].remove(noun)
            self.inventory.append(noun)
            return f'Taken: {noun}.'
        if verb == 'drop':
            if noun not in self.inventory:
                return 'You are not carrying that.'
            self.inventory.remove(noun)
            room['items'].append(noun)
            return f'Dropped: {noun}.'
        return 'Unknown command.'

    def snapshot(self):
        return {'room': self.room, 'inventory': sorted(self.inventory), 'unlocked': self.unlocked,
                'items': {name: sorted(room['items']) for name, room in self.world.items()}}

    @classmethod
    def from_snapshot(cls, data):
        game = cls()
        if not isinstance(data, dict) or set(data) != {'room', 'inventory', 'unlocked', 'items'}:
            raise ValueError('invalid snapshot keys')
        room = data['room']
        if not isinstance(room, str) or room not in game.world or type(data['unlocked']) is not bool:
            raise ValueError('invalid room or lock')
        if room == 'vault' and not data['unlocked']:
            raise ValueError('locked vault')
        items = data['items']
        if not isinstance(items, dict) or set(items) != set(game.world):
            raise ValueError('invalid rooms')
        lists = [data['inventory']] + list(items.values())
        if any(not isinstance(values, list) or any(not isinstance(v, str) for v in values) for values in lists):
            raise ValueError('invalid item list')
        all_items = [v for values in lists for v in values]
        if sorted(all_items) != ['coin', 'gem', 'key']:
            raise ValueError('invalid item distribution')
        game.room = room
        game.unlocked = data['unlocked']
        game.inventory = list(data['inventory'])
        for name in game.world:
            game.world[name]['items'] = list(items[name])
        return game
