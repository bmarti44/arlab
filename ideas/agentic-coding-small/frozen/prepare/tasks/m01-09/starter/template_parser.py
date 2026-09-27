import re
NAME = '[A-Za-z_][A-Za-z0-9_]*'
PATH = NAME + '(?:\\.' + NAME + ')*'
FOR = re.compile('for\\s+(' + NAME + ')\\s+in\\s+(' + PATH + ')')

def parse_template(template):
    """Parse balanced blocks into a tree using a stack of child lists."""
    raise NotImplementedError()
