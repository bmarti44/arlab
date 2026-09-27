import re

NAME = r"[A-Za-z_][A-Za-z0-9_]*"
PATH = NAME + r"(?:\." + NAME + r")*"
FOR = re.compile(r"for\s+(" + NAME + r")\s+in\s+(" + PATH + r")")


def parse_template(template):
    """Parse balanced blocks into a tree using a stack of child lists."""
    root = []
    stack = [root]
    position = 0
    while position < len(template):
        variable = template.find("{{", position)
        block = template.find("{%", position)
        starts = [start for start in (variable, block) if start >= 0]
        if not starts:
            stack[-1].append(("text", template[position:]))
            break
        start = min(starts)
        if start > position:
            stack[-1].append(("text", template[position:start]))
        is_variable = template.startswith("{{", start)
        closer = "}}" if is_variable else "%}"
        end = template.find(closer, start + 2)
        if end < 0:
            raise ValueError("unclosed tag")
        content = template[start + 2:end].strip()
        if is_variable:
            if not re.fullmatch(PATH, content):
                raise ValueError("invalid variable path")
            stack[-1].append(("var", content))
        elif content == "endfor":
            if len(stack) == 1:
                raise ValueError("unexpected endfor")
            stack.pop()
        else:
            match = FOR.fullmatch(content)
            if match is None:
                raise ValueError("invalid block tag")
            alias, path = match.groups()
            children = []
            stack[-1].append(("for", alias, path, children))
            stack.append(children)
        position = end + 2
    if len(stack) != 1:
        raise ValueError("unclosed loop")
    return root
