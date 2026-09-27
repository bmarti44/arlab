import html
from template_parser import parse_template


def _resolve(path, context):
    """Look up mappings only; report the complete path on lookup failure."""
    value = context
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def _render_nodes(nodes, context):
    """Render a parsed subtree with lexical loop bindings."""
    result = []
    for node in nodes:
        kind = node[0]
        if kind == "text":
            result.append(node[1])
        elif kind == "var":
            value = _resolve(node[1], context)
            text = "" if value is None else str(value)
            result.append(html.escape(text, quote=True))
        else:
            alias, path, children = node[1:]
            values = _resolve(path, context)
            if not isinstance(values, (list, tuple)):
                raise TypeError("loop value must be a list or tuple")
            for value in values:
                local = dict(context)
                local[alias] = value
                result.append(_render_nodes(children, local))
    return "".join(result)


def render(template, context):
    """Parse before rendering so malformed syntax always fails."""
    nodes = parse_template(template)
    return _render_nodes(nodes, context)
