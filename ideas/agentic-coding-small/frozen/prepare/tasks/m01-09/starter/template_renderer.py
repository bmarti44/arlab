import html
from template_parser import parse_template

def _resolve(path, context):
    """Look up mappings only; report the complete path on lookup failure."""
    raise NotImplementedError()

def _render_nodes(nodes, context):
    """Render a parsed subtree with lexical loop bindings."""
    raise NotImplementedError()

def render(template, context):
    """Parse before rendering so malformed syntax always fails."""
    raise NotImplementedError()
