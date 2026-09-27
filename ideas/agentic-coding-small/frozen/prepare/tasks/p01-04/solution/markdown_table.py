def escape_cell(value):
    text = '' if value is None else str(value)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.replace('\\', '\\\\').replace('|', '\\|').replace('\n', '<br>')

def format_table(headers, rows, alignments=None):
    if not headers:
        raise ValueError('no columns')
    count = len(headers)
    if any(len(row) != count for row in rows):
        raise ValueError('row length')
    if alignments is None:
        alignments = ['left'] * count
    if len(alignments) != count or any(a not in ('left', 'right', 'center') for a in alignments):
        raise ValueError('alignment')
    cells = [[escape_cell(v) for v in row] for row in [headers] + rows]
    widths = [max(3, *(len(row[i]) for row in cells)) for i in range(count)]
    def pad(text, width, alignment):
        extra = width - len(text)
        left = 0 if alignment == 'left' else extra if alignment == 'right' else extra // 2
        return ' ' * left + text + ' ' * (extra - left)
    def line(row):
        return '| ' + ' | '.join(row) + ' |'
    rendered = [line([pad(v, widths[i], alignments[i]) for i, v in enumerate(row)]) for row in cells]
    separators = []
    for width, alignment in zip(widths, alignments):
        if alignment == 'left':
            separators.append(':' + '-' * (width - 1))
        elif alignment == 'right':
            separators.append('-' * (width - 1) + ':')
        else:
            separators.append(':' + '-' * (width - 2) + ':')
    rendered.insert(1, line(separators))
    return '\n'.join(rendered)
