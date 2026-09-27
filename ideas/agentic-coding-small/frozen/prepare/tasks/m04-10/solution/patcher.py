from diff_parser import parse_patch


def apply_patch(original, patch):
    if original and not original.endswith('\n'):
        raise ValueError('original must end in LF')
    source = original.split('\n')[:-1] if original else []
    hunks = parse_patch(patch)
    output = []
    cursor = 0
    for hunk in hunks:
        old_index = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        if old_index < cursor or old_index > len(source):
            raise ValueError('hunk outside source or overlaps earlier hunk')
        output.extend(source[cursor:old_index])
        cursor = old_index
        new_index = hunk.new_start - 1 if hunk.new_count else hunk.new_start
        if new_index != len(output):
            raise ValueError('inconsistent new-file coordinate')
        for prefix, content in hunk.lines:
            if prefix in ' -':
                if cursor >= len(source) or source[cursor] != content:
                    raise ValueError('source content mismatch')
                cursor += 1
            if prefix in ' +':
                output.append(content)
    output.extend(source[cursor:])
    if not output:
        return ''
    return '\n'.join(output) + '\n'
