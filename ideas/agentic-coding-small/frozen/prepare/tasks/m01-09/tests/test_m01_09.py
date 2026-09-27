import pytest
from template_parser import parse_template
from template_renderer import render

def test_parse_exact_tree():
    source = 'Hi {{ user.name }}!{% for x in rows %}[{{x}}]{% endfor %}.'
    assert parse_template(source) == [('text', 'Hi '), ('var', 'user.name'), ('text', '!'), ('for', 'x', 'rows', [('text', '['), ('var', 'x'), ('text', ']')]), ('text', '.')]
    assert parse_template('') == []
    assert parse_template('}}%}') == [('text', '}}%}')]

def test_escaping_and_literal_text():
    assert render('<b>{{v}}</b>{{nothing}}{{count}}', {'v': '&<>"\'', 'nothing': None, 'count': 0}) == '<b>&amp;&lt;&gt;&quot;&#x27;</b>0'
    assert render('{{x}}', {'x': '{{other}}'}) == '{{other}}'

def test_loop_order_empty_and_tuple():
    source = '{%for item in rows%}{{item.name}};{%endfor%}'
    assert render(source, {'rows': ({'name': 'A&'}, {'name': 'B'})}) == 'A&amp;;B;'
    assert render('{% for x in xs %}{{missing}}{% endfor %}', {'xs': []}) == ''
    assert parse_template('{%for x in xs%}{%endfor%}') == [('for', 'x', 'xs', [])]

def test_nested_scope_and_unchanged_context():
    context = {'x': 'outer', 'rows': [{'name': 'A', 'children': ['<1>', '2']}, {'name': 'B', 'children': []}]}
    source = '{{x}}|{% for x in rows %}{{x.name}}:{% for x in x.children %}{{x}},{% endfor %}{{x.name}};{% endfor %}|{{x}}'
    assert render(source, context) == 'outer|A:&lt;1&gt;,2,A;B:B;|outer'
    assert context == {'x': 'outer', 'rows': [{'name': 'A', 'children': ['<1>', '2']}, {'name': 'B', 'children': []}]}

def test_path_failures_and_loop_types():
    for template, context, path in [('{{a.b}}', {'a': {}}, 'a.b'), ('{{a.b}}', {'a': 3}, 'a.b'), ('{%for x in rows%}x{%endfor%}', {}, 'rows')]:
        with pytest.raises(KeyError) as error:
            render(template, context)
        assert error.value.args == (path,)
    for value in ('abc', 1, None, {'a': 1}):
        with pytest.raises(TypeError):
            render('{%for x in xs%}{{x}}{%endfor%}', {'xs': value})

def test_invalid_syntax():
    invalid = ['{{x', '{%for x in xs', '{{}}', '{{a..b}}', '{{a .b}}', '{{0a}}', '{%if x%}', '{%endfor%}', '{%for a.b in xs%}{%endfor%}', '{%for x in xs%}', '{%FOR x in xs%}{%endfor%}']
    for template in invalid:
        with pytest.raises(ValueError):
            parse_template(template)
        with pytest.raises(ValueError):
            render(template, {})

def test_whitespace_adjacent_tags_and_local_alias():
    source = '{% for\n_x\tin\tdata.items %}{{_x}}{{suffix}}{% endfor %}{{suffix}}'
    assert render(source, {'data': {'items': [1, 2]}, 'suffix': '!'}) == '1!2!!'
    assert parse_template('{{a}}{{b}}') == [('var', 'a'), ('var', 'b')]
    with pytest.raises(KeyError) as error:
        render('{%for y in xs%}{{y}}{%endfor%}{{y}}', {'xs': [1]})
    assert error.value.args == ('y',)
