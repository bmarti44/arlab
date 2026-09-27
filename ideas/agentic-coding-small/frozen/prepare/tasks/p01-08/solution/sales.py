import csv
import io
import re


def parse_sales(text):
    try:
        rows = [row for row in csv.reader(io.StringIO(text, newline=''), strict=True) if row]
    except csv.Error as exc:
        raise ValueError('invalid csv') from exc
    required = {'region', 'product', 'quantity', 'unit_price'}
    if not rows:
        raise ValueError('missing header')
    headers = [value.strip() for value in rows[0]]
    if len(headers) != 4 or set(headers) != required:
        raise ValueError('invalid header')
    result = []
    for row in rows[1:]:
        if len(row) != 4:
            raise ValueError('invalid row length')
        record = dict(zip(headers, (value.strip() for value in row)))
        if not record['region'] or not record['product']:
            raise ValueError('blank group')
        if re.fullmatch(r'[1-9][0-9]*', record['quantity']) is None:
            raise ValueError('invalid quantity')
        if re.fullmatch(r'[0-9]+(?:\.[0-9]{1,2})?', record['unit_price']) is None:
            raise ValueError('invalid price')
        dollars, _, fraction = record['unit_price'].partition('.')
        cents = int(dollars) * 100 + int(fraction.ljust(2, '0'))
        result.append({'region': record['region'], 'product': record['product'], 'quantity': int(record['quantity']), 'unit_price_cents': cents})
    return result


def aggregate_sales(sales, group_by='region', sort_by='revenue'):
    if group_by not in ('region', 'product') or sort_by not in ('group', 'quantity', 'revenue'):
        raise ValueError('invalid grouping or sorting')
    groups = {}
    for sale in sales:
        name = sale[group_by]
        row = groups.setdefault(name, {'group': name, 'quantity': 0, 'revenue_cents': 0, 'orders': 0})
        row['quantity'] += sale['quantity']
        row['revenue_cents'] += sale['quantity'] * sale['unit_price_cents']
        row['orders'] += 1
    if sort_by == 'group':
        return sorted(groups.values(), key=lambda row: row['group'])
    field = 'quantity' if sort_by == 'quantity' else 'revenue_cents'
    return sorted(groups.values(), key=lambda row: (-row[field], row['group']))
