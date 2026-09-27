import csv
import io


def _money(cents):
    return f'{cents // 100}.{cents % 100:02d}'


def render_report(store):
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['sku', 'name', 'quantity', 'unit_price', 'value'])
    for product, quantity in store.items():
        writer.writerow([product.sku, product.name, quantity, _money(product.price_cents), _money(product.price_cents * quantity)])
    writer.writerow(['TOTAL', '', '', '', _money(store.total_value())])
    return output.getvalue()
