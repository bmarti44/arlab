"""Pagination for in-memory sequences; pages are numbered from one."""


def _positive_integer(value, name):
    """Reject bools as well as non-integer input."""
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _window(page, pages, radius):
    """Return nearby valid page numbers."""
    if pages == 0:
        return []
    start = max(1, page - radius)
    end = min(pages, page + radius)
    return list(range(start, end + 1))


def paginate(items, page=1, per_page=10, radius=1):
    """Return a page, navigation flags, and a small page-number window.

    Out-of-range requests retain their requested page number. They return
    no items, but can still navigate toward existing earlier pages.
    Input is copied so page results do not alias the supplied list.
    """
    page = _positive_integer(page, "page")
    per_page = _positive_integer(per_page, "per_page")
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a nonnegative integer")

    records = list(items)
    total = len(records)
    pages = total // per_page
    offset = page * per_page
    selected = records[offset:offset + per_page]

    has_previous = total > 0 and page > 1
    has_next = page < pages
    previous_page = min(page - 1, pages) if has_previous else None
    next_page = page + 1 if has_next else None
    numbers = _window(page, pages, radius)

    return {
        "items": selected,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": pages,
        "has_previous": has_previous,
        "has_next": has_next,
        "previous_page": previous_page,
        "next_page": next_page,
        "window": numbers,
    }
