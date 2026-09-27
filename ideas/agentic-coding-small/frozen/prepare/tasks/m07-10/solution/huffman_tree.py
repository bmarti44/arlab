"""Construct deterministic Huffman codewords from character counts."""
from collections import Counter
import heapq


def build_codes(text: str) -> dict[str, str]:
    frequencies = Counter(text)
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        return {next(iter(frequencies)): "0"}

    heap = []
    for char, frequency in frequencies.items():
        heap.append((frequency, char, char))
    heapq.heapify(heap)
    while len(heap) > 1:
        left_frequency, left_minimum, left = heapq.heappop(heap)
        right_frequency, right_minimum, right = heapq.heappop(heap)
        parent = (left, right)
        heapq.heappush(heap, (
            left_frequency + right_frequency,
            min(left_minimum, right_minimum),
            parent,
        ))

    codes = {}
    stack = [(heap[0][2], "")]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, str):
            codes[node] = prefix
        else:
            left, right = node
            stack.append((right, prefix + "1"))
            stack.append((left, prefix + "0"))
    return codes
