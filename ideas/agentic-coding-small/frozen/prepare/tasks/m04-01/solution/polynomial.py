class Polynomial:
    """An integer polynomial stored in ascending coefficient order."""

    def __init__(self, coefficients):
        values = list(coefficients)
        while len(values) > 1 and values[-1] == 0:
            values.pop()
        self.coefficients = tuple(values or [0])

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        size = max(len(self.coefficients), len(other.coefficients))
        result = [0] * size
        for index, value in enumerate(self.coefficients):
            result[index] += value
        for index, value in enumerate(other.coefficients):
            result[index] += value
        return Polynomial(result)

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        size = len(self.coefficients) + len(other.coefficients) - 1
        result = [0] * size
        for left_power, left_value in enumerate(self.coefficients):
            for right_power, right_value in enumerate(other.coefficients):
                result[left_power + right_power] += left_value * right_value
        return Polynomial(result)

    def __call__(self, x):
        result = 0
        for value in reversed(self.coefficients):
            result = result * x + value
        return result

    def __str__(self):
        terms = []
        for power in range(len(self.coefficients) - 1, -1, -1):
            value = self.coefficients[power]
            if value == 0:
                continue
            magnitude = abs(value)
            if power == 0:
                body = str(magnitude)
            else:
                coefficient = '' if magnitude == 1 else str(magnitude)
                variable = 'x' if power == 1 else f'x^{power}'
                body = coefficient + variable
            if not terms:
                prefix = '-' if value < 0 else ''
            else:
                prefix = ' - ' if value < 0 else ' + '
            terms.append(prefix + body)
        return ''.join(terms) or '0'
