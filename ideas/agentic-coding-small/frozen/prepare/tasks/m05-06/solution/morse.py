CODES = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..",
    "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}


def normalize_text(text: str) -> list[str]:
    """Validate before uppercasing so Unicode case expansion is rejected."""
    for character in text:
        ascii_letter = ("A" <= character <= "Z" or "a" <= character <= "z")
        ascii_digit = "0" <= character <= "9"
        if not (ascii_letter or ascii_digit or character.isspace()):
            raise ValueError("unsupported plaintext character")
    return text.upper().split()


def split_morse(message: str) -> list[list[str]]:
    """Parse canonical spacing separately from alphabet lookup.

    Splitting with explicit separators retains empty tokens. Those expose
    extra spaces and empty words rather than silently normalizing them.
    """
    if message == "":
        return []
    words = []
    for word in message.split(" / "):
        tokens = word.split(" ")
        for token in tokens:
            if not token:
                raise ValueError("empty code token")
            if any(character not in ".-" for character in token):
                raise ValueError("invalid Morse character or separator")
        words.append(tokens)
    return words


class MorseCodec:
    """Encode text and strictly decode canonical Morse wire messages."""

    def __init__(self):
        self._reverse = {code: letter for letter, code in CODES.items()}

    def encode(self, text: str) -> str:
        """Normalize plaintext whitespace and emit canonical separators."""
        words = normalize_text(text)
        encoded = []
        for word in words:
            tokens = [CODES[letter] for letter in word]
            encoded.append(" ".join(tokens))
        return " / ".join(encoded)

    def decode(self, message: str) -> str:
        """Require every syntactically valid code to belong to the alphabet."""
        words = split_morse(message)
        decoded = []
        for tokens in words:
            letters = []
            for token in tokens:
                if token not in self._reverse:
                    raise ValueError("unknown Morse code")
                letters.append(self._reverse[token])
            decoded.append("".join(letters))
        return " ".join(decoded)
