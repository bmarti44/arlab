CODES = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.'}

def normalize_text(text: str) -> list[str]:
    """Validate before uppercasing so Unicode case expansion is rejected."""
    raise NotImplementedError()

def split_morse(message: str) -> list[list[str]]:
    """Parse canonical spacing separately from alphabet lookup.

    Splitting with explicit separators retains empty tokens. Those expose
    extra spaces and empty words rather than silently normalizing them.
    """
    raise NotImplementedError()

class MorseCodec:
    """Encode text and strictly decode canonical Morse wire messages."""

    def __init__(self):
        raise NotImplementedError()

    def encode(self, text: str) -> str:
        """Normalize plaintext whitespace and emit canonical separators."""
        raise NotImplementedError()

    def decode(self, message: str) -> str:
        """Require every syntactically valid code to belong to the alphabet."""
        raise NotImplementedError()
