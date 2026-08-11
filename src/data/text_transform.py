import string
from typing import List, Dict

class TextTransform:
    """
    Maps characters to integers and vice versa for CTC loss and speech recognition.
    Includes standard English alphabet, space, apostrophes, Akan special characters
    (ɛ, ɔ) and the digits 0-9.

    Characters outside the vocabulary are dropped during encoding. Digits are
    included because corpora such as the health ASR set write dosages and dates
    numerically; dropping them would leave the audio saying a number that the
    label does not contain, which teaches the model to skip speech. Digits are
    appended last so the letter indices stay stable when the flag is toggled.
    """

    def __init__(self, custom_chars: List[str] = None, include_digits: bool = True):
        if custom_chars is not None:
            char_list = custom_chars
        else:
            # Special tokens & Akan characters
            # Index 0: space, 1: apostrophe, 2-3: Akan vowels, then a-z, then 0-9
            special = ["<SPACE>", "'", "ɛ", "ɔ"]
            alphabet = list(string.ascii_lowercase)
            digits = list(string.digits) if include_digits else []
            # Combine characters avoiding duplicates
            char_list = []
            for c in special + alphabet + digits:
                if c not in char_list:
                    char_list.append(c)

        self.char_map: Dict[str, int] = {}
        self.index_map: Dict[int, str] = {}

        for i, ch in enumerate(char_list):
            self.char_map[ch] = i
            self.index_map[i] = ch

        # Add <SPACE> mapping explicitly if space character is passed
        self.char_map[" "] = self.char_map["<SPACE>"]
        # CTC Blank token index is usually the last index or explicit
        self.blank_label = len(self.index_map)
        self.char_map["<BLANK>"] = self.blank_label
        self.index_map[self.blank_label] = "<BLANK>"

    def text_to_int(self, text: str) -> List[int]:
        """Converts a text string to a list of integer character indices."""
        text = self.normalize(text).lower()
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = '<SPACE>'
            else:
                ch = c
            if ch in self.char_map:
                int_sequence.append(self.char_map[ch])
        return int_sequence

    # Corpora typeset with smart quotes write the elision in m’abankɛseɛ with
    # U+2019, which is not the U+0027 in the vocabulary - so the apostrophe was
    # being dropped and two words silently glued together. Mapping the variants
    # onto the plain form recovers them.
    PUNCTUATION_ALIASES = {
        "’": "'",   # right single quotation mark
        "‘": "'",   # left single quotation mark
        "ʼ": "'",   # modifier letter apostrophe
        "´": "'",   # acute accent used as apostrophe
        "–": "-",   # en dash
        "—": "-",   # em dash
        " ": " ",   # non-breaking space
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """Folds typographic variants onto the characters the vocabulary holds."""
        return "".join(cls.PUNCTUATION_ALIASES.get(ch, ch) for ch in text)

    def int_to_text(self, labels: List[int]) -> str:
        """Converts a sequence of integer character indices back to string."""
        string_chars = []
        for i in labels:
            if i in self.index_map:
                ch = self.index_map[i]
                if ch == '<SPACE>':
                    string_chars.append(' ')
                elif ch != '<BLANK>':
                    string_chars.append(ch)
        return "".join(string_chars)

    def int_to_text_remove_pad(self, labels: List[int]) -> str:
        """Converts sequence of indices to string omitting CTC blanks and padding."""
        return self.int_to_text(labels)

    @property
    def vocab_size(self) -> int:
        return len(self.index_map)
