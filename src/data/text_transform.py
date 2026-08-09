import string
from typing import List, Dict

class TextTransform:
    """
    Maps characters to integers and vice versa for CTC loss and speech recognition.
    Includes standard English alphabet, space, apostrophes, and Akan special characters (ɛ, ɔ).
    """

    def __init__(self, custom_chars: List[str] = None):
        if custom_chars is not None:
            char_list = custom_chars
        else:
            # Special tokens & Akan characters
            # Index 0: blank (or custom token), 1: space, 2: apostrophe
            special = ["<SPACE>", "'", "ɛ", "ɔ"]
            alphabet = list(string.ascii_lowercase)
            # Combine characters avoiding duplicates
            char_list = []
            for c in special + alphabet:
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
        text = text.lower()
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = '<SPACE>'
            else:
                ch = c
            if ch in self.char_map:
                int_sequence.append(self.char_map[ch])
        return int_sequence

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
