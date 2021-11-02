""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "abcdefghijklmnopqrstuvwxyzäöü"
_silences = ["@sp", "@spn", "@sil"]
_phonemes = ['b', 'E0', 'y0', 'a1', 'U0', '+', 'B1', 't', 'I0', 'i1', 'h', '/1', 'g', 'O1', 'd', 'q1', 'q0', '&1', 'k', '|0', 'j', '=', 'B0', 'v', 'o1', 'a0', 'Y0', 'z', 'Y1', 'e1', '&0', 'E1', '~1', 's', 'e0', 'Z', '/0', 'l', 'm', 'W0', 'I1', 'r', 'S', 'x', 'i0', 'X0', 'p', '@0', 'n', 'f', 'O0', 'y1', 'null1', 'U1', 'X1', 'u0', 'J', ')1', ')0', 'N', '|1', 'u1', 'o0', 'W1']

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
# _pinyin = ["@" + s for s in pinyin.valid_symbols]
_prepended_phonemes = ["@" + s for s in _phonemes]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
#    + _pinyin
    + _phonemes
    + _silences
    + _prepended_phonemes
)
