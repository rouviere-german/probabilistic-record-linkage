"""American Soundex (Russell and Odell, US patents 1261167 and 1435663; NARA coding rules).

Used as a sort key for sorted-neighbourhood blocking and, keyed with HMAC, as a blocking token in
privacy mode. Rules: keep the first letter; code the rest as b f p v -> 1, c g j k q s x z -> 2,
d t -> 3, l -> 4, m n -> 5, r -> 6; letters with the same code are written once when adjacent or
separated only by ``h`` or ``w`` (vowels and ``y`` separate them); pad or cut to four characters.
"""

from __future__ import annotations

__all__ = ["soundex"]

_CODES: dict[str, str] = {
    char: digit
    for letters, digit in (
        ("bfpv", "1"),
        ("cgjkqsxz", "2"),
        ("dt", "3"),
        ("l", "4"),
        ("mn", "5"),
        ("r", "6"),
    )
    for char in letters
}


def soundex(name: str) -> str:
    """Four-character Soundex code of ``name``; empty string when it has no ASCII letters."""
    letters = [char for char in name.lower() if "a" <= char <= "z"]
    if not letters:
        return ""
    digits: list[str] = []
    previous = _CODES.get(letters[0], "")
    for char in letters[1:]:
        code = _CODES.get(char, "")
        if code and code != previous:
            digits.append(code)
            if len(digits) == 3:
                break
        if char not in "hw":
            previous = code
    return (letters[0].upper() + "".join(digits)).ljust(4, "0")
