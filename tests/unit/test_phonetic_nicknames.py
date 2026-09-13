from __future__ import annotations

import re
import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.phonetic import soundex
from probabilistic_record_linkage.linkage.vocab import World

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Robert", "R163"),
        ("Rupert", "R163"),
        ("Rubin", "R150"),
        ("Ashcraft", "A261"),
        ("Ashcroft", "A261"),
        ("Tymczak", "T522"),
        ("Pfister", "P236"),
        ("Honeyman", "H555"),
        ("Gutierrez", "G362"),
        ("Jackson", "J250"),
        ("Lee", "L000"),
    ],
)
def test_soundex_matches_the_published_coding_rules(name: str, code: str) -> None:
    assert soundex(name) == code


@given(st.text(max_size=30))
def test_soundex_is_empty_or_a_letter_and_three_digits(text: str) -> None:
    code = soundex(text)
    if any("a" <= char <= "z" for char in text.lower()):
        assert re.fullmatch(r"[A-Z][0-6]{3}", code)
    else:
        assert code == ""


@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
def test_soundex_ignores_case(text: str) -> None:
    assert soundex(text) == soundex(text.swapcase())


def test_nickname_reverse_index_and_json_round_trip() -> None:
    table = NicknameTable({"brandor": ("bran", "brandy"), "branwen": ("bran",)})
    assert table.canonical_names("bran") == ("brandor", "branwen")
    assert table.canonical_names("brandor") == ("brandor",)
    assert table.canonical_names("unknown") == ()
    assert table.nicknames_of("branwen") == ("bran",)
    restored = NicknameTable.from_json(table.to_json())
    assert restored == table
    assert restored.sha256() == table.sha256()


@pytest.mark.parametrize("text", ["[]", '{"a": [1]}', '{"a": "b"}'])
def test_malformed_nickname_files_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="nickname"):
        NicknameTable.from_json(text)


def test_a_nickname_cannot_equal_its_canonical_name() -> None:
    with pytest.raises(ValueError, match="differ"):
        NicknameTable({"anna": ("anna",)})


def test_derived_nicknames_never_collide_with_vocabulary_words(world: World) -> None:
    first_names = {
        name for group in world.groups for name in (*group.female_first, *group.male_first)
    }
    surnames = {name for group in world.groups for name in group.surnames}
    table = world.nicknames
    assert table.nicknames
    for canonical, nicks in table.nicknames.items():
        assert canonical in first_names
        for nick in nicks:
            assert nick != canonical
            assert nick not in first_names
            assert nick not in surnames
