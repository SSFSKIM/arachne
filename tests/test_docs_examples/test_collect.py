from __future__ import annotations

from .collect import EXCLUDED, _blocks_in_text, collect, sample_indices

# Every ``.. manim::`` block Sphinx renders, in docstrings and in docs/source.
EXPECTED_BLOCKS = 487
EXPECTED_STILL = 294
EXPECTED_EXCLUDED = 8


def test_collector_finds_every_block():
    examples = collect()
    assert len(examples) == EXPECTED_BLOCKS
    assert len({example.name for example in examples}) == EXPECTED_BLOCKS


def test_exclusions_are_named_blocks_with_reasons():
    names = {example.name for example in collect()}
    assert set(EXCLUDED) <= names
    assert len(EXCLUDED) == EXPECTED_EXCLUDED
    assert all(reason.strip() for reason in EXCLUDED.values())


def test_examples_split_into_still_and_animated():
    examples = collect()
    still = [example for example in examples if example.still]
    animated = [example for example in examples if not example.still]
    assert len(still) == EXPECTED_STILL
    assert len(still) + len(animated) == EXPECTED_BLOCKS


def test_animated_example_compares_four_spread_frames_and_its_last():
    assert sample_indices(100) == [0, 25, 50, 75, 99]
    assert sample_indices(7) == [0, 1, 3, 5, 6]
    assert sample_indices(1) == [0, 0, 0, 0, 0]


def test_literal_blocks_are_not_examples():
    text = "\n".join(
        [
            "Write it like this::",
            "",
            "    .. manim:: Shown",
            "",
            "        class Shown(Scene): ...",
            "",
            ".. code-block:: rst",
            "",
            "    .. manim:: AlsoShown",
            "",
            ".. note::",
            "",
            "    .. manim:: Rendered",
            "        :save_last_frame:",
            "",
            "        class Rendered(Scene):",
            "            def construct(self):",
            "                self.add(Dot())",
        ]
    )
    blocks = _blocks_in_text(text, "doc")
    assert [(scene, still) for scene, _, still, _ in blocks] == [("Rendered", True)]
    assert blocks[0][3] == (
        "class Rendered(Scene):\n    def construct(self):\n        self.add(Dot())\n"
    )


def test_directive_may_have_one_space_before_its_colons():
    text = ".. manim :: Spaced\n\n    class Spaced(Scene): ...\n"
    assert [scene for scene, *_ in _blocks_in_text(text, "doc")] == ["Spaced"]


def test_doctest_blocks_keep_only_their_code():
    text = "\n".join(
        [
            ".. manim:: Doctested",
            "",
            "    >>> dot = Dot()",
            "    >>> dot.color",
            "    ManimColor('#FFFFFF')",
            "    >>> class Doctested(Scene):",
            "    ...     def construct(self):",
            "    ...         self.add(dot)",
        ]
    )
    [(scene, _, still, code)] = _blocks_in_text(text, "doc")
    assert (scene, still) == ("Doctested", False)
    assert code == (
        "dot = Dot()\ndot.color\nclass Doctested(Scene):\n"
        "    def construct(self):\n        self.add(dot)\n"
    )
