from noorix.evaluation import character_error_counts, edit_distance, error_rate, normalize_text, word_error_counts


def test_edit_distance_handles_strings_and_sequences():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1


def test_normalize_text_only_collapses_whitespace():
    assert normalize_text("  Hello\n  World  ") == "Hello World"
    assert normalize_text("Hello, WORLD") == "Hello, WORLD"


def test_character_error_counts_ignores_whitespace_only():
    assert character_error_counts("hello world", "helloworld") == (0, 10)
    assert character_error_counts("cat", "cut") == (1, 3)


def test_word_error_counts_uses_whitespace_tokens():
    assert word_error_counts("one two three", "one three") == (1, 3)


def test_error_rate_handles_empty_reference():
    assert error_rate(2, 4) == 0.5
    assert error_rate(0, 0) is None
