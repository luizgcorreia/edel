"""Unit tests for the Isabelle parser."""

from edel.il.parser import parse_source_segments, extract_lemma_name, group_segments_to_lemmas

def test_parse_source_segments():
    raw_source = (
        "   0  theory Test imports Main begin\n"
        "   2  lemma test1 [simp]: \"A & B ==> A\"\n"
        "        by simp\n"
        "   4  definition test2 where \"test2 x = x\"\n"
    )
    
    segments = parse_source_segments(raw_source)
    assert len(segments) == 3
    assert segments[0] == "theory Test imports Main begin"
    assert segments[2] == "lemma test1 [simp]: \"A & B ==> A\"\n        by simp"
    assert segments[4] == "definition test2 where \"test2 x = x\""

def test_extract_lemma_name():
    assert extract_lemma_name('lemma test1 [simp]: "A & B ==> A"') == "test1"
    assert extract_lemma_name('theorem complex_thm:\n  shows "P"') == "complex_thm"
    assert extract_lemma_name('lemma "True"') == ""

def test_group_segments_to_lemmas():
    seg_map = {
        0: {"keyword": "theory", "line": 1, "offset": 0, "theory": "Test", "file": "Test.thy"},
        2: {"keyword": "lemma", "line": 5, "offset": 50, "theory": "Test", "file": "Test.thy"},
        4: {"keyword": "by", "line": 6, "offset": 90, "theory": "Test", "file": "Test.thy"},
        6: {"keyword": "definition", "line": 8, "offset": 110, "theory": "Test", "file": "Test.thy"},
    }
    segments = {
        0: "theory Test imports Main begin",
        2: 'lemma test1 [simp]: "A & B ==> A"',
        4: "by simp",
        6: 'definition test2 where "test2 x = x"',
    }
    
    units = group_segments_to_lemmas(seg_map, segments)
    assert len(units) == 2
    
    # Check Lemma unit
    unit1 = units[0]
    assert unit1["name"] == "test1"
    assert unit1["keyword"] == "lemma"
    assert unit1["statement_text"] == 'lemma test1 [simp]: "A & B ==> A"'
    assert unit1["proof_text"] == "by simp"
    
    # Check Definition unit
    unit2 = units[1]
    assert unit2["name"] == "test2"
    assert unit2["keyword"] == "definition"
    assert unit2["statement_text"] == 'definition test2 where "test2 x = x"'
    assert unit2["proof_text"] == ""

