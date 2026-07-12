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


def test_group_segments_to_lemmas_nested_proof():
    # Test case where a proof block has an intermediate "by" and ends with "qed"
    seg_map = {
        0: {"keyword": "lemma", "line": 1, "offset": 0, "theory": "Test", "file": "Test.thy"},
        2: {"keyword": "proof", "line": 2, "offset": 30, "theory": "Test", "file": "Test.thy"},
        4: {"keyword": "case", "line": 3, "offset": 50, "theory": "Test", "file": "Test.thy"},
        6: {"keyword": "by", "line": 4, "offset": 70, "theory": "Test", "file": "Test.thy"},  # intermediate by
        8: {"keyword": "next", "line": 5, "offset": 90, "theory": "Test", "file": "Test.thy"},
        10: {"keyword": "case", "line": 6, "offset": 110, "theory": "Test", "file": "Test.thy"},
        12: {"keyword": "by", "line": 7, "offset": 130, "theory": "Test", "file": "Test.thy"},  # intermediate by
        14: {"keyword": "qed", "line": 8, "offset": 150, "theory": "Test", "file": "Test.thy"},  # terminal qed
        16: {"keyword": "lemma", "line": 9, "offset": 170, "theory": "Test", "file": "Test.thy"},  # next lemma
        18: {"keyword": "by", "line": 10, "offset": 200, "theory": "Test", "file": "Test.thy"},
    }
    segments = {
        0: 'lemma test_induct: "P xs"',
        2: 'proof (induction xs)',
        4: '  case Nil',
        6: '  then show ?case by simp',
        8: 'next',
        10: '  case (Cons x xs)',
        12: '  then show ?case by simp',
        14: 'qed',
        16: 'lemma test_simple: "A"',
        18: 'by simp',
    }
    
    units = group_segments_to_lemmas(seg_map, segments)
    assert len(units) == 2
    
    # Check that test_induct includes the entire proof from "proof" to "qed"
    unit1 = units[0]
    assert unit1["name"] == "test_induct"
    assert "proof (induction xs)" in unit1["proof_text"]
    assert "qed" in unit1["proof_text"]
    assert len(unit1["skeleton_segments"]) == 5
    assert len(unit1["tactic_segments"]) == 2
    
    # Check that the second lemma is parsed correctly
    unit2 = units[1]
    assert unit2["name"] == "test_simple"
    assert unit2["proof_text"] == "by simp"


def test_format_aspect_with_metadata():
    from edel.il.aspects import format_aspect_with_metadata
    
    # 1. 0-Simplex (all four equal)
    aspect_text_dict = {
        "problem": "definition test_def",
        "method": "definition test_def",
        "finding": "definition test_def",
        "interpretation": "definition test_def",
    }
    # Check that they all format to the exact same "Statement" label
    f_p = format_aspect_with_metadata("TestTheory", "TestTheory.test_def", "problem", aspect_text_dict)
    f_m = format_aspect_with_metadata("TestTheory", "TestTheory.test_def", "method", aspect_text_dict)
    f_f = format_aspect_with_metadata("TestTheory", "TestTheory.test_def", "finding", aspect_text_dict)
    f_i = format_aspect_with_metadata("TestTheory", "TestTheory.test_def", "interpretation", aspect_text_dict)
    
    assert f_p == f_m == f_f == f_i
    assert "Statement:\ndefinition test_def" in f_p
    assert "Lemma: test_def" in f_p
    
    # 2. 2-Simplex (M=F)
    aspect_text_dict_2 = {
        "problem": "A ==> B",
        "method": "by simp",
        "finding": "by simp",
        "interpretation": "B",
    }
    f_m_2 = format_aspect_with_metadata("TestTheory", "TestTheory.test_lemma", "method", aspect_text_dict_2)
    f_f_2 = format_aspect_with_metadata("TestTheory", "TestTheory.test_lemma", "finding", aspect_text_dict_2)
    assert f_m_2 == f_f_2
    assert "Proof:\nby simp" in f_m_2
    assert "Lemma: test_lemma" in f_m_2


def test_extract_aspects_cartouches():
    from edel.il.aspects import extract_aspects
    
    lemma = {
        "statement_text": "lemma compact_img_set_of: fixes X :: ‹real interval› assumes ‹continuous_on (set_of X) f› shows ‹compact (f ` set_of X)›",
        "proof_text": "by simp",
        "keyword": "lemma"
    }
    
    aspects = extract_aspects(lemma)
    assert aspects["aspect_statement"] == "continuous_on (set_of X) f"
    assert aspects["aspect_context"] == "compact (f ` set_of X)"



