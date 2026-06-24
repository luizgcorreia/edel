"""Unit tests for the AFP metadata parser."""

from edel.isabelle.metadata import AFPMetadataParser, find_afp_metadata_dir

def test_find_afp_metadata_dir():
    metadata_dir = find_afp_metadata_dir()
    assert metadata_dir is not None
    assert metadata_dir.exists()
    assert metadata_dir.is_dir()

def test_load_entry_metadata():
    parser = AFPMetadataParser()
    meta = parser.load_entry_metadata("Multiset_Ordering_NPC")
    assert meta
    assert meta["title"] == "The Generalized Multiset Ordering is NP-Complete"
    assert "Logic/Rewriting" in meta["topics"]
    assert "NP-complete" in meta["abstract"]
    assert "thiemann" in meta["authors"]
