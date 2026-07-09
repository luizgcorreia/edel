"""Unit tests for the AFP metadata parser."""

from edel.il.metadata import AFPMetadataParser, find_afp_metadata_dir

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

def test_load_entry_metadata_variants():
    parser = AFPMetadataParser()
    # Test underscore to hyphen resolution (AVL_Trees -> AVL-Trees)
    meta = parser.load_entry_metadata("AVL_Trees")
    assert meta
    assert meta["title"] == "AVL Trees"
    
    # Test lowercase resolution (avl_trees -> AVL-Trees)
    meta_lower = parser.load_entry_metadata("avl_trees")
    assert meta_lower
    assert meta_lower["title"] == "AVL Trees"

    # Test non-existent entry returns empty dict
    meta_none = parser.load_entry_metadata("Non_Existent_Entry_Name")
    assert meta_none == {}
