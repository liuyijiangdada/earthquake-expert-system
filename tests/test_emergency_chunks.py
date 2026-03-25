import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_load_emergency_chunks_count_and_ids():
    from rag.emergency_rag import load_emergency_chunks

    path = os.path.join(ROOT, "data", "emergency_knowledge.json")
    chunks = load_emergency_chunks(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(chunks) == len(data["topics"])
    ids = {c["topic_id"] for c in chunks}
    assert "em_indoor" in ids and "em_kit" in ids
    assert all(c["text"] and c["title"] for c in chunks)
