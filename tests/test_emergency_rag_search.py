import numpy as np


def test_top_k_ordering_with_stub_embedder():
    from rag.emergency_rag import EmergencyRAG

    chunks = [
        {"topic_id": "a", "title": "t1", "source": "s", "text": "alpha"},
        {"topic_id": "b", "title": "t2", "source": "s", "text": "beta gamma"},
    ]

    def embed(text: str) -> np.ndarray:
        if "beta" in text:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    rag = EmergencyRAG(chunks=chunks, embed=embed, embed_query=lambda _: q)
    hits = rag.search("anything", top_k=2)
    assert hits[0]["topic_id"] == "b"
