import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EMBED_TESTS") != "1",
    reason="set RUN_LIVE_EMBED_TESTS=1 to run embedding integration test",
)


def test_build_rag_live():
    pytest.importorskip("pymilvus")
    from pymilvus import connections

    from config.config import Config
    from rag.emergency_rag import build_emergency_rag_from_config

    try:
        connections.connect(alias="default", host="localhost", port=19530, timeout=5)
        connections.disconnect("default")
    except Exception:
        pytest.skip("Milvus not reachable on localhost:19530 (docker compose up milvus-standalone)")

    rag = build_emergency_rag_from_config(Config())
    assert rag is not None
    hits = rag.search("室内地震怎么躲", top_k=3)
    assert any(h["topic_id"] == "em_indoor" for h in hits)
