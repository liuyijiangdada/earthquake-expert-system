import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EMBED_TESTS") != "1",
    reason="set RUN_LIVE_EMBED_TESTS=1 to run embedding integration test",
)


def test_build_rag_live():
    from config.config import Config
    from rag.emergency_rag import build_emergency_rag_from_config

    rag = build_emergency_rag_from_config(Config())
    assert rag is not None
    hits = rag.search("室内地震怎么躲", top_k=3)
    assert any(h["topic_id"] == "em_indoor" for h in hits)
