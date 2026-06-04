#!/usr/bin/env python3
"""政府网站 RSS/公告摘要（轻量，失败时降级为固定门户链接）。"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

_POLICY_KW = ("政策", "补贴", "重建", "补偿", "救助", "复课", "路况", "恢复", "安置")


class PolicyFeedFetcher:
    def __init__(self, config=None):
        self._enabled = True
        self._timeout = 12
        self._cache_ttl = 3600
        self._cache: Optional[List[Dict[str, str]]] = None
        self._cache_ts = 0.0
        self._feeds = [
            "https://www.mem.gov.cn/xw/yjgl/rss.xml",
            "https://www.gov.cn/xinwen/gwyw/rss.xml",
        ]
        if config:
            self._enabled = getattr(config, "LAYER3_POLICY_RSS_ENABLED", True)
            self._timeout = int(getattr(config, "LAYER3_POLICY_RSS_TIMEOUT", 12))
            self._cache_ttl = int(getattr(config, "LAYER3_POLICY_RSS_CACHE_SECONDS", 3600))
            custom = getattr(config, "LAYER3_POLICY_RSS_URLS", None)
            if custom:
                self._feeds = list(custom)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_handle(self, text: str) -> bool:
        return bool(text) and any(k in text for k in _POLICY_KW)

    def _parse_rss_xml(self, content: bytes, base_url: str) -> List[Dict[str, str]]:
        items = []
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return items

        for item in root.iter():
            if item.tag.split("}")[-1] != "item":
                continue
            title = link = ""
            for child in item:
                tag = child.tag.split("}")[-1]
                if tag == "title" and child.text:
                    title = child.text.strip()
                elif tag == "link" and child.text:
                    link = child.text.strip()
            if title:
                if link and not link.startswith("http"):
                    link = urljoin(base_url, link)
                items.append({"title": title, "link": link or base_url})
        return items

    def fetch_entries(self, force: bool = False) -> List[Dict[str, str]]:
        if not self.enabled:
            return []

        now = time.time()
        if not force and self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        merged: List[Dict[str, str]] = []
        for url in self._feeds:
            try:
                resp = requests.get(
                    url,
                    timeout=self._timeout,
                    headers={"User-Agent": "EarthquakeQA/1.0"},
                )
                if resp.status_code != 200:
                    continue
                merged.extend(self._parse_rss_xml(resp.content, url))
            except Exception as e:
                logger.debug("RSS 获取失败 %s: %s", url, e)

        if merged:
            self._cache = merged[:15]
            self._cache_ts = now
            return self._cache

        return []

    def match_for_query(self, user_text: str, max_items: int = 2) -> List[Dict[str, Any]]:
        if not self.should_handle(user_text):
            return []

        entries = self.fetch_entries()
        if not entries:
            return [{
                "id": "policy_fallback_mem",
                "type": "link",
                "url": "https://www.mem.gov.cn/",
                "caption": "应急管理部官网 — 灾后政策与通报",
                "source": "应急管理部",
                "phase": "震后",
            }]

        scored = []
        for ent in entries:
            title = ent.get("title", "")
            score = sum(1 for k in _POLICY_KW if k in user_text and k in title)
            score += sum(1 for k in _POLICY_KW if k in title) * 0.3
            if score > 0 or not scored:
                scored.append((score, ent))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, ent in scored[:max_items]:
            link = ent.get("link") or "https://www.mem.gov.cn/"
            out.append({
                "id": f"policy_{abs(hash(ent.get('title', ''))) % 10**8}",
                "type": "link",
                "url": link,
                "caption": ent.get("title", "最新政策通告")[:80],
                "source": "政府 RSS",
                "phase": "震后",
            })
        return out
