from __future__ import annotations
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class NewsItem:
    def __init__(
        self, title: str, link: str, source: str, published: str, snippet: str = ""
    ):
        self.title = title
        self.link = link
        self.source = source
        self.published = published
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "link": self.link,
            "source": self.source,
            "published": self.published,
            "snippet": self.snippet,
        }


class NewsFetcher:
    RSS_FEEDS = {
        "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
        "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories",
        "reuters": "https://www.reutersagency.com/feed/?best-regions=americas&post_type=best",
        "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._cache: dict[str, list[NewsItem]] = {}
        self._cache_duration_minutes = 15

    def fetch_for_symbol(self, symbol: str, limit: int = 5) -> list[NewsItem]:
        cache_key = f"{symbol.upper()}_{limit}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        news_items = []

        feed_key = f"yahoo_finance_{symbol.upper()}"
        feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        items = self._fetch_rss(feed_url, symbol)
        news_items.extend(items)

        if len(news_items) < limit:
            general_items = self._fetch_rss(self.RSS_FEEDS["yahoo_finance"], symbol)
            news_items.extend(general_items)

        unique_items = self._deduplicate(news_items)

        self._cache[cache_key] = unique_items[:limit]

        return self._cache[cache_key]

    def fetch_multiple(
        self, symbols: list[str], limit_per_symbol: int = 3
    ) -> dict[str, list[NewsItem]]:
        results = {}

        for symbol in symbols:
            results[symbol.upper()] = self.fetch_for_symbol(symbol, limit_per_symbol)

        return results

    def _fetch_rss(self, url: str, context_symbol: str = "") -> list[NewsItem]:
        try:
            import xml.etree.ElementTree as ET
            import urllib.request
            import certifi
            import ssl

            ctx = ssl.create_default_context(cafile=certifi.where())
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                content = response.read()

            root = ET.fromstring(content)

            items = []
            namespace = {"atom": "http://www.w3.org/2005/Atom"}

            for item in root.findall(".//item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                desc_elem = item.find("description")
                pub_date_elem = item.find("pubDate")

                title = title_elem.text if title_elem is not None else "No title"
                link = link_elem.text if link_elem is not None else ""
                desc = desc_elem.text if desc_elem is not None else ""
                pub_date = pub_date_elem.text if pub_date_elem is not None else ""

                if title and title != "No title":
                    items.append(
                        NewsItem(
                            title=self._clean_html(title),
                            link=link,
                            source=self._extract_source(url),
                            published=self._parse_date(pub_date),
                            snippet=self._clean_html(desc[:200]) if desc else "",
                        )
                    )

            return items

        except ImportError:
            logger.warning("feedparser not available, using basic XML parsing")
            return []
        except Exception as e:
            logger.warning(f"Error fetching RSS {url}: {e}")
            return []

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        seen = set()
        unique = []

        for item in items:
            key = hashlib.md5(item.title.encode()).hexdigest()[:16]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique

    def _clean_html(self, text: str) -> str:
        if not text:
            return ""

        import re

        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _extract_source(self, url: str) -> str:
        if "yahoo" in url.lower():
            return "Yahoo Finance"
        elif "marketwatch" in url.lower():
            return "MarketWatch"
        elif "reuters" in url.lower():
            return "Reuters"
        elif "cnbc" in url.lower():
            return "CNBC"
        else:
            return "Unknown"

    def _parse_date(self, date_str: str) -> str:
        if not date_str:
            return ""

        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(date_str)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return date_str[:16] if len(date_str) > 16 else date_str

    def get_market_news(self, limit: int = 10) -> list[NewsItem]:
        items = self._fetch_rss(self.RSS_FEEDS["yahoo_finance"])
        return items[:limit]

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("News cache cleared")
