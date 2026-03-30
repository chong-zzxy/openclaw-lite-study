"""
Web 搜索工具 — 使用 requests 请求 DuckDuckGo HTML 版本并解析结果。

DuckDuckGo 的 HTML 版本（html.duckduckgo.com）不依赖 JS 渲染，
纯 HTTP 请求即可获取搜索结果，兼容性好。
"""

import re
import requests
from urllib.parse import quote_plus
from tools.registry import ToolDefinition, ToolResult

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _parse_ddg_results(html: str, max_results: int) -> list[dict]:
    """从 DuckDuckGo HTML 中提取搜索结果"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select(".result__body")[:max_results]:
            title_el = item.select_one(".result__a")
            snippet_el = item.select_one(".result__snippet")
            title = title_el.get_text(strip=True) if title_el else ""
            link = title_el.get("href", "") if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if title:
                results.append({"title": title, "link": link, "snippet": snippet})
        return results
    except ImportError:
        return _fallback_parse_ddg(html, max_results)


def _fallback_parse_ddg(html: str, max_results: int) -> list[dict]:
    """bs4 不可用时的正则兜底"""
    results = []
    pattern = r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.+?)</a>'
    for link, title_html in re.findall(pattern, html)[:max_results]:
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if title:
            results.append({"title": title, "link": link, "snippet": ""})
    return results


def web_search(query: str, max_results: int = 5) -> ToolResult:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        results = _parse_ddg_results(resp.text, max_results)

        if not results:
            return ToolResult(True, f"搜索 '{query}' 未找到结果。")

        lines = []
        for i, r in enumerate(results):
            entry = f"{i+1}. {r['title']}\n   {r['link']}"
            if r["snippet"]:
                entry += f"\n   {r['snippet']}"
            lines.append(entry)

        output = f"搜索 '{query}' 的结果：\n\n" + "\n\n".join(lines)
        return ToolResult(True, output)

    except requests.RequestException as e:
        return ToolResult(False, "", f"搜索请求失败: {e}")
    except Exception as e:
        return ToolResult(False, "", f"搜索失败: {e}")


def create_web_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description="Search the web using DuckDuckGo.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        handler=web_search,
    )
