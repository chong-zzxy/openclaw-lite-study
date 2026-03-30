"""
Web 搜索工具 — 使用 requests + BeautifulSoup 解析 Bing 搜索结果。

注意：依赖 Bing 的 HTML 结构，Bing 改版可能导致解析失败。
"""

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


def _parse_bing_results(html: str, max_results: int) -> list[dict]:
    """从 Bing HTML 中提取搜索结果"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for item in soup.select("li.b_algo")[:max_results]:
        title_el = item.select_one("h2 a")
        snippet_el = item.select_one(".b_caption p")
        if not snippet_el:
            snippet_el = item.select_one(".b_lineclamp2")

        title = title_el.get_text(strip=True) if title_el else ""
        link = title_el.get("href", "") if title_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""

        if title:
            results.append({"title": title, "link": link, "snippet": snippet})

    return results


def web_search(query: str, max_results: int = 5) -> ToolResult:
    try:
        url = f"https://www.bing.com/search?q={quote_plus(query)}&ensearch=0"
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        results = _parse_bing_results(resp.text, max_results)

        if not results:
            # 可能是 bs4 没装，尝试用正则兜底
            return _fallback_parse(resp.text, query, max_results)

        lines = []
        for i, r in enumerate(results):
            lines.append(f"{i+1}. {r['title']}\n   {r['link']}\n   {r['snippet']}")

        output = f"搜索 '{query}' 的结果：\n\n" + "\n\n".join(lines)
        return ToolResult(True, output)

    except requests.RequestException as e:
        return ToolResult(False, "", f"搜索请求失败: {e}")
    except Exception as e:
        return ToolResult(False, "", f"搜索失败: {e}")


def _fallback_parse(html: str, query: str, max_results: int) -> ToolResult:
    """bs4 不可用时的正则兜底解析"""
    import re
    pattern = r'<h2><a[^>]+href="([^"]+)"[^>]*>(.+?)</a></h2>'
    matches = re.findall(pattern, html)[:max_results]

    if not matches:
        return ToolResult(True, f"搜索 '{query}' 未找到结果（建议安装 beautifulsoup4 以获得更好的解析效果）。")

    lines = []
    for i, (link, title_html) in enumerate(matches):
        title = re.sub(r'<[^>]+>', '', title_html).strip()
        lines.append(f"{i+1}. {title}\n   {link}")

    output = f"搜索 '{query}' 的结果：\n\n" + "\n\n".join(lines)
    return ToolResult(True, output)


def create_web_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description="Search the web using Bing.",
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
