"""
Web 搜索工具 — 使用 Playwright 驱动 Bing 搜索。
对应 OpenClaw: web_search (Brave Search API)

注意：依赖 Bing 的 DOM 结构（li.b_algo 等选择器），Bing 改版可能导致解析失败。
"""

from urllib.parse import quote_plus
from tools.registry import ToolDefinition, ToolResult


def web_search(query: str, max_results: int = 5) -> ToolResult:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ToolResult(
            False, "",
            "请安装 playwright: pip install playwright && playwright install chromium",
        )

    browser = None
    pw = None
    try:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(15000)

        # 用 Bing 搜索，query 做 URL 编码
        url = f"https://www.bing.com/search?q={quote_plus(query)}&ensearch=0"
        page.goto(url, wait_until="domcontentloaded")

        # 等待搜索结果加载
        page.wait_for_selector("li.b_algo", timeout=10000)

        # 提取搜索结果
        items = page.query_selector_all("li.b_algo")
        results = []
        for i, item in enumerate(items[:max_results]):
            title_el = item.query_selector("h2 a")
            snippet_el = item.query_selector(".b_caption p, .b_lineclamp2")

            title = title_el.inner_text().strip() if title_el else ""
            link = title_el.get_attribute("href") if title_el else ""
            snippet = snippet_el.inner_text().strip() if snippet_el else ""

            if title:
                results.append(f"{i+1}. {title}\n   {link}\n   {snippet}")

        if not results:
            return ToolResult(True, f"搜索 '{query}' 未找到结果。")

        output = f"搜索 '{query}' 的结果：\n\n" + "\n\n".join(results)
        return ToolResult(True, output)

    except Exception as e:
        return ToolResult(False, "", f"搜索失败: {e}")
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if pw:
            try:
                pw.stop()
            except Exception:
                pass


def create_web_search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description="Search the web using Bing via browser automation.",
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
