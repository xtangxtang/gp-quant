"""测试混合方案: DrissionPage 拿 cookie → curl_cffi 请求 API

对比:
  - DrissionPage fetch_json (浏览器导航每页): 每页 ~5-7s
  - curl_cffi + 浏览器 cookie: 预期每页 ~0.3-0.5s
"""

import json
import time
from curl_cffi import requests as cffi_requests
from DrissionPage import ChromiumPage, ChromiumOptions

XUEQIU_HOME = "https://xueqiu.com"
XUEQIU_API = "https://xueqiu.com/query/v1/symbol/search/status.json"

TEST_SYMBOLS = ["SZ000001", "SH600519", "SZ000858"]


def get_cookies_from_browser() -> dict:
    """启动 chromium 访问雪球首页, 拿到 cookie 后关闭"""
    print("[1] 启动浏览器获取 cookie ...")
    t0 = time.time()

    co = ChromiumOptions()
    co.set_browser_path("/usr/bin/chromium-browser")
    co.headless()
    co.set_argument("--no-sandbox")
    co.set_argument("--disable-dev-shm-usage")
    co.set_argument("--disable-gpu")

    page = ChromiumPage(co)
    page.get(XUEQIU_HOME)
    time.sleep(5)

    # 从浏览器提取 cookie
    cookies = {}
    for c in page.cookies():
        cookies[c["name"]] = c["value"]

    elapsed = time.time() - t0
    print(f"    浏览器耗时: {elapsed:.1f}s")
    print(f"    cookies: {list(cookies.keys())}")

    has_token = "xq_a_token" in cookies
    print(f"    xq_a_token: {'✓' if has_token else '✗'}")

    page.quit()
    return cookies


def make_cffi_session(cookies: dict) -> cffi_requests.Session:
    """用浏览器 cookie 创建 curl_cffi session"""
    s = cffi_requests.Session(impersonate="chrome")
    for k, v in cookies.items():
        s.cookies.set(k, v, domain=".xueqiu.com")
    return s


def fetch_page(s: cffi_requests.Session, symbol: str, page: int) -> dict | None:
    url = (
        f"{XUEQIU_API}?"
        f"symbol={symbol}&count=100&comment=0"
        f"&page={page}&sort=time"
    )
    t0 = time.time()
    resp = s.get(url)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"    page {page}: HTTP {resp.status_code}, {elapsed:.3f}s")
        return None

    try:
        data = resp.json()
    except Exception:
        body = resp.text[:150]
        print(f"    page {page}: JSON 失败, {elapsed:.3f}s, body: {body}")
        return None

    posts = data.get("list", [])
    print(f"    page {page}: {len(posts)} 条, {elapsed:.3f}s")
    return data


def test_stock(s: cffi_requests.Session, symbol: str, max_pages: int = 10, delay: float = 0.5):
    print(f"\n[测试] {symbol} — {max_pages} 页, 延迟 {delay}s")
    total_posts = 0
    t_start = time.time()

    for p in range(1, max_pages + 1):
        data = fetch_page(s, symbol, p)
        if data is None:
            break
        posts = data.get("list", [])
        total_posts += len(posts)
        if not posts:
            break
        if p < max_pages:
            time.sleep(delay)

    total = time.time() - t_start
    print(f"    合计: {total_posts} 条, {total:.1f}s, 均 {total / max(p, 1):.2f}s/页")


def main():
    print("=" * 60)
    print("混合方案: 浏览器 cookie + curl_cffi 请求")
    print("=" * 60)

    # 1. 浏览器拿 cookie
    cookies = get_cookies_from_browser()
    if "xq_a_token" not in cookies:
        print("❌ 未获取到 xq_a_token")
        return

    # 2. 创建 curl_cffi session
    s = make_cffi_session(cookies)

    # 3. 正常速度测试 (0.5s 延迟)
    for sym in TEST_SYMBOLS:
        test_stock(s, sym, max_pages=10, delay=0.5)

    # 4. 快速测试 (0.2s 延迟)
    print("\n" + "=" * 60)
    print("快速模式 (0.2s 延迟)")
    print("=" * 60)
    test_stock(s, "SZ000001", max_pages=20, delay=0.2)

    # 5. 极速测试 (无延迟)
    print("\n" + "=" * 60)
    print("极速模式 (0s 延迟)")
    print("=" * 60)
    test_stock(s, "SH600519", max_pages=10, delay=0.0)

    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()
