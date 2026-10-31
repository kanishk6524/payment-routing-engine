"""Open the hosted dashboard in a real browser so Streamlit counts a visit (and wakes it if asleep)."""
import sys

from playwright.sync_api import sync_playwright

URL = "https://payment-routing-engine-jxgj9t7um2zmusnu2u6cgw.streamlit.app"

with sync_playwright() as p:
    page = p.chromium.launch().new_page()
    page.goto(URL, timeout=120_000)
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        print("app was asleep; waking it")
        wake.click()
    try:
        # the app renders inside an iframe; wait for the title text
        page.frame_locator("iframe").first.get_by_text("Payment Routing").first.wait_for(timeout=150_000)
        print("dashboard is up")
    except Exception as exc:
        print("could not confirm dashboard:", exc)
        sys.exit(1)
