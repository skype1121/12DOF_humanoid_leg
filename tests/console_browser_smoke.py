"""Playwright smoke test. Refuses any console that is not explicitly in DEMO.

Start operator_console.py --demo separately, then run this file. Screenshots are
written to a new temporary directory. No browser dependency is needed on Jetson.
"""
import argparse
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18800")
    args = parser.parse_args()
    with urlopen(args.url + "/api/state", timeout=3) as response:
        assert json.load(response).get("demo") is True, "REFUSED: not a demo console"
    output = Path(tempfile.mkdtemp(prefix="console-browser-"))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("state?.demo === true")

        def action(key, selector=None):
            page.wait_for_function("k => state.trial.allowed[k].enabled", arg=key)
            page.locator(selector or f'#operate [data-action="{key}"]').click()
            if key in ("hang_stand", "gain_ramp", "ground", "handoff", "arm", "stand", "support_confirm", "release", "op_slack_done", "new_trial"):
                page.locator("#accept").click()

        def flow():
            for key in ("can_up", "sensors", "boot_pose", "selftest", "stage8", "hang_stand", "gain_ramp", "ground", "handoff", "baseline", "arm", "bridge", "stand"):
                mode = page.evaluate("state.trial.mode")
                if key == "ground" and mode == "ground":
                    action("op_touchdown")
                    page.wait_for_function("state.foot.contact?.every(Boolean)")
                page.wait_for_function("k => state.trial.next_action === k && state.trial.allowed[k].enabled", arg=key)
                action(key, "#next")
                page.wait_for_function("k => state.trial.completed.includes(k)", arg=key)

        flow()
        assert page.evaluate("state.trial.measurement.state") == "waiting"
        action("op_slack_done")
        page.wait_for_function("state.trial.measurement.elapsed >= 5", timeout=10000)
        page.evaluate("scrollTo(0,0)")
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        before = page.evaluate("state.trial.measurement.elapsed")
        page.reload()
        page.wait_for_function("state?.trial.measurement.state === 'measuring'")
        assert page.evaluate("state.trial.measurement.elapsed") >= before
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        action("op_assist")
        page.wait_for_function("state.trial.measurement.state === 'interrupted'")
        page.route("**/api/state", lambda route: route.abort())
        page.wait_for_function("connected === false")
        assert page.locator("#next").is_disabled()
        assert page.locator("#estop").is_enabled()
        assert page.locator("#leftContact").inner_text() == "미확인"
        page.unroute("**/api/state")
        page.wait_for_function("connected === true")
        page.locator("#estop").click()
        page.wait_for_function("state.trial.stop?.confirmed === true")
        for key in ("support_confirm", "stack_down", "release"):
            action(key)
            page.wait_for_function("k => k === 'support_confirm' ? state.trial.support_confirmed : state.trial.completed.includes(k)", arg=key)
        page.locator('[data-tab="review"]').click()
        action("session_end", '#review [data-action="session_end"]')
        page.wait_for_function("state.session.active === false")
        action("new_trial", '#review [data-action="new_trial"]')
        page.wait_for_function("state.trial.number === 2")
        page.locator('[data-tab="setup"]').click()
        page.locator("#mode").select_option("air")
        action("configure", '[data-action="configure"]')
        page.wait_for_function("state.trial.mode === 'air'")
        page.locator('[data-tab="operate"]').click()
        flow()
        assert page.locator('#operate [data-action="op_slack_done"]').is_disabled()
        action("op_touchdown")
        page.wait_for_function("state.foot.contact.every(Boolean)")
        action("op_slack_done")
        page.wait_for_function("state.trial.measurement.state === 'measuring'")
        page.locator("#estop").click()
        page.wait_for_function("state.trial.measurement.state === 'interrupted'")
        assert not errors, errors
        browser.close()
    print(f"Browser smoke PASS: grounded/airborne, confirmations, reload, offline, stop, recovery, mobile; screenshots: {output}")


if __name__ == "__main__":
    main()
