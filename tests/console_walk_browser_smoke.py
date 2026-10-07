"""Playwright smoke test for the walk panel(걷기 조작 카드). Refuses any console that is not explicitly in DEMO.

Start `operator_console.py --demo --port 18800` separately, then run this file. Screenshots go to a new temporary
directory. DEMO 의 걷기 세션은 합성이다(판정만 실제 감독기·래치 클록) — 화면·상태 표시·버튼 잠금의 확인이지 실물 검증이 아니다.

확인: 걷기 키가 아니면 패드 숨김 + 이유 · 걷기 키 선택(표시 포함) · guard_mode=data 경고 · 비활성 버튼(회색·사유 글씨·툴팁·API 직접 호출 거부)
· 활성 버튼의 시뮬 참고 이동량 표기 · 확인 창(취소하면 아무것도 나가지 않음) · 수락됨 → 서기 복귀 확인됨 · 걷는 중 버튼 잠금
· 새로고침 뒤 같은 요청 상태 유지 · 정지 래치 → 재무장 · 걷기 끄기 · 정지 요청 뒤 걷기 불가(끄기는 가능) · 폰 폭 가로 넘침 없음.
"""
import argparse
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
from playwright.sync_api import sync_playwright

FLOW = ("can_up", "sensors", "boot_pose", "selftest", "stage8", "hang_stand", "gain_ramp", "ground", "handoff", "baseline", "arm", "bridge", "stand")
CONFIRM = ("hang_stand", "gain_ramp", "ground", "handoff", "arm", "stand", "support_confirm", "release", "op_slack_done", "new_trial")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18800")
    parser.add_argument("--key", default="stand_kew3_g21a_g24ra_w3s3d15800", help="걷기 실험 키(합성 또는 W3 단독)")
    args = parser.parse_args()
    with urlopen(args.url + "/api/state", timeout=3) as response:
        first = json.load(response)
    assert first.get("demo") is True, "REFUSED: not a demo console"
    assert "walk" in first and first["walk"]["module_error"] is None, first.get("walk")
    output = Path(tempfile.mkdtemp(prefix="console-walk-browser-"))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1300}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("state?.demo === true")

        def action(key, selector=None):
            page.wait_for_function("k => state.trial.allowed[k].enabled", arg=key)
            page.locator(selector or f'#operate [data-action="{key}"]').click()
            if key in CONFIRM:
                page.locator("#accept").click()

        def stage():
            return page.evaluate("state.walk.current?.stage ?? null")

        def wait_stage(*names, timeout=9000):
            page.wait_for_function("names => names.includes(state.walk.current?.stage)", arg=list(names), timeout=timeout)

        def walk_click(name, accept=True):
            page.wait_for_function("n => state.walk.buttons.find(b => b.name === n)?.enabled === true", arg=name, timeout=9000)
            page.locator(f'[data-walk="{name}"]').click()
            assert page.locator("#confirmDialog").is_visible()
            page.locator("#accept" if accept else "#cancel").click()

        # 1) 걷기 키가 아닌 기본 시험: 패드는 접혀 있고 이유가 보인다
        assert page.locator("#walkCard").is_visible() and not page.locator("#walkBody").is_visible()
        assert "걷기 세션 아님" in page.locator("#walkMode").inner_text()
        assert "걷기 실험 키" in page.locator("#walkReason").inner_text(), page.locator("#walkReason").inner_text()
        # 2) 걷기 실험 키 선택(표시가 붙어 있다) → 기존 단계별 절차로 정책 시작
        page.locator('[data-tab="setup"]').click()
        option = page.locator(f'#ckpt option[value="{args.key}"]')
        assert option.count() == 1 and "걷기 실험" in option.inner_text(), option.inner_text()
        page.locator("#ckpt").select_option(args.key)
        action("configure", '[data-action="configure"]')
        page.wait_for_function("k => state.trial.checkpoint === k", arg=args.key)
        page.locator('[data-tab="operate"]').click()
        page.wait_for_function("state.walk.session === true")
        assert page.locator("#walkBody").is_visible() and page.locator('[data-walk="fwd1"]').is_disabled()
        assert "정책 실행 단계가 아닙니다" in page.locator("#walkReason").inner_text(), page.locator("#walkReason").inner_text()
        for key in FLOW:
            if key == "ground":
                action("op_touchdown")
                page.wait_for_function("state.foot.contact?.every(Boolean)")
            page.wait_for_function("k => state.trial.next_action === k && state.trial.allowed[k].enabled", arg=key)
            action(key, "#next")
            page.wait_for_function("k => state.trial.completed.includes(k)", arg=key)
        page.wait_for_function("state.walk.available === true", timeout=9000)
        walk = page.evaluate("state.walk")
        composite = walk["composite"]
        # 3) 고정 경고·버튼 표시(비활성·참고 이동량)
        assert page.locator("#walkGuard").is_visible() and "guard_mode=data" in page.locator("#walkGuard").inner_text()
        assert "현장 확인 전" in page.locator("#walkSign").inner_text()
        off = [b for b in walk["buttons"] if b["allowed"] is False]
        on = [b for b in walk["buttons"] if b["allowed"]]
        assert len(walk["buttons"]) == page.locator("[data-walk]").count() and len(on) >= 3, walk["buttons"]
        for b in on:
            button = page.locator(f'[data-walk="{b["name"]}"]')
            assert button.is_enabled() and b["label"] in button.inner_text(), b
            if b["sim_ref"]:
                assert f'시뮬 {b["sim_ref"]}' in button.inner_text(), button.inner_text()
        if any(b["sim_ref"] for b in on):
            assert "실물 미확인" in page.locator("#walkRefNote").inner_text(), page.locator("#walkRefNote").inner_text()
        for b in off:                                   # 시뮬 평가에서 비활성: 회색(disabled + walkOff)·사유가 툴팁과 패드 아래 글씨에
            button = page.locator(f'[data-walk="{b["name"]}"]')
            assert button.is_disabled() and "walkOff" in (button.get_attribute("class") or ""), b
            assert b["why_disabled"] in (button.get_attribute("title") or ""), button.get_attribute("title")
            assert "비활성" in button.inner_text() and b["why_disabled"] in page.locator("#walkOffNote").inner_text()
            result = page.evaluate("""async name => {const r = await fetch('/api/action', {method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'w3_move', move: name, confirmed: true, seen: state.server_time, boot_id: state.trial.boot_id,
                revision: state.trial.revision, request_id: crypto.randomUUID()})}); return [r.status, await r.json()]}""", b["name"])
            assert result[0] == 409 and "비활성" in result[1]["message"] and b["why_disabled"] in result[1]["message"], result
        assert page.locator("#walkOffNote").is_visible() is bool(off)
        assert page.evaluate("state.walk.current") is None
        page.screenshot(path=str(output / "walk_ready.png"), full_page=True)
        # 4) 확인 창: 취소하면 아무것도 나가지 않는다 → 확인하면 수락됨 → (걷는 중 잠금) → 서기 복귀 확인됨
        walk_click("fwd1", accept=False)
        page.wait_for_timeout(700)
        assert page.evaluate("state.walk.current") is None
        page.locator('[data-walk="fwd1"]').click()
        assert "전진 한 걸음" in page.locator("#confirmTitle").inner_text()
        text = page.locator("#confirmText").inner_text()
        assert "스스로 끕니다" in text and "현장 확인 전" in text, text
        page.locator("#accept").click()
        wait_stage("walking", "ending", "done")
        assert all(page.locator(f'[data-walk="{b["name"]}"]').is_disabled() for b in walk["buttons"]) or stage() == "done"
        wait_stage("done")
        assert "서기 복귀 확인됨" in page.locator("#walkStage").inner_text(), page.locator("#walkStage").inner_text()
        assert page.evaluate("state.walk.bridge.standing") is True and (page.evaluate("state.walk.mode") == "ke") is composite
        first_req = page.evaluate("state.walk.current.req_id")
        # 5) 새로고침: 걷는 중에 다시 불러도 같은 요청의 상태가 그대로 보이고 버튼은 잠겨 있다(상태는 서버에 있다)
        walk_click("fwd3")
        wait_stage("walking")
        req = page.evaluate("state.walk.current.req_id")
        assert req != first_req
        page.reload()
        page.wait_for_function("state?.walk?.current != null")
        assert page.evaluate("state.walk.current.req_id") == req and stage() in ("walking", "ending", "done")
        if stage() != "done":
            assert page.locator('[data-walk="fwd1"]').is_disabled()
        page.screenshot(path=str(output / "walk_reloaded.png"), full_page=True)
        wait_stage("done")
        assert page.evaluate("state.walk.current.req_id") == req
        # 6) 정지 래치 → 걷기 버튼 잠금 + 이유 → 재무장(확인 창) → 다시 가능
        page.wait_for_function("state.trial.allowed.w3_stop.enabled")
        page.locator('[data-action="w3_stop"]').click()
        page.wait_for_function("state.walk.bridge.stop_latched === true")
        assert page.locator('[data-walk="fwd1"]').is_disabled() and "정지 래치" in page.locator("#walkReason").inner_text()
        assert "정지 래치" in page.locator("#walkBridge").inner_text()
        page.wait_for_function("state.trial.allowed.w3_rearm.enabled")
        page.locator('[data-action="w3_rearm"]').click()
        assert "재무장" in page.locator("#confirmText").inner_text()
        page.locator("#accept").click()
        page.wait_for_function("state.walk.bridge.stop_latched === false")
        # 7) 걷는 중 '걷기 끄기(서기)' → 걷기 끝 요청됨(운영자 끄기) → 서기 복귀 확인됨
        walk_click("turn_l")
        wait_stage("walking")
        page.locator('[data-action="w3_off"]').click()
        page.wait_for_function("state.walk.current.end_by === 'off' || state.walk.current.stage === 'done'")
        wait_stage("done")
        assert page.evaluate("state.walk.current.end_by") == "off"
        # 8) 폰 폭: 가로 넘침 없음
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#walkCard").scroll_into_view_if_needed()
        page.screenshot(path=str(output / "walk_mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1300})
        # 9) 정지 요청: 걷는 중이면 '중단' — 그 뒤 걷기 버튼·재무장은 잠기고 걷기 끄기는 남는다
        walk_click("fwd3")
        wait_stage("walking")
        page.locator("#estop").click()
        wait_stage("interrupted")
        assert "중단됨" in page.locator("#walkStage").inner_text()
        assert page.locator('[data-walk="fwd1"]').is_disabled() and page.locator('[data-action="w3_rearm"]').is_disabled()
        assert page.locator('[data-action="w3_off"]').is_enabled()
        page.screenshot(path=str(output / "walk_interrupted.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print(f"Walk panel browser smoke PASS ({args.key}): 패드 접힘/펼침, 경고, 비활성 {len(off)}·활성 {len(on)} 버튼, 확인 창, "
          f"수락→서기 복귀, 새로고침, 정지 래치·재무장, 걷기 끄기, 정지 요청, 폰 폭; screenshots: {output}")


if __name__ == "__main__":
    main()
