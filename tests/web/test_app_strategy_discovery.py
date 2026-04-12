from pathlib import Path
import unittest
from unittest.mock import patch

from web import app as web_app


class WebStrategyDiscoveryTests(unittest.TestCase):
    def test_continuous_decline_recovery_uses_scan_entrypoint(self) -> None:
        strategy = web_app._resolve_strategy("continuous_decline_recovery")
        self.assertEqual(strategy["entrypoint_path"].name, "run_continuous_decline_recovery_scan.py")

    def test_scan_entrypoint_has_higher_priority_than_diagnostics(self) -> None:
        self.assertLess(
            web_app._entrypoint_sort_key(Path("run_continuous_decline_recovery_scan.py")),
            web_app._entrypoint_sort_key(Path("run_continuous_decline_recovery_diagnostics.py")),
        )

    def test_apply_route_uses_scan_entrypoint(self) -> None:
        captured: dict[str, object] = {}
        app = web_app.create_app()

        def fake_run_strategy(strategy: dict[str, object], values: dict[str, object]) -> dict[str, object]:
            captured["entrypoint_name"] = strategy["entrypoint_path"].name
            captured["values"] = values
            return {"ok": True, "entrypoint_name": strategy["entrypoint_path"].name}

        with patch("web.app._run_strategy", side_effect=fake_run_strategy):
            response = app.test_client().post(
                "/api/strategies/continuous_decline_recovery/apply",
                json={"values": {"start_date": "2026-03-01", "end_date": "2026-03-30"}},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["entrypoint_name"], "run_continuous_decline_recovery_scan.py")
        self.assertEqual(captured["entrypoint_name"], "run_continuous_decline_recovery_scan.py")

    def test_strategies_route_supports_package_relative_imports(self) -> None:
        app = web_app.create_app()

        response = app.test_client().get("/api/strategies")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        strategy_ids = {item["id"] for item in payload["strategies"]}
        self.assertIn("four_layer_entropy_system", strategy_ids)


if __name__ == "__main__":
    unittest.main()