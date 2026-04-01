import csv
import tempfile
import unittest
from pathlib import Path

from src.strategy.uptrend_hold_state_flow.target_resolver import resolve_target


class TargetResolverTests(unittest.TestCase):
    def test_resolve_target_uses_unique_pinyin_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            basic_path = tmp_path / "stock_basic.csv"
            data_dir = tmp_path / "daily"
            data_dir.mkdir()

            self._write_basic_csv(
                basic_path,
                [
                    {"ts_code": "688163.SH", "symbol": "688163", "name": "赛伦生物", "area": "上海", "industry": "生物制药", "market": "科创板"},
                    {"ts_code": "300583.SZ", "symbol": "300583", "name": "赛托生物", "area": "山东", "industry": "化学制药", "market": "创业板"},
                    {"ts_code": "688065.SH", "symbol": "688065", "name": "凯赛生物", "area": "上海", "industry": "化工原料", "market": "科创板"},
                ],
            )
            for symbol in ["sh688163", "sz300583", "sh688065"]:
                (data_dir / f"{symbol}.csv").write_text("trade_date,close\n20260331,1\n", encoding="utf-8")

            target = resolve_target("赛轮生物", str(data_dir), str(basic_path))

            self.assertEqual(target["ts_code"], "688163.SH")
            self.assertEqual(target["name"], "赛伦生物")

    def test_resolve_target_prefers_candidate_with_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            basic_path = tmp_path / "stock_basic.csv"
            data_dir = tmp_path / "daily"
            data_dir.mkdir()

            self._write_basic_csv(
                basic_path,
                [
                    {"ts_code": "000001.SZ", "symbol": "000001", "name": "示例科技", "area": "深圳", "industry": "软件服务", "market": "主板"},
                    {"ts_code": "600001.SH", "symbol": "600001", "name": "示例科技", "area": "上海", "industry": "软件服务", "market": "主板"},
                ],
            )
            (data_dir / "sh600001.csv").write_text("trade_date,close\n20260331,1\n", encoding="utf-8")

            target = resolve_target("示例科技", str(data_dir), str(basic_path))

            self.assertEqual(target["ts_code"], "600001.SH")
            self.assertEqual(target["symbol"], "sh600001")

    @staticmethod
    def _write_basic_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ts_code", "symbol", "name", "area", "industry", "market"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()