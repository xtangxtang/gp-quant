"""Agent 4: 1-minute data sync — recent trading days."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downloader"))

from base_agent import BaseAgent


class MinuteAgent(BaseAgent):
    name = "minute"
    description = "增量同步 1 分钟 K 线数据"

    def run(self, **kwargs):
        from sync_a_share_1m import main as sync_main

        recent_days = kwargs.get("recent_open_days", 3)
        threads = kwargs.get("threads", 4)
        source = kwargs.get("source", "ts")
        fqt = kwargs.get("fqt", 0)
        retry_rounds = kwargs.get("retry_rounds", 2)

        self.update_progress("sync_1m_data")

        argv_backup = sys.argv
        try:
            sys.argv = [
                "sync_a_share_1m.py",
                "--output_dir", self.data_dir,
                "--token", self.token,
                "--threads", str(threads),
                "--source", source,
                "--fqt", str(fqt),
                "--recent_open_days", str(recent_days),
                "--retry_failed_rounds", str(retry_rounds),
            ]
            try:
                sync_main()
            except SystemExit:
                pass
            self.update_progress("done")
        finally:
            sys.argv = argv_backup


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--token", default="")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--source", default="ts")
    p.add_argument("--fqt", type=int, default=0)
    p.add_argument("--recent-open-days", type=int, default=3)
    p.add_argument("--retry-rounds", type=int, default=2)
    args = p.parse_args()
    agent = MinuteAgent(args.data_dir, args.token)
    ok = agent.execute(
        recent_open_days=args.recent_open_days,
        threads=args.threads,
        source=args.source,
        fqt=args.fqt,
        retry_rounds=args.retry_rounds,
    )
    sys.exit(0 if ok else 1)
