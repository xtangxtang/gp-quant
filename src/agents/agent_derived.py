"""Agent 5: Derived data — generate weekly 5d bars and other derived datasets."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "downloader"))

from base_agent import BaseAgent


class DerivedDataAgent(BaseAgent):
    name = "derived"
    description = "生成衍生数据（5 日周线等）"

    def run(self, **kwargs):
        import glob
        from multiprocessing import Pool, cpu_count

        # Import the weekly generator
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        sys.path.insert(0, scripts_dir)
        from gen_weekly_5d import process_file, WINDOW, OUT_DIR, DATA_DIR

        self.update_progress("weekly_5d", 0, 0)

        os.makedirs(OUT_DIR, exist_ok=True)
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
        total = len(files)

        workers = min(16, cpu_count())
        done = 0
        errors = 0
        with Pool(workers) as pool:
            for filepath, nrows, err in pool.imap_unordered(process_file, files, chunksize=32):
                done += 1
                if err:
                    errors += 1
                if done % 500 == 0 or done == total:
                    self.update_progress("weekly_5d", done, total)

        self.state.set_success(0, {"files_written": done - errors})


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    agent = DerivedDataAgent(args.data_dir)
    ok = agent.execute()
    sys.exit(0 if ok else 1)
