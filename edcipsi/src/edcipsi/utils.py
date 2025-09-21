from __future__ import annotations
from datetime import datetime
import io, os

class TeeWithTimestamp(io.TextIOBase):
    """stdout/stderr"""
    def __init__(self, *targets):
        self.targets = targets
        self._buf = ""
    def _stamp(self) -> str:
        return datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    def write(self, s: str):
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stamped = self._stamp() + line + "\n"
            for t in self.targets:
                try:
                    t.write(stamped); t.flush()
                except Exception:
                    pass
        return len(s)
    def flush(self):
        if self._buf:
            stamped = self._stamp() + self._buf
            for t in self.targets:
                try:
                    t.write(stamped); t.flush()
                except Exception:
                    pass
            self._buf = ""

def parse_bool(s: str) -> bool:
    t = str(s).strip().lower()
    if t in ("1","true","t","yes","y","on"):  return True
    if t in ("0","false","f","no","n","off"): return False
    raise ValueError(f"bool parse error: {s}")

def _log_read(path: str):
    try:
        ap = os.path.abspath(path)
    except Exception:
        ap = str(path)
    print(f"[READ] {ap}")

