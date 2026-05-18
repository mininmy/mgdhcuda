"""
round_transfer.py — rsync round_*.json files from the training server to local.

Usage
-----
  python round_transfer.py --host user@myserver \
      --remote-path /home/user/mgdhcuda/poly_saves \
      --local-path ./poly_saves

  # Specific rounds only:
  python round_transfer.py --host user@myserver \
      --remote-path /home/user/mgdhcuda/poly_saves \
      --rounds 2 3

  # Preview without transferring:
  python round_transfer.py --host user@myserver \
      --remote-path /home/user/mgdhcuda/poly_saves \
      --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def sync_rounds(
    remote_host: str,
    remote_path: str,
    local_path: str,
    rounds: Optional[list[int]] = None,
    ssh_key: Optional[str] = None,
    dry_run: bool = False,
) -> list[Path]:
    """
    rsync round_*.json files from a remote SSH host to a local directory.

    Parameters
    ----------
    remote_host : e.g. "user@192.168.1.10"
    remote_path : absolute path on the server, e.g. "/home/user/mgdhcuda/poly_saves"
    local_path  : local destination directory (created if absent)
    rounds      : round indices to transfer; None = all found on the server
    ssh_key     : path to SSH private key (optional)
    dry_run     : pass --dry-run to rsync (shows what would be transferred)

    Returns
    -------
    list of local Path objects that were transferred (empty on dry_run)
    """
    local = Path(local_path)
    local.mkdir(parents=True, exist_ok=True)

    ssh_extra = f"ssh -i {ssh_key}" if ssh_key else "ssh"
    base = ["rsync", "-avz", "--progress", "-e", ssh_extra]
    if dry_run:
        base.append("--dry-run")

    transferred: list[Path] = []

    if rounds is None:
        cmd = base + [
            "--include=round_*.json",
            "--exclude=*",
            f"{remote_host}:{remote_path}/",
            str(local) + "/",
        ]
        _run(cmd)
        if not dry_run:
            transferred = sorted(local.glob("round_*.json"))
    else:
        for r in rounds:
            fname = f"round_{r}.json"
            cmd = base + [f"{remote_host}:{remote_path}/{fname}", str(local / fname)]
            _run(cmd)
            if not dry_run:
                p = local / fname
                if p.exists():
                    transferred.append(p)

    return transferred


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command exited with code {res.returncode}: {' '.join(cmd)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="rsync boosted-GMDH round JSON files from the training server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host", metavar="USER@HOST", required=True,
                   help="Remote SSH host, e.g. user@192.168.1.10")
    p.add_argument("--remote-path", metavar="PATH", required=True,
                   help="Absolute path to poly_saves on the server")
    p.add_argument("--local-path", metavar="DIR", default="./poly_saves",
                   help="Local destination directory (default: ./poly_saves)")
    p.add_argument("--rounds", metavar="N", nargs="+", type=int,
                   help="Round indices to transfer (default: all found)")
    p.add_argument("--ssh-key", metavar="FILE",
                   help="Path to SSH private key")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be transferred without doing it")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"Syncing from {args.host}:{args.remote_path} → {args.local_path}")
    files = sync_rounds(
        remote_host=args.host,
        remote_path=args.remote_path,
        local_path=args.local_path,
        rounds=args.rounds,
        ssh_key=args.ssh_key,
        dry_run=args.dry_run,
    )
    if files:
        print(f"\nTransferred {len(files)} file(s):")
        for f in files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
