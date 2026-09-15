#!/usr/bin/env python3
"""Train Lulu/ReN-OPD; see ../README.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulu.training import main
if __name__ == '__main__':
    main()
