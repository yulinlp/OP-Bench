#!/usr/bin/env python3
from start_agent import main

if __name__ == "__main__":
    import sys

    sys.argv[1:1] = ["--agent", "ldagent"]
    main()

