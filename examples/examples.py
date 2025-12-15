"""Deprecated wrapper.

This module previously contained the getting-started example. Use
`examples.getting_started` instead; this file now forwards to that module.
"""

import subprocess
import sys

print("'examples' module is deprecated; running 'examples.getting_started' as a module.")
subprocess.run([sys.executable, "-m", "examples.getting_started"], check=True)
