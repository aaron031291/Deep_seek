#!/usr/bin/env python3
__all__ = ["Mapping", "Sequence"]

try:
    from collections.abc import Mapping, Sequence
except ImportError:
    from collections import Mapping, Sequence
