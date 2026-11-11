#!/usr/bin/env python3
"""Read an in-flight request count from either TensorFold health layout."""
import json
import sys


def count(health):
    if not isinstance(health, dict):
        return 0
    live = health.get("live")
    if isinstance(live, dict) and "connections" in live:
        value = live["connections"]
    else:
        value = health.get("running", health.get("requests_running", 0))
    return value if type(value) is int and 0 <= value <= 2147483647 else 0


if __name__ == "__main__":
    try:
        print(count(json.load(sys.stdin)))
    except (ValueError, TypeError):
        print(0)
