"""Keep pytest focused on offline tests.

The transport smoke scripts expose ``run(...)`` entry points and need live
servers or Modal deployments. Ignoring them avoids optional deployment imports
during normal pytest collection.
"""

collect_ignore = [
    "test_quic.py",
    "test_ws.py",
    "test_tunnel.py",
    "test_modal_quic.py",
]
