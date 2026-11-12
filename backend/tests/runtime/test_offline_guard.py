"""Check the opt-in CI network guard itself without connecting to a service."""
import os
import socket

import pytest


@pytest.mark.skipif(os.environ.get("MAR_OFFLINE_TESTS") != "1", reason="CI offline guard is opt-in")
def test_offline_guard_blocks_socket_connections():
    with socket.socket() as connection:
        with pytest.raises(RuntimeError, match="offline_tests_network_disabled"):
            connection.connect(("127.0.0.1", 9))
