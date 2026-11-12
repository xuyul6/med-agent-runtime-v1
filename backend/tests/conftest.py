"""Optional network guard for the deterministic CI suite."""
import os
import socket

import pytest


@pytest.fixture(scope="session", autouse=True)
def offline_test_guard():
    if os.environ.get("MAR_OFFLINE_TESTS") != "1":
        yield
        return

    def blocked(*_args, **_kwargs):
        raise RuntimeError("offline_tests_network_disabled")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", blocked)
        patch.setattr(socket.socket, "connect_ex", blocked)
        patch.setattr(socket.socket, "sendto", blocked)
        yield
