"""A partial/buffered IPC transfer must remain covered by worker timeout."""
import threading
import pytest
from lulu.persistent import MemoryChannel


@pytest.mark.parametrize('operation', ['send', 'recv'])
def test_stalled_tensor_transfer_is_bounded(operation):
    unblock = threading.Event()
    exited = threading.Event()
    class StalledConnection:
        def send_bytes(self, message):
            unblock.wait()
            exited.set()
        def recv_bytes(self):
            unblock.wait()
            exited.set()
            return b'\x80\x04N.'  # pickled None
    channel = MemoryChannel(StalledConnection(), timeout=.02)
    try:
        with pytest.raises(TimeoutError, match='IPC transfer'):
            channel.send({'payload': 'value'}) if operation == 'send' else channel.recv()
    finally:
        unblock.set()
        assert exited.wait(1)
