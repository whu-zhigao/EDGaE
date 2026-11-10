"""Length-prefixed RPC over TCP with auditable send-side pacing.

Wire format: 8-byte big-endian payload length + JSON header line + raw body.
Pacing: server-side sends are chunked and slept to a target Mbps; every
message updates byte counters so achieved throughput is logged per worker.
"""
from __future__ import annotations
import json
import socket
import struct
import time


CHUNK_BYTES = 256 * 1024


class PacedSender:
    def __init__(self, sock, target_bytes_per_s=None):
        self.sock = sock
        self.target = target_bytes_per_s
        self.bytes_sent = 0

    def sendall_paced(self, data: bytes):
        self.bytes_sent += len(data)
        if not self.target:
            self.sock.sendall(data)
            return
        view = memoryview(data)
        chunk = CHUNK_BYTES
        start = time.perf_counter()
        sent = 0
        while sent < len(data):
            end = min(sent + chunk, len(data))
            self.sock.sendall(view[sent:end])
            sent = end
            if self.target:
                delay = (sent / self.target
                         - (time.perf_counter() - start))
                if delay > 0:
                    time.sleep(delay)
        # Do not sleep for the tail beyond data.


def send_msg(sock, header: dict, body: bytes = b"", sender: PacedSender = None):
    hdr = json.dumps(header).encode()
    frame = struct.pack(">QI", 8 + 4 + len(hdr) + len(body), len(hdr)) + hdr
    if sender is not None:
        sender.sendall_paced(frame)
        if body:
            sender.sendall_paced(body)
    else:
        sock.sendall(frame)
        if body:
            sock.sendall(body)


def recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        piece = sock.recv(n - len(buf))
        if not piece:
            raise ConnectionError("peer closed")
        buf.extend(piece)
    return bytes(buf)


def recv_msg(sock):
    (total,) = struct.unpack(">Q", recv_exact(sock, 8))
    (hlen,) = struct.unpack(">I", recv_exact(sock, 4))
    hdr = json.loads(recv_exact(sock, hlen))
    body = recv_exact(sock, total - 8 - 4 - hlen) if total > 12 + hlen else b""
    return hdr, body
