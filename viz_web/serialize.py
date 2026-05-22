from __future__ import annotations

import base64
import numpy as np


def encode_layers_u8(layers: np.ndarray, *, expected_n: int | None = 7) -> dict:
    """
    Encode world layers for frontend rendering.

    Args:
        layers: float32/float64 array (H, W, N) ideally in [0,1]
        expected_n: if not None, enforce N==expected_n (default 7)

    Returns:
        {
          "b64": base64 encoded bytes of uint8 array laid out in C-order (H,W,N),
          "H": H, "W": W, "N": N,
          "byte_len": number of raw bytes
        }
    """
    if layers is None:
        raise ValueError("encode_layers_u8: layers is None")

    if not isinstance(layers, np.ndarray):
        raise TypeError(f"encode_layers_u8: layers must be np.ndarray, got {type(layers)}")

    if layers.ndim != 3:
        raise ValueError(f"encode_layers_u8: expected layers.ndim==3 (H,W,N), got shape={layers.shape}")

    H, W, N = layers.shape

    if expected_n is not None and N != expected_n:
        raise ValueError(
            f"encode_layers_u8: expected N={expected_n} layers but got N={N}. "
            f"Frontend renderer likely assumes {expected_n} layers."
        )

    # Clamp to [0,1] then convert to uint8
    x = np.clip(layers, 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8, copy=False)

    # Ensure predictable byte layout
    if not u8.flags["C_CONTIGUOUS"]:
        u8 = np.ascontiguousarray(u8)

    raw = u8.tobytes(order="C")
    b64 = base64.b64encode(raw).decode("ascii")

    out = {"b64": b64, "H": int(H), "W": int(W), "N": int(N), "byte_len": int(len(raw))}

    # Debug (kept lightweight)
    print(f"[Serialize] layers {H}x{W}x{N} -> {len(raw)} bytes, b64 chars={len(b64)}")

    return out
