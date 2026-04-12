"""Shared warmup observation factory for model compilation.

Kept in its own module to avoid pulling heavy dependencies (quic-portal,
Modal) into callers that only need the warmup observation.
"""

import numpy as np


def make_aloha_warmup_observation() -> dict:
    """Create a dummy ALOHA observation for triggering torch.compile warmup.

    Matches the input shape expected by pi0/pi05 ALOHA model configs:
    14-dim state vector + four 224x224 camera images.
    """
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "warmup",
    }
