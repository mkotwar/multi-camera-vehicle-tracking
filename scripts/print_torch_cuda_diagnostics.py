from __future__ import annotations

import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
