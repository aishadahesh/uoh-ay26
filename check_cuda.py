"""Check whether CUDA is available through PyTorch."""

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for index in range(device_count):
            print(f"Device {index}: {torch.cuda.get_device_name(index)}")
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    else:
        print("No CUDA-capable GPU is available to PyTorch.")


if __name__ == "__main__":
    main()
