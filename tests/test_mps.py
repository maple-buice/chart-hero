import torch


def main():
    """
    This script checks for MPS availability and runs a simple
    tensor operation on the MPS device if available.
    """
    print("--- MPS Device Test ---")

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "❌ MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "❌ MPS not available because the current macOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
        return

    print("✅ MPS is available!")
    print("-" * 20)

    try:
        mps_device = torch.device("mps")

        # Create a tensor directly on the MPS device
        x = torch.ones(5, device=mps_device)
        print(f"Tensor created on MPS device:\n{x}")
        print("-" * 20)

        # Perform a simple operation
        y = x * 2
        print(f"Result of tensor operation on MPS device:\n{y}")
        print("-" * 20)

        # Move a tensor from CPU to MPS
        z_cpu = torch.rand(2, 3)
        print(f"Tensor created on CPU:\n{z_cpu}")
        z_mps = z_cpu.to(mps_device)
        print(f"Tensor moved to MPS device:\n{z_mps}")
        print("-" * 20)

        print("✅ Successfully performed operations on the MPS device.")

    except Exception as e:
        print(f"❌ An error occurred while using the MPS device: {e}")


if __name__ == "__main__":
    main()
