from torchvision.datasets import CelebA


if __name__ == "__main__":
    dataset = CelebA(root="/home/erogullari/datasets", split="all", download=False)
    print(f"Loaded CelebA dataset with {len(dataset)} samples.")