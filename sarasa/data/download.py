if __name__ == "__main__":
    import argparse

    from sarasa.data import Datasets

    p = argparse.ArgumentParser("sarasa data download")
    p.add_argument(
        "dataset", type=Datasets, help=f"Dataset to download. Supported datasets: {', '.join(Datasets.__members__)}"
    )
    args = p.parse_args()

    args.dataset.load(download=True)
