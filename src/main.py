from cli import Paths, run


def main() -> None:
    run(
        Paths(
            means="means.csv",
            stds="std.csv",
            weights="model.npz",
            checkpoint="model.pt",
            web_model="model.bin",
        )
    )


if __name__ == "__main__":
    main()