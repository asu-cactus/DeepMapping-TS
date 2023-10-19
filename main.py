import importlib


def main():
    run_func = getattr(importlib.import_module("exp.train_tcn"), "run")
    run_func()


if __name__ == "__main__":
    main()