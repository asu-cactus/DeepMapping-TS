import importlib


def main():
    run_func = getattr(importlib.import_module("exp.sz3_compression"), "run")
    run_func()


if __name__ == "__main__":
    main()
