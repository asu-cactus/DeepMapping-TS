import importlib
import random


random.seed(42)


def main():
    run_func = getattr(importlib.import_module("exp.inference"), "run")
    run_func()


if __name__ == "__main__":
    main()
