from pathlib import Path

from theremin import Theremin


def main():
    config_path = Path(__file__).with_name("config.yaml")
    Theremin(config_path).start()


if __name__ == "__main__":
    main()
