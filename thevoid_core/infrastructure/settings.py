from pathlib import Path


class Settings:

    ORGANIZATION = "the Void"
    APPLICATION = "the Void"

    @staticmethod
    def data_directory() -> Path:
        return Path.home() / ".the_void"