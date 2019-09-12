from pathlib import Path


class Dataset:
    """"Base dataset class"""

    @classmethod
    def data_path(cls):
        return Path(__file__).resolve().parents[2]

    @classmethod
    def raw_data_path(cls):
        return cls.data_path() / "data" / "raw"

    @classmethod
    def cache_data_path(cls):
        path = cls.data_path() / "data" / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def processed_data_path(cls):
        path = cls.data_path() / "data" / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

