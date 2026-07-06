from pathlib import Path


class FileStorage:
    def __init__(self, base_dir: str = "uploads") -> None:
        self.base_dir = Path(base_dir)

    def ensure_module_dir(self, module_name: str) -> Path:
        target = self.base_dir / module_name
        target.mkdir(parents=True, exist_ok=True)
        return target
