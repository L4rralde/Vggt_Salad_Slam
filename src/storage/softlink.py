from pathlib import Path
import shutil


class SoftLink:
    def __init__(self, root: str, other_root: str, clear: bool = False):
        self.root = Path(root).resolve()
        self.other_root = Path(other_root).resolve()

        if clear and self.root.exists():
            shutil.rmtree(self.root)

        self.root.mkdir(parents=True, exist_ok=True)

    def copy(self, path: str) -> None:
        path = Path(path).resolve()

        try:
            rel_path = path.relative_to(self.other_root)
        except ValueError:
            raise ValueError(f"{path} does not correspond to {self.other_root}")

        new_path = self.root / rel_path
        new_path.parent.mkdir(parents=True, exist_ok=True)

        if new_path.exists() or new_path.is_symlink():
            new_path.unlink()

        new_path.symlink_to(path, target_is_directory=path.is_dir())
