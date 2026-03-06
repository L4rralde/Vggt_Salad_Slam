from abc import ABC, abstractmethod
from typing import Any, Tuple, List
from pathlib import Path
import sqlite3
import shutil
import threading


class Repository(ABC):
    def __init__(
        self,
        root: str,
        table_name: str="DATA",
        clear: bool=False
    ):
        self.root = Path(root)
        self.db_path = self.root / "metadata.db"
        self.data_dir = self.root / table_name.lower()
        self.table_name = table_name

        self._init_dirs(clear)
        self._init_db(clear)

        self._lock = threading.Lock()

    @abstractmethod
    def _init_db(self, clear: bool) -> None:
        raise NotImplementedError

    def _init_dirs(self, clear: bool) -> None:
        if clear and self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    @abstractmethod
    def append(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get(self, frame_id: int) -> Any:
        raise NotImplementedError()

    def list_all_ids(self) -> List[int]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"SELECT frame_id from {self.table_name}")
        rows = cursor.fetchall()
        conn.close()

        return [r[0] for r in rows]

    def __contains__(self, frame_id: int) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT EXISTS (SELECT 1 FROM {self.table_name} where frame_id=?) ",
            (frame_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None or row[0] == 0:
            return False
        return True
