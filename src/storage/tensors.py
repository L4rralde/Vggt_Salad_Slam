from typing import Tuple

import torch

from .repository import Repository

class TensorRepository(Repository):
    def __init__(self, root: str, clear: bool=False):
        super().__init__(root, "TENSORS", clear)

    def _init_db(self, clear: bool) -> None:
        if not clear:
            return
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS TENSORS")
        cursor.execute("""
            CREATE TABLE TENSORS (
                frame_id INTEGER NOT NULL,
                path TEXT NOT NULL
            );            
        """)
        conn.close()

    def append(self, frame_id: int, tensor: torch.Tensor) -> int:
        with self._lock:
            tensors_subdir = self.data_dir / f"{frame_id // 1000:06d}"
            tensors_subdir.mkdir(exist_ok=True)
            tensor_path = tensors_subdir / f"{frame_id:09d}.pt"
            torch.save(tensor.cpu(), tensor_path)

            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT into TENSORS (frame_id, path) VALUES (?, ?)",
                (frame_id, str(tensor_path))
            )

            conn.commit()
            conn.close()

            return frame_id

    def get(self, frame_id: int) -> Tuple[torch.Tensor]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT path FROM TENSORS WHERE frame_id=?",
            (frame_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise KeyError(f"Id {frame_id} not found")
        
        tensor = torch.load(row[0])
        return tensor
