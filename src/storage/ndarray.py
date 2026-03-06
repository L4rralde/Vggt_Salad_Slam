from typing import Dict

import numpy as np

from .repository import Repository


class NdarrayRepository(Repository):
    def __init__(self, root: str, clear: bool=False):
        super().__init__(root, "NDARRAYS", clear)

    def _init_db(self, clear: bool) -> None:
        if not clear:
            return
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS NDARRAYS")
        cursor.execute("""
            CREATE TABLE NDARRAYS (
                id INTEGER NOT NULL,
                path TEXT NOT NULL
            );
        """)
        conn.close()

    def append(self, id: int, array_data: Dict[str, np.ndarray]|np.ndarray) -> int:
        with self._lock:
            arrays_subdir = self.data_dir / f"{id // 1000:06d}"
            arrays_subdir.mkdir(exist_ok=True)
            array_path = arrays_subdir / f"{id:09d}.npz"
            if isinstance(array_data, np.ndarray):
                array_data = {'__ndarray_data': array_data}
            np.savez(array_path, **array_data)

            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT into NDARRAYS (id, path) VALUES (?, ?)",
                (id, str(array_path))
            )
            conn.commit()
            conn.close()

            return id

    def get(self, id: int, copy: bool=False) -> Dict[str, np.ndarray]|np.ndarray:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT path FROM NDARRAYS WHERE id=?",
            (id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise KeyError(f"Id {id} not found")
    
        data = np.load(row[0])
        if copy:
            data = dict(data)
        if len(data.keys()) == 1 and '__ndarray_data' in data.keys():
            data = data['__ndarray_data']
        
        return data
