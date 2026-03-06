from typing import Tuple

from PIL import Image

from .repository import Repository


class FrameRepository(Repository):
    def __init__(self, root: str, clear: bool=False):
        super().__init__(root, "FRAMES", clear)

    def _init_db(self, clear: bool) -> None:
        if not bool:
            return
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS FRAMES")
        cursor.execute("""
            CREATE TABLE FRAMES (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                path TEXT NOT NULL
            );
        """)

        conn.close()

    def append(self, frame: Image.Image, stamp: float) -> int:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            #insert timestamp only to get frame id
            cursor.execute(
                "INSERT into FRAMES (timestamp, path) VALUES (?, '')",
                (stamp, )
            )
            conn.commit()
            # '?' Binds data to the sql query.
            # This is the recommended way, not string formatting. 
            # The latter is vulnerable to sql injection attacks
            frame_id = cursor.lastrowid

            subdir = self.data_dir / f"{frame_id // 1000:06d}"
            subdir.mkdir(exist_ok=True)
            fname = f"{frame_id:09d}.jpg"
            fpath = subdir / fname

            frame.save(fpath)

            #Insert path to db
            cursor.execute(
                "UPDATE FRAMES SET path=? WHERE frame_id=?",
                (str(fpath), frame_id)
            )
            conn.commit()

            conn.close()

            return frame_id

    def get(self, frame_id: int) -> Tuple[Image.Image, float]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT path, timestamp FROM FRAMES WHERE frame_id=?",
            (frame_id, )
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise KeyError(f"Frame {frame_id} not found")
        
        path, timestamp = row
        with Image.open(path) as img:
            frame = img.copy()
        
        return frame, timestamp

    def get_path(self, frame_id: int) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT path FROM FRAMES WHERE frame_id=?",
            (frame_id, )
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise KeyError(f"Frame {frame_id} not found")
        
        path = row[0]
        return path
