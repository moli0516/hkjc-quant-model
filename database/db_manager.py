import sqlite3
import pathlib


class dbManager:
    def __init__(self, db_path = pathlib.Path(__file__).parent / "hkjc_racing.db"):
        self.db_path = db_path
    def get_connection(self):
        try:
            return sqlite3.connect(self.db_path)
        except:
            print("Cannot connect to database!")
    def init_db(self):
        create_cmds = ["""
        CREATE TABLE IF NOT EXISTS race(
            raceid TEXT PRIMARY KEY, 
            racecourse TEXT, 
            race_date TEXT, 
            race_no INTEGER, 
            class TEXT, 
            track_condition TEXT, 
            length INTEGER, 
            track_type TEXT, 
            track_texture TEXT
        );
        ""","""CREATE TABLE race_runners(
            race_id TEXT,
            horse_id TEXT,
            horse_name TEXT
            draw INTEGER,
            weight INTEGER,
            horse_weight INTEGER
            jockey TEXT,
            trainer TEXT,
            odds REAL,
            finish_time TEXT,
            placing INTEGER,
            incident_report TEXT,
            PRIMARY KEY (race_id, horse_id),
            FOREIGN KEY (race_id) REFERENCES race(race_id));"""] 
        with self.get_connection() as conn:
            cur = conn.cursor()
            for cmd in create_cmds:
                cur.execute(cmd)
            conn.commit()
        def insert_db(self, *args):
            with self.get_connection() as conn:
                cur = conn.cursor()
                for cmd in args:
                    cur.execute(cmd)
                conn.commit()

if __name__ == "__main__":
    db = dbManager()
    db.init_db()



