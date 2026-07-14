import json
import pandas as pd
import pathlib
import fastparquet

class Feature_pipeline:
    def __init__(self, json_path):
        self.json_path = json_path
        self.df = self._load_original_json_df(self.json_path).sort_values(by=[ "date", "races.race_id"]).reset_index(drop=True)
        self.bl_win = 1/14
        self.bl_place = 3/14
    def _load_original_json_df(self, json_path):
        combine_data = []
        for file in json_path.iterdir():
            print(file.name)
            if file.is_file():
                file_path = json_path / file.name
                with open(file_path, "r", encoding='utf-8') as f:
                    raw_json = json.load(f)
                    if isinstance(raw_json, list):
                        combine_data.extend(raw_json)
                    else:
                        combine_data.append(raw_json)
        df = pd.json_normalize(combine_data)
        return df
    def _build_win_placing(self):
        self.df['is_win'] = (self.df['placing'] == 1).astype(int)
        self.df['is_place'] = (self.df['placing'] < 4).astype(int)
    #def _build_draw_features(self):
    #    self.df['draw_win_rate'] = self.df.groupby('draw')['is_win'].mean()
    #    self.df['draw_place_rate'] = self.df.groupby('draw')['is_place'].mean()
    def _build_h_speed_z_features(self):
        #1. Absolute speed
        self.df["h_speed"] = self.df["length"] / self.df["finished_time_sec"]
        #2. Z score of speed in every race
        self.df["h_speed_z"] = self.df.groupby("races.race_id")["h_speed"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
        
        #!!!IMPORTANT!!! GROUP BY HORSE ID AND ACCESS Z SCORE OF SPEED
        h_grpby_z = self.df.groupby("horse_id")["h_speed_z"]
        #3. Previous Z score of speed in the last races
        self.df["h_prev_speed_z"] = h_grpby_z.shift(1)
        #4. Average Z score over the history
        self.df["h_mean_speed_z"] = h_grpby_z.transform(lambda x: x.shift(1).expanding(min_periods=1).mean()).fillna(0)
        #5. Average Z score of speed in the last 2 races (Do the horse has outstanding performances recently)
        self.df["h_rolling_2_mean_speed_z"] = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=2, min_periods=2).mean()).fillna(self.df["h_speed_z"])
        #6. Standard deviation of Z score of speed in the last 2 races (Do the horse has stable performances recently)
        self.df["h_rolling_2_speed_z_std"] = h_grpby_z.transform(lambda x: x.shift(1).rolling(window=2, min_periods=2).std()).fillna(0)
        
    def _build_j_smoothing_rolling_n_features(self, n):
        alpha = 15
        j_grp_win = self.df.groupby('jockey')["is_win"]
        j_recent_30_cnt = j_grp_win.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).count()).fillna(0) #can be used again in the place part
        j_recent_30_wins = j_grp_win.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()).fillna(0)
        self.df[f"j_smoothed_rolling_{n}_win_rate"] = ((j_recent_30_wins + alpha * self.bl_win) / (j_recent_30_cnt + alpha)).fillna(self.bl_win)
        j_grp_place = self.df.groupby('jockey')["is_place"]
        j_recent_30_place = j_grp_place.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()).fillna(0)
        self.df[f"j_smoothed_rolling_{n}_place_rate"] = ((j_recent_30_place + alpha * self.bl_place) / (j_recent_30_cnt + alpha)).fillna(self.bl_place)
    
    def _build_j_smoothing_features(self):
        alpha = 15
        #!!!IMPORTANT!!! GROUP BY JOCKEY AND ACCESS WIN STATUS
        j_grp_win = self.df.groupby('jockey')["is_win"]
        j_cnt = j_grp_win.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0) #can be used again in the place part
        j_wins = j_grp_win.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        self.df["j_smoothed_win_rate"] = ((j_wins + alpha * self.bl_win) / (j_cnt + alpha)).fillna(self.bl_win)
        #!!!IMPORTANT!!! GROUP BY JOCKEY AND ACCESS PLACE STATUS
        j_grp_place = self.df.groupby('jockey')["is_place"]
        j_places = j_grp_place.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        self.df["j_smoothed_place_rate"] = ((j_places + alpha * self.bl_place) / (j_cnt + alpha)).fillna(self.bl_place)
    
    def _build_t_smoothing_rolling_n_features(self, n):
        alpha = 15
        t_grp_win = self.df.groupby('trainer')["is_win"]
        t_recent_30_cnt = t_grp_win.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).count()).fillna(0) #can be used again in the place part
        t_recent_30_wins = t_grp_win.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()).fillna(0)
        self.df[f"t_smoothed_rolling_{n}_win_rate"] = ((t_recent_30_wins + alpha * self.bl_win) / (t_recent_30_cnt + alpha)).fillna(self.bl_win)
        t_grp_place = self.df.groupby('trainer')["is_place"]
        t_recent_30_place = t_grp_place.transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()).fillna(0)
        self.df[f"t_smoothed_rolling_{n}_place_rate"] = ((t_recent_30_place + alpha * self.bl_place) / (t_recent_30_cnt + alpha)).fillna(self.bl_place)
    
    def _build_t_smoothing_features(self):
        alpha = 15
        #!!!IMPORTANT!!! GROUP BY JOCKEY AND ACCESS WIN STATUS
        t_grp_win = self.df.groupby('trainer')["is_win"]
        t_cnt = t_grp_win.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0) #can be used again in the place part
        t_wins = t_grp_win.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        self.df["t_smoothed_win_rate"] = ((t_wins + alpha * self.bl_win) / (t_cnt + alpha)).fillna(self.bl_win)
        
        #!!!IMPORTANT!!! GROUP BY JOCKEY AND ACCESS PLACE STATUS
        t_grp_place = self.df.groupby('trainer')["is_place"]
        t_places = t_grp_place.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        self.df["t_smoothed_place_rate"] = ((t_places + alpha * self.bl_place) / (t_cnt + alpha)).fillna(self.bl_place)
        
if __name__ == "__main__":
    json_path = pathlib.Path.cwd().parent / "data" / "cleaned_json" / "flatten"