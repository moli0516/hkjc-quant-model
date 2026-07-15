import pandas as pd
import json
import pathlib
import os
import re

class Data_cleaning_manager_flatten:
    def __init__(self, raw_json_path, cleaned_json_path, rating_json_path):
        self.raw_json_path = raw_json_path
        self.cleaned_json_path = cleaned_json_path
        self.cleaned_json_path = rating_json_path
        self.df = self.create_df()
    def create_df(self):
        raw_json = None
        with open(self.raw_json_path, "r", encoding='utf-8') as f:
            raw_json = json.load(f)

        df = pd.json_normalize(
            raw_json,
            record_path=['races','horses'],
            meta=['date','venue',['races','race_id'],['races','basic_info'],['races','track_condition'],['races','track_info'],['races','cumulative_finish_time'],['races','sectional_finish_time']]
        )
        return df
    def merge_rating(self):
        with open(self.rating_json_path, "r", encoding='utf-8') as f:
            rating_data = json.load(f)
        rating_df = pd.DataFrame(rating_data)
        
        self.df['horse_id'] = self.df['horse_id'].astype(str)
        rating_df['horse_id'] = rating_df['horse_id'].astype(str)
        
        self.df = self.df.merge(rating_df, on='horse_id', how='left')
        
        self.df['rating'] = self.df['rating'].fillna(0).astype(int)
    def get_horse_id(self,horse_name):
        match = re.search(r'[A-Z]\d{1,3}', horse_name)
        if match:
            return match.group(0)
    def clean_horse_name(self,horse_name):
        index = horse_name.find("(")
        return horse_name[:index]
    def clean_head_horse_dist(self,hhd):
        if isinstance(hhd, str):
            dist = 0
            if "-" in hhd and "/" in hhd:
                int_fraction = hhd.split("-")
                dist = int(int_fraction[0])
                fraction = int_fraction[1].split("/")
                dist += int(fraction[0]) / int(fraction[1])
                return dist
            elif "/" in hhd:
                fraction = hhd.split("/")
                dist += int(fraction[0]) / int(fraction[1])
                return dist
            else:
                margin_map = {
                        "---": 0.0, "鼻位": 0.05, "短馬頭位": 0.1, "頭位": 0.2, "頸位": 0.3,
                        "多個馬身": 99.0, "未能完成賽事": None, "退出": None, "多個馬位": 99.0, "平頭馬":0.0
                        }
                if hhd in margin_map.keys():
                    return margin_map[hhd]
                else:
                    return 0.25
        return hhd
    def convert_min_to_sec(self,time):
        if time != "---":
            return int(time[0])*60 + float(time[2:])
    def get_class(self, class_str):
        class_map = {
            "第五班": 5,
            "第四班": 4,
            "第三班": 3,
            "第二班": 2,
            "第一班": 1,
        }
        if class_str in class_map:
            return class_map[class_str]
        else:
            return class_str
    def remove_W(self):
        hkjc_all_special_codes = [
            "DISQ", "DNF", "FE", "ML", "PU", "TNP", "TO", "UR", 
            "VOID", "WR", "WV", "WV-A", "WX", "WX-A", "WXNR"
        ]

        try:
            self.df = self.df[~self.df['placing'].isin(hkjc_all_special_codes)]
            self.df = self.df.copy()
        except:
            pass
    def extract_track_info(self,text):
        clean_text = text.replace('"', '').replace('“', '').replace('”', '').strip()
        
        # 情況 1：全天候跑道（無移欄）
        if "全天候" in clean_text:
            return clean_text, "N/A"
            
        match = re.search(r"([^\s-]+)\s*-\s*([A-Za-z0-9+]+)", clean_text)
        if match:
            surface = match.group(1)
            course_type = match.group(2)
            return surface, course_type
        return clean_text, "N/A"
    def remove_empty_basic_info(self):
        try:
            self.df = self.df[~self.df['races.basic_info'] != ""]
            self.df = self.df.copy()
        except:
            pass
        
    def start_clean(self):
        self.remove_W()
        self.df['placing'] = self.df['placing'].astype(str)
        try:
            self.df["placing"] = pd.to_numeric(self.df["placing"].str.extract(r'(\d+)')[0], errors='coerce')
        except:
            self.df["placing"] = self.df["placing"].astype(int)
            
        self.df["draw"] = self.df["draw"].astype(int)
        self.df['finished_time_sec'] = self.df["finished_time"].apply(self.convert_min_to_sec)
        self.df['head_horse_dist_cleaned'] = self.df["head_horse_dist"].apply(self.clean_head_horse_dist)
        self.df['horse_id'] = self.df["horse_name"].apply(self.get_horse_id)
        self.df['horse_name'] = self.df["horse_name"].apply(self.clean_horse_name)
        self.df['class'] = self.df["races.basic_info"].str.slice(stop=3).apply(self.get_class)
        self.df['length'] = pd.to_numeric(self.df["races.basic_info"].str.slice(start=6,stop=10).str.extract(r'(\d+)')[0], errors='coerce')
        
        # 【修正 1】拆開賦值，防止 DataFrame 變成雙層 MultiIndex 欄位
        track_res = self.df["races.track_info"].apply(self.extract_track_info)
        self.df["track_texture"] = [x[0] for x in track_res]
        self.df["track_type"] = [x[1] for x in track_res]
        
        self.df = self.df[~self.df['class'].apply(lambda x: isinstance(x, str))]
        self.df = self.df.copy()
        
        # 【修正 2】使用 100% 成功的過濾法來移除欄位
        cols_to_drop = ["races.basic_info", "head_horse_dist", "finished_time", "track_info"]
        self.df = self.df[[col for col in self.df.columns if col not in cols_to_drop]]
        
        print("【成功】移除後的最終欄位：", list(self.df.columns))
    
    def start_clean_oid(self):
        self.remove_W()
        extracted = self.df["placing"].astype(str).str.extract(r'^.*?(\d+)')
        self.df["placing"] = pd.to_numeric(extracted[0], errors='coerce').fillna(0).astype(int)
        self.df["draw"] = self.df["draw"].astype(int)
        self.df['finished_time_sec'] = self.df["finished_time"].apply(self.convert_min_to_sec)
        self.df['head_horse_dist_cleaned'] = self.df["head_horse_dist"].apply(self.clean_head_horse_dist)
        self.df['horse_id'] = self.df["horse_name"].apply(self.get_horse_id)
        self.df['horse_name'] = self.df["horse_name"].apply(self.clean_horse_name)
        self.df['class'] = self.df["races.basic_info"].str.slice(stop=3).apply(self.get_class)
        self.df['length'] = pd.to_numeric(self.df["races.basic_info"].str.slice(start=6,stop=10).str.extract(r'(\d+)')[0], errors='coerce')
        track_res = self.df["races.track_info"].apply(self.extract_track_info)
        self.df["track_texture"] = [x[0] for x in track_res]
        self.df["track_type"] = [x[1] for x in track_res]
        self.df = self.df[~self.df['class'].apply(lambda x: isinstance(x, str))]
        self.df = self.df.copy()
        print("目前真實的欄位名稱列表：", list(self.df.columns))
        self.df.drop(columns=["races.basic_info", "head_horse_dist", "finished_time", "track_info"], inplace=True)
    def save_json(self):
        with open(self.cleaned_json_path ,"w",encoding="utf-8") as f:
            self.df.to_json(f, orient="records",indent=4,force_ascii=False)
    


    
if __name__ == "__main__":
    raw_json_path = pathlib.Path.cwd() / "data/"  "raw_json"
    cleaned_json_path = pathlib.Path.cwd() /"data" / "cleaned_json" / "flatten" 
    rating_json_path = pathlib.Path.cwd() /"data" / "horses_rating.json"
    for file in raw_json_path.iterdir():
        print(file.name)
        # Ensure it is a file and not a folder
        if file.is_file():
            raw_file_path = raw_json_path / file.name
            target_file_path = cleaned_json_path / file.name
            try:
                data_cleaner = Data_cleaning_manager_flatten(raw_json_path=raw_file_path, cleaned_json_path=target_file_path, rating_json_path=c)
                data_cleaner.start_clean()
                data_cleaner.save_json()
            except:
                pass