import pandas as pd
import json
import pathlib
import os
import re
\

class Data_cleaning_manager_flatten:
    def __init__(self, raw_json_path, cleaned_json_path):
        self.raw_json_path = raw_json_path
        self.cleaned_json_path = cleaned_json_path
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
                        "多個馬身": 99.0, "未能完成賽事": None, "退出": None
                        }
                if hhd in margin_map:
                    return margin_map[hhd]
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
        self.df[["track_texture", "track_type"]] =  self.df["races.track_info"].apply(self.extract_track_info).tolist()
        self.df = self.df[~self.df['class'].apply(lambda x: isinstance(x, str))]
        
        self.df = self.df.copy()
    def save_json(self):
        with open(self.cleaned_json_path ,"w",encoding="utf-8") as f:
            self.df.to_json(f, orient="records",indent=4,force_ascii=False)
    


    
if __name__ == "__main__":
    raw_json_path = pathlib.Path.cwd() / "data/"  "raw_json"
    cleaned_json_path = pathlib.Path.cwd() /"data" / "cleaned_json" / "flatten" 
    for file in raw_json_path.iterdir():
        print(file.name)
        # Ensure it is a file and not a folder
        if file.is_file():
            raw_file_path = raw_json_path / file.name
            target_file_path = cleaned_json_path / file.name
            try:
                data_cleaner = Data_cleaning_manager_flatten(raw_json_path=raw_file_path, cleaned_json_path=target_file_path)
                data_cleaner.start_clean()
                data_cleaner.save_json()
            except:
                pass