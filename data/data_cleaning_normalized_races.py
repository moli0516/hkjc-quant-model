import pandas as pd
import json
import pathlib
import os
import re


class Data_cleaning_manager_normalize_race:
    def __init__(self, raw_json_path, cleaned_json_path):
        self.raw_json_path = raw_json_path
        self.cleaned_json_path = cleaned_json_path
        self.df = self.create_df()
    def create_df(self):
        raw_json = None
        with open(self.raw_json_path, "r", encoding='utf-8') as f:
            raw_json = json.load(f)

        venue = raw_json.get("venue")
        date = raw_json.get("date")
        
        # 2. 提取 races 列表
        races_list = raw_json.get("races", [])

        # 3. 直接將 races 轉換成 DataFrame
        df = pd.DataFrame(races_list)

        # 4. 手動把外層的 venue 和 date 補進去（Pandas 會自動廣播到每一列）
        df["venue"] = venue
        df["date"] = date
        df = df.drop(columns=['horses'], errors='ignore')
        return df
    
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
    
    def start_clean(self):
        self.df['class'] = self.df["basic_info"].str.slice(stop=3).apply(self.get_class)
        self.df['length'] = pd.to_numeric(self.df["basic_info"].str.slice(start=6,stop=10).str.extract(r'(\d+)')[0], errors='coerce')
        self.df[["track_texture", "track_type"]] =  self.df["track_info"].apply(self.extract_track_info).tolist()
        self.df = self.df[~self.df['class'].apply(lambda x: isinstance(x, str))]
        self.df = self.df.copy()
    def save_json(self):
        with open(self.cleaned_json_path ,"w",encoding="utf-8") as f:
            self.df.to_json(f, orient="records",indent=4,force_ascii=False)
    


    
if __name__ == "__main__":
    raw_json_path = pathlib.Path.cwd() / "data" /  "raw_json"
    cleaned_json_path = pathlib.Path.cwd() /"data" / "cleaned_json" / "normalized" / "races"
    for file in raw_json_path.iterdir():
        print(file.name)
        # Ensure it is a file and not a folder
        if file.is_file():
            raw_file_path = raw_json_path / file.name
            target_file_path = cleaned_json_path / file.name
            try:
                data_cleaner = Data_cleaning_manager_normalize_race(raw_json_path=raw_file_path, cleaned_json_path=target_file_path)
                data_cleaner.start_clean()
                data_cleaner.save_json()
            except:
                print("I am gay")