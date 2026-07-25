from config.settings import settings
import pandas as pd
import numpy as np
from models.inference import Quant_inference_engine
from models.race_card_loader import Race_card
import json

def main():
    raw_json = None
    with open(settings.today_rc_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    features = settings.latest_features + settings.PHYSICAL_FEATURES
    target = "is_win"
    
    race_card = Race_card(raw_data=raw_json)
    quant_interence_engine = Quant_inference_engine(features=features, target=target)
    
    race_card.run()

