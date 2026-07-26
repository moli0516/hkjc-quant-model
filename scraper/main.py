import asyncio
import sys
from typing import List
from race_pipeline import RaceScrapingPipeline


def validate_years(years: List[int]) -> bool:
    """
    驗證 years 參數的合法性:
    1. 必須是長度為 2 的清單 [start_year, end_year]
    2. 元素必須全為整數
    3. 起始年份必須小於或等於結束年份 (start_year <= end_year)
    """
    if not isinstance(years, list) or len(years) != 2:
        print("❌ 錯誤：`years` 必須是包含「首尾兩年」的長度為 2 的 List！例如: [2020, 2025]")
        return False

    start_year, end_year = years

    if not isinstance(start_year, int) or not isinstance(end_year, int):
        print("❌ 錯誤：`years` 中的年份必須是整數！")
        return False

    if start_year > end_year:
        print(f"❌ 錯誤：起始年份 ({start_year}) 不能大於結束年份 ({end_year})！例如應為 [{end_year}, {start_year}]")
        return False

    return True


async def main(target_years: List[int]) -> None:
    """主程序入口"""
    print(f"🚀 開始初始化賽事爬蟲 Pipeline...")
    
    # 1. 驗證年份參數
    if not validate_years(target_years):
        print("⛔ 參數驗證失敗，程式結束。")
        sys.exit(1)

    print(f"📅 目標年份範圍: {target_years[0]} 年 至 {target_years[1]} 年")

    # 2. 初始化 Pipeline 並執行下載
    try:
        pipeline = RaceScrapingPipeline(max_concurrent_days=5)
        await pipeline.run(target_years)
        print("🎉 所有賽事數據下載與解析任務完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 使用者手動中斷程序。")
    except Exception as e:
        print(f"\n💥 執行過程發生未預期例外錯誤: {e}")


if __name__ == "__main__":
    # 🌟 設定預設的執行年份 [首年, 尾年]
    # 可直接在這裡微調測試，例如: [2020, 2025]
    years_to_run = [2020, 2026]

    # 🌟 支援 CLI 命令行輸入參數 (例如: python main.py 2021 2024)
    if len(sys.argv) == 3:
        try:
            years_to_run = [int(sys.argv[1]), int(sys.argv[2])]
        except ValueError:
            print("❌ CLI 參數解析失敗：請輸入有效的年份整數！例如: python main.py 2020 2025")
            sys.exit(1)

    # 啟動 Asyncio Event Loop
    asyncio.run(main(years_to_run))