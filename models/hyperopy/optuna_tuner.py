import logging
from typing import Callable, Dict, Any, Optional
import optuna

from models.model_pipeline import ModelPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Optuna 自動超參數尋優器 (Hyperparameter Tuner)
    封裝對 ModelPipeline 的調用，防範資料洩漏並集中管理搜尋實驗。
    """

    def __init__(
        self,
        pipeline: ModelPipeline,
        model_name: str = "xgb_ranker",
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
        direction: str = "maximize",
    ):
        """
        :param pipeline: 已初始化的 ModelPipeline 實例
        :param model_name: 在 ModelRegistry 註冊的模型名稱 (例如 'xgb_ranker')
        :param val_days: 驗證集切分天數
        :param metric_name: 評估指標名稱 ('top1_win_rate', 'ndcg@5', 'top3_rate')
        :param direction: 優化方向 ('maximize' 或 'minimize')
        """
        self.pipeline = pipeline
        self.model_name = model_name
        self.val_days = val_days
        self.metric_name = metric_name
        self.direction = direction

    def _create_objective(self, param_fn: Callable[[optuna.Trial], Dict[str, Any]]) -> Callable[[optuna.Trial], float]:
        """建立內部使用的 Objective 函數"""

        def objective(trial: optuna.Trial) -> float:
            # 1. 透過外部傳入的 param_fn 生成該輪 Trial 的超參數組合
            model_params = param_fn(trial)

            try:
                # 2. 調用 Pipeline 進行標準訓練與驗證 (自動處理 TimeSplit)
                _, metrics = self.pipeline.run_train_pipeline(
                    model_name=self.model_name,
                    model_params=model_params,
                    val_days=self.val_days,
                )

                # 3. 提取指定的評估指標
                score = metrics.get(self.metric_name, 0.0)
                return float(score)

            except Exception as e:
                # 防禦機制：若極端參數導致崩潰，給予低分並跳過
                logger.warning(f"⚠️ Trial #{trial.number} 執行異常: {e}")
                return 0.0 if self.direction == "maximize" else 999.0

        return objective

    def optimize(
        self,
        param_fn: Callable[[optuna.Trial], Dict[str, Any]],
        n_trials: int = 30,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> optuna.Study:
        """
        執行自動調參流程
        
        :param param_fn: 接受 trial 並回傳 model_params 字典的函數
        :param n_trials: 試驗輪數
        :param timeout: 最大搜尋時間限制 (秒)
        :param study_name: 實驗名稱
        :return: 完成後的 Optuna Study 物件
        """
        # 隱藏 Optuna 過多的預設資訊
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_name = study_name or f"{self.model_name}_optimization"
        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(),
        )

        logger.info(
            f"🚀 開始執行 Optuna 自動超參數尋優 (模型: {self.model_name}, 輪數: {n_trials}, 優化指標: {self.metric_name})..."
        )

        objective_fn = self._create_objective(param_fn)
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(f"🏆 尋優完成！最佳指標 [{self.metric_name}]: {study.best_value:.4f}")
        return study