import logging
from typing import Callable, Dict, Any, Optional
import optuna

from models.model_pipeline import ModelPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    """Optuna 自動超參數尋優器 (Hyperparameter Tuner)
    
    支援單次驗證 (方法 1) 與交叉驗證 (方法 2) 兩類尋優模式。
    """

    def __init__(
        self,
        pipeline: ModelPipeline,
        model_name: str = "xgb_ranker",
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
        direction: str = "maximize",
        use_cv: bool = False,
        n_splits: int = 3,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.val_days = val_days
        self.metric_name = metric_name
        self.direction = direction
        self.use_cv = use_cv
        self.n_splits = n_splits

    def _create_objective(self, param_fn: Callable[[optuna.Trial], Dict[str, Any]]) -> Callable[[optuna.Trial], float]:
        def objective(trial: optuna.Trial) -> float:
            model_params = param_fn(trial)

            try:
                if self.use_cv:
                    # 方法 2：跨時間序列 CV 平均指標尋優
                    _, metrics = self.pipeline.run_cv_train_pipeline(
                        model_name=self.model_name,
                        model_params=model_params,
                        n_splits=self.n_splits,
                        val_days=self.val_days,
                    )
                else:
                    # 方法 1：單次驗證集指標尋優
                    _, metrics = self.pipeline.run_train_pipeline(
                        model_name=self.model_name,
                        model_params=model_params,
                        val_days=self.val_days,
                    )

                score = metrics.get(self.metric_name, 0.0)
                return float(score)

            except Exception as e:
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
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_name = study_name or f"{self.model_name}_opt"
        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(),
        )

        mode_str = "方法 2 [TimeSeries CV]" if self.use_cv else "方法 1 [Single Holdout]"
        logger.info(f"🚀 開始 Optuna 自動超參數尋優 ({mode_str}, 輪數: {n_trials}, 目標: {self.metric_name})...")

        objective_fn = self._create_objective(param_fn)
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(f"🏆 尋優完成！最佳指標 [{self.metric_name}]: {study.best_value:.4f}")
        return study