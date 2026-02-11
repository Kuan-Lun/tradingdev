# Architecture Overview

## Class Diagram

```mermaid
classDiagram
    direction TB

    %% ──────────────── ABC 基底類別 ────────────────

    class BaseCrawler {
        <<abstract>>
        +fetch(symbol, timeframe, start, end) pd.DataFrame*
        +save_raw(df, output_path) None*
    }

    class BaseIndicator {
        <<abstract>>
        +calculate(df) pd.DataFrame*
        +get_parameters() dict*
    }

    class BaseStrategy {
        <<abstract>>
        +fit(df) None
        +generate_signals(df) pd.DataFrame*
        +get_parameters() dict*
    }

    class BaseModel {
        <<abstract>>
        +train(df, eval_df) None*
        +predict(df) pd.Series*
    }

    %% ──────────────── Crawler 實作 ────────────────

    class BinanceAPICrawler {
        -_exchange: ccxt.binance
        +fetch(symbol, timeframe, start, end) pd.DataFrame
        +save_raw(df, output_path) None
    }

    class BinanceVisionCrawler {
        +fetch(symbol, timeframe, start, end) pd.DataFrame
        +save_raw(df, output_path) None
    }
    note for BinanceVisionCrawler "Skeleton — 尚未實作"

    BaseCrawler <|-- BinanceAPICrawler
    BaseCrawler <|-- BinanceVisionCrawler

    %% ──────────────── Indicator 實作 ────────────────

    class KDIndicator {
        -_k_period: int
        -_d_period: int
        -_smooth_k: int
        +calculate(df) pd.DataFrame
        +get_parameters() dict
    }

    BaseIndicator <|-- KDIndicator

    %% ──────────────── Strategy 實作 ────────────────

    class KDStrategy {
        -_config: KDStrategyConfig
        -_indicator: KDIndicator
        -_fit_config: KDFitConfig | None
        -_backtest_engine: BacktestEngine | None
        +fit(df) None
        +generate_signals(df) pd.DataFrame
        +get_parameters() dict
    }

    class XGBoostStrategy {
        -_config: XGBoostStrategyConfig
        -_model: XGBoostDirectionModel | None
        -_feature_engineer: FeatureEngineer | None
        -_best_lookback: int | None
        +fit(df) None
        +generate_signals(df) pd.DataFrame
        +get_parameters() dict
    }

    BaseStrategy <|-- KDStrategy
    BaseStrategy <|-- XGBoostStrategy
    KDStrategy *-- KDIndicator : 組合
    KDStrategy *-- KDStrategyConfig : 組合
    XGBoostStrategy *-- XGBoostStrategyConfig : 組合
    XGBoostStrategy *-- XGBoostDirectionModel : 組合
    XGBoostStrategy *-- FeatureEngineer : 組合

    %% ──────────────── ML 模組 ────────────────

    class XGBoostDirectionModel {
        -_config: XGBoostModelConfig
        -_model: XGBClassifier | None
        -_label_encoder: LabelEncoder
        +train(df, eval_df) None
        +predict(df) pd.Series
        +predict_proba(df) pd.DataFrame
        +get_parameters() dict
    }

    class FeatureEngineer {
        -_lookback: int
        -_feature_names: list
        +transform(df, include_target) pd.DataFrame
        +get_feature_names() list
    }

    BaseModel <|-- XGBoostDirectionModel

    %% ──────────────── Validation 模組 ────────────────

    class WalkForwardValidator {
        -_config: WalkForwardConfig
        -_engine: BacktestEngine
        +validate(strategy, df) list~WalkForwardResult~
        -_split_data(df) list
        +summary(results)$ dict
    }

    class WalkForwardResult {
        <<dataclass>>
        +fold_index: int
        +train_start: datetime
        +train_end: datetime
        +test_start: datetime
        +test_end: datetime
        +train_metrics: dict
        +test_metrics: dict
        +strategy_params: dict
    }

    WalkForwardValidator *-- WalkForwardConfig : 組合
    WalkForwardValidator *-- BacktestEngine : 組合
    WalkForwardValidator ..> WalkForwardResult : 產生
    WalkForwardValidator ..> BaseStrategy : 呼叫 fit/generate_signals

    %% ──────────────── Pydantic 資料模型 ────────────────

    class OHLCVBar {
        <<pydantic>>
        +timestamp: datetime
        +open: float
        +high: float
        +low: float
        +close: float
        +volume: float
    }

    class BacktestConfig {
        <<pydantic>>
        +symbol: str
        +timeframe: str
        +start_date: datetime
        +end_date: datetime
        +init_cash: float = 10000.0
        +fees: float = 0.0006
        +slippage: float = 0.0005
    }

    class KDStrategyConfig {
        <<pydantic>>
        +k_period: int = 14
        +d_period: int = 3
        +smooth_k: int = 3
        +overbought: float = 80.0
        +oversold: float = 20.0
    }

    class KDFitConfig {
        <<pydantic>>
        +k_period_range: list~int~
        +d_period_range: list~int~
        +smooth_k_range: list~int~
        +overbought_range: list~float~
        +oversold_range: list~float~
        +target_metric: str = "sharpe_ratio"
    }

    class WalkForwardConfig {
        <<pydantic>>
        +train_start: datetime | None
        +train_end: datetime | None
        +test_start: datetime | None
        +test_end: datetime | None
        +n_splits: int = 5
        +train_ratio: float = 0.7
        +expanding: bool = False
        +target_metric: str = "sharpe_ratio"
    }

    class XGBoostModelConfig {
        <<pydantic>>
        +n_estimators: int = 100
        +max_depth: int = 6
        +learning_rate: float = 0.1
        +subsample: float = 0.8
        +colsample_bytree: float = 0.8
        +early_stopping_rounds: int | None
        +random_state: int = 42
    }

    class XGBoostStrategyConfig {
        <<pydantic>>
        +model: XGBoostModelConfig
        +lookback_candidates: list~int~
        +retrain_interval: int = 24
        +validation_ratio: float = 0.2
        +signal_threshold: float = 0.5
    }

    %% ──────────────── 資料管線 ────────────────

    class DataLoader {
        +load_parquet(path) pd.DataFrame
        +load_csv(path) pd.DataFrame
    }

    class DataProcessor {
        +process(raw_df) pd.DataFrame
        +save_processed(df, output_path) None
    }

    %% ──────────────── 回測引擎 ────────────────

    class BacktestEngine {
        -_init_cash: float
        -_fees: float
        -_slippage: float
        -_freq: str
        +run(df) dict
    }

    %% ──────────────── 工具模組 ────────────────

    class config_mod ["utils/config"] {
        +load_config(path) dict
    }

    class logger_mod ["utils/logger"] {
        +setup_logger(name, level) Logger
    }

    class metrics_mod ["backtest/metrics"] {
        +calculate_metrics(pf) dict
        +format_metrics_report(metrics) str
    }

    BacktestEngine ..> metrics_mod : 使用
```

## Pipeline Sequence Diagram

### 單次回測（向後相容）

```mermaid
sequenceDiagram
    autonumber
    participant Main as main()
    participant Cfg as load_config
    participant Crawler as BinanceAPICrawler
    participant Proc as DataProcessor
    participant Loader as DataLoader
    participant Strat as KDStrategy
    participant Ind as KDIndicator
    participant Engine as BacktestEngine
    participant Metrics as calculate_metrics

    Main->>Cfg: load_config(yaml_path)
    Cfg-->>Main: dict (raw config)
    Main->>Main: BacktestConfig(**config["backtest"])
    Main->>Main: KDStrategyConfig(**config["strategy"]["parameters"])

    rect rgb(230, 245, 255)
        Note over Crawler,Proc: 資料取得與清洗
        Main->>Crawler: fetch(symbol, timeframe, start, end)
        Crawler-->>Main: raw DataFrame (OHLCV)
        Main->>Crawler: save_raw(df, raw_path)
        Main->>Proc: process(raw_df)
        Proc-->>Main: cleaned DataFrame
        Main->>Proc: save_processed(df, processed_path)
    end

    rect rgb(255, 245, 230)
        Note over Strat,Ind: 信號產生
        Main->>Strat: generate_signals(df)
        Strat->>Ind: calculate(df)
        Ind-->>Strat: df + stoch_k, stoch_d
        Strat-->>Main: df + signal (1 / -1 / 0)
    end

    rect rgb(230, 255, 230)
        Note over Engine,Metrics: 回測執行
        Main->>Engine: run(df_with_signals)
        Note right of Engine: shift signal by 1 bar<br/>轉換 entry/exit 布林陣列<br/>建立 vectorbt Portfolio
        Engine->>Metrics: calculate_metrics(portfolio)
        Metrics-->>Engine: metrics dict
        Engine-->>Main: metrics dict
        Main->>Main: format_metrics_report(metrics)
    end
```

### Walk-Forward 驗證流程

```mermaid
sequenceDiagram
    autonumber
    participant Main as main()
    participant WFV as WalkForwardValidator
    participant Strat as Strategy (KD / XGBoost)
    participant Engine as BacktestEngine

    Main->>Main: load_config → 偵測 validation 區塊
    Main->>Main: _create_strategy(config) → Strategy

    rect rgb(245, 235, 255)
        Note over Main,WFV: 初始化驗證器
        Main->>WFV: WalkForwardValidator(wf_config, engine)
        Main->>WFV: validate(strategy, df)
    end

    rect rgb(255, 245, 230)
        Note over WFV,Engine: 每個 Fold 重複
        loop 每個 (train_df, test_df) 分割
            WFV->>Strat: fit(train_df)
            Note right of Strat: KD: grid search 最佳參數<br/>XGBoost: 選 lookback + 訓練模型

            WFV->>Strat: generate_signals(train_df)
            WFV->>Engine: run(train_signals) → train_metrics

            WFV->>Strat: generate_signals(test_df)
            Note right of Strat: XGBoost: 每 N bar 滾動重訓
            WFV->>Engine: run(test_signals) → test_metrics

            WFV->>WFV: 收集 WalkForwardResult
        end
    end

    WFV-->>Main: list[WalkForwardResult]
    Main->>Main: format_walk_forward_report(results)
```

## Module Dependency Graph

```mermaid
graph TD
    subgraph Entry["進入點"]
        MAIN["main.py"]
    end

    subgraph Crawlers["crawlers/"]
        BC[BaseCrawler]
        BAC[BinanceAPICrawler]
        BVC[BinanceVisionCrawler]
        BC --> BAC
        BC --> BVC
    end

    subgraph Data["data/"]
        SCH["schemas.py<br/>OHLCVBar · BacktestConfig<br/>KDStrategyConfig · KDFitConfig<br/>WalkForwardConfig<br/>XGBoostModelConfig<br/>XGBoostStrategyConfig"]
        DL[DataLoader]
        DP[DataProcessor]
    end

    subgraph Indicators["indicators/"]
        BI[BaseIndicator]
        KDI[KDIndicator]
        BI --> KDI
    end

    subgraph Strategies["strategies/"]
        BS[BaseStrategy]
        KDS[KDStrategy]
        XGBS[XGBoostStrategy]
        BS --> KDS
        BS --> XGBS
    end

    subgraph Backtest["backtest/"]
        BE[BacktestEngine]
        MT[metrics.py]
    end

    subgraph Validation["validation/"]
        WFV[WalkForwardValidator]
        WFR[WalkForwardResult]
    end

    subgraph ML["ml/"]
        BM[BaseModel]
        XGBM[XGBoostDirectionModel]
        FE[FeatureEngineer]
        BM --> XGBM
    end

    subgraph Utils["utils/"]
        CFG[config.py]
        LOG[logger.py]
    end

    MAIN --> CFG
    MAIN --> LOG
    MAIN --> SCH
    MAIN --> BAC
    MAIN --> DL
    MAIN --> DP
    MAIN --> KDS
    MAIN --> XGBS
    MAIN --> BE
    MAIN --> WFV

    KDS --> KDI
    KDS --> SCH
    KDS --> BE

    XGBS --> XGBM
    XGBS --> FE
    XGBS --> SCH

    WFV --> BE
    WFV --> SCH
    WFV --> WFR
    WFV -.-> BS

    BE --> MT

    style BVC stroke-dasharray: 5 5
```
