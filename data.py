from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import apimoex
import numpy as np
import pandas as pd
import requests

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    GaussianHMM = None
    HMM_AVAILABLE = False

EPS = 1e-8


@dataclass
class DataConfig:
    tickers: List[str]
    train_start: str = "2009-01-01"
    train_end: str = "2024-07-31"
    val_start: str = "2024-08-01"
    val_end: str = "2025-07-31"
    test_start: str = "2025-08-02"
    test_end: str = pd.Timestamp.today().strftime("%Y-%m-%d")
    calm_start: str = "2013-01-01"
    calm_end: str = "2019-12-31"
    data_file: str = "moex_data_v3.csv"
    turbulence_lookback: int = 252
    turbulence_percentile: float = 85.0
    n_regimes: int = 3
    rvi_file: str = "rvi_data.csv"
    use_rvi: bool = False
    use_hmm: bool = False


DEFAULT_CONFIG = DataConfig(tickers=["SBER", "GAZP", "LKOH", "NVTK", "GMKN"],
                            test_end="2026-05-02")


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def download_moex_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    with requests.Session() as session:
        data = apimoex.get_board_history(
            session,
            ticker,
            start=start,
            end=end,
            columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"),
        )
    if not data:
        raise ValueError(f"Нет данных для {ticker} за период {start}–{end}")
    df = pd.DataFrame(data)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.set_index("TRADEDATE").sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.add_prefix(f"{ticker}_")
    return df.replace(0, np.nan)


def _required_columns(tickers: List[str]) -> List[str]:
    return [f"{t}_{c}" for t in tickers for c in ["open", "high", "low", "close", "volume"]]


def _cache_is_valid(df: pd.DataFrame, tickers: List[str], start: str, end: str) -> bool:
    required = set(_required_columns(tickers))
    if not required.issubset(df.columns):
        return False
    cache_end = pd.Timestamp(end) - pd.tseries.offsets.BDay(5)
    if df.index.min() > pd.Timestamp(start) or df.index.max() < cache_end:
        return False
    return True


def load_or_download(tickers: List[str], start: str, end: str, cache_file: str) -> pd.DataFrame:
    if os.path.exists(cache_file):
        try:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True).sort_index()
            if _cache_is_valid(cached, tickers, start, end):
                print(f"Загружаем из валидного кэша: {cache_file}")
                return cached[_required_columns(tickers)].copy()
            print(f"Кэш {cache_file} не соответствует tickers/period; перезагружаем.")
        except Exception as exc:
            print(f"Кэш {cache_file} не прочитан: {exc}; перезагружаем.")

    print("Скачиваем данные с MOEX ISS...")
    frames = []
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        frames.append(download_moex_ohlcv(ticker, start, end))
        print("OK")
    df = pd.concat(frames, axis=1).sort_index().ffill()
    df = df[_required_columns(tickers)]
    df.to_csv(cache_file)
    print(f"Сохранено в {cache_file}")
    return df


def extract_close_prices(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    return df[[f"{ticker}_close" for ticker in tickers]].copy()


def _ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=max(2, length // 2)).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / (roll_down + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    money = typical * volume
    sign = np.sign(typical.diff().fillna(0.0))
    pos = money.where(sign > 0, 0.0).rolling(length, min_periods=length).sum()
    neg = money.where(sign < 0, 0.0).rolling(length, min_periods=length).sum().abs()
    ratio = pos / (neg + EPS)
    return 100.0 - 100.0 / (1.0 + ratio)


def compute_turbulence(close_prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    returns = close_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    turbulence = pd.Series(index=returns.index, dtype=float, name="turbulence")
    for i in range(lookback, len(returns)):
        hist = returns.iloc[i - lookback : i]
        y = returns.iloc[i].values - hist.mean().values
        try:
            turb = float(y @ np.linalg.pinv(hist.cov().values) @ y)
        except Exception:
            turb = 0.0
        turbulence.iloc[i] = max(turb, 0.0)
    return turbulence.fillna(0.0)


def compute_raw_atr_values(df: pd.DataFrame, tickers: List[str]) -> np.ndarray:
    values = []
    for ticker in tickers:
        close = df[f"{ticker}_close"]
        high = df[f"{ticker}_high"]
        low = df[f"{ticker}_low"]
        atr_ratio = (_atr(high, low, close, 14) / (close.abs() + EPS)).replace([np.inf, -np.inf], np.nan)
        values.append(atr_ratio.ffill().fillna(0.02).clip(0.001, 0.25).values)
    return np.column_stack(values).astype(np.float32)


def _market_features(close_prices: pd.DataFrame) -> pd.DataFrame:
    market = close_prices.div(close_prices.iloc[0]).mean(axis=1)
    ret = np.log(market / market.shift(1)).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=close_prices.index)
    out["market_ret_1"] = ret
    out["market_ret_5"] = np.log(market / market.shift(5))
    out["market_ret_20"] = np.log(market / market.shift(20))
    out["market_vol_20"] = ret.rolling(20, min_periods=5).std()
    roll_max = market.rolling(60, min_periods=10).max()
    out["market_dd_60"] = (market - roll_max) / (roll_max + EPS)
    return out


def compute_features(df: pd.DataFrame, tickers: List[str], fit_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    parts: list[pd.DataFrame] = []
    names: list[str] = []

    for ticker in tickers:
        close = df[f"{ticker}_close"].astype(float)
        high = df[f"{ticker}_high"].astype(float)
        low = df[f"{ticker}_low"].astype(float)
        open_ = df[f"{ticker}_open"].astype(float)
        volume = df[f"{ticker}_volume"].astype(float)
        ret1 = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

        sub = pd.DataFrame(index=df.index)
        sub[f"{ticker}_open_rel"] = open_ / (close.shift(1) + EPS) - 1.0
        sub[f"{ticker}_high_rel"] = high / (close.shift(1) + EPS) - 1.0
        sub[f"{ticker}_low_rel"] = low / (close.shift(1) + EPS) - 1.0
        sub[f"{ticker}_close"] = close
        sub[f"{ticker}_log_volume"] = np.log1p(volume)
        sub[f"{ticker}_ret_1"] = ret1
        sub[f"{ticker}_ret_3"] = np.log(close / close.shift(3))
        sub[f"{ticker}_ret_5"] = np.log(close / close.shift(5))
        sub[f"{ticker}_ret_10"] = np.log(close / close.shift(10))
        sub[f"{ticker}_ret_20"] = np.log(close / close.shift(20))
        sub[f"{ticker}_vol_10"] = ret1.rolling(10, min_periods=5).std()
        sub[f"{ticker}_vol_20"] = ret1.rolling(20, min_periods=5).std()
        ema10 = _ema(close, 10)
        ema30 = _ema(close, 30)
        sub[f"{ticker}_trend_10_30"] = (ema10 - ema30) / (close.abs() + EPS)
        sma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        sub[f"{ticker}_z_20"] = (close - sma20) / (std20 + EPS)
        low20 = low.rolling(20, min_periods=10).min()
        high20 = high.rolling(20, min_periods=10).max()
        sub[f"{ticker}_price_pos_20"] = (close - low20) / (high20 - low20 + EPS)
        sub[f"{ticker}_rsi_14"] = _rsi(close, 14) / 100.0
        macd = _ema(close, 12) - _ema(close, 26)
        sub[f"{ticker}_macd_rel"] = macd / (close.abs() + EPS)
        sub[f"{ticker}_mom_10"] = close / (close.shift(10) + EPS) - 1.0
        bb_low = sma20 - 2.0 * std20
        bb_high = sma20 + 2.0 * std20
        sub[f"{ticker}_bb_pct"] = (close - bb_low) / (bb_high - bb_low + EPS)
        sub[f"{ticker}_atr_ratio"] = _atr(high, low, close, 14) / (close.abs() + EPS)
        sub[f"{ticker}_obv"] = _obv(close, volume)
        sub[f"{ticker}_mfi_14"] = _mfi(high, low, close, volume, 14) / 100.0
        sub = sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        parts.append(sub)
        names.extend(sub.columns.tolist())

    close_prices = extract_close_prices(df, tickers)
    market = _market_features(close_prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    parts.append(market)
    names.extend(market.columns.tolist())

    feat = pd.concat(parts, axis=1).astype(float)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        mean = feat.loc[fit_mask].mean(axis=0)
        std = feat.loc[fit_mask].std(axis=0).replace(0.0, 1.0) + EPS
    else:
        mean = feat.mean(axis=0)
        std = feat.std(axis=0).replace(0.0, 1.0) + EPS
    feat = ((feat - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat.values.astype(np.float32), names


def download_rvi(start: str, end: str, cache_file: str = "rvi_data.csv") -> pd.Series:
    if os.path.exists(cache_file):
        try:
            s = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
            s.name = "rvi"
            print(f"  RVI: загружен из кэша ({cache_file})")
            return s.astype(float)
        except Exception:
            pass
    try:
        with requests.Session() as session:
            data = apimoex.get_board_history(
                session,
                security="RVI",
                start=start,
                end=end,
                market="index",
                engine="stock",
                board="SNDX",
                columns=("TRADEDATE", "CLOSE"),
            )
        if data and len(data) > 30:
            df = pd.DataFrame(data)
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
            rvi = df.set_index("TRADEDATE")["CLOSE"].rename("rvi").replace(0, np.nan).ffill()
            rvi.to_csv(cache_file, header=True)
            print(f"  RVI: скачан с MOEX ISS ({len(rvi)} дней)")
            return rvi.astype(float)
    except Exception as exc:
        warnings.warn(f"RVI download failed: {exc}. Используем rolling volatility proxy.")
    return pd.Series(dtype=float, name="rvi")


def _normalize_series_by_train(series: pd.Series, train_mask: np.ndarray, index: pd.Index) -> np.ndarray:
    aligned = series.reindex(index).ffill().bfill().astype(float)
    train_vals = aligned.loc[np.asarray(train_mask, dtype=bool)]
    mn = float(train_vals.min()) if len(train_vals) else float(aligned.min())
    mx = float(train_vals.max()) if len(train_vals) else float(aligned.max())
    norm = ((aligned - mn) / (mx - mn + EPS)).clip(0.0, 1.0)
    return norm.fillna(0.5).values.astype(np.float32)


def compute_regimes(close_prices: pd.DataFrame, train_mask: np.ndarray, n_regimes: int = 3, random_state: int = 42) -> Optional[np.ndarray]:
    if not HMM_AVAILABLE:
        warnings.warn("hmmlearn не установлен; HMM regimes отключены.")
        return None
    log_ret = np.log(close_prices / close_prices.shift(1)).mean(axis=1).fillna(0.0)
    roll_vol = log_ret.rolling(20, min_periods=5).std().fillna(0.0)
    market = close_prices.div(close_prices.iloc[0]).mean(axis=1)
    roll_max = market.rolling(60, min_periods=10).max()
    drawdown = ((market - roll_max) / (roll_max + EPS)).fillna(0.0)
    X = np.column_stack([log_ret.values, roll_vol.values, drawdown.values]).astype(float)
    train_mask = np.asarray(train_mask, dtype=bool)
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0) + EPS
    Xn = (X - mean) / std
    try:
        model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=500, random_state=random_state, tol=1e-5)
        model.fit(Xn[train_mask])
        raw = np.zeros(len(Xn), dtype=np.int8)
        for i in range(len(Xn)):
            raw[i] = int(model.predict(Xn[: i + 1])[-1])
    except Exception as exc:
        print(f"  HMM fit/predict FAILED: {exc}")
        return None

    train_raw = raw[train_mask]
    dd_train = drawdown.values[train_mask]
    mean_dd = np.array([
        dd_train[train_raw == r].mean() if np.any(train_raw == r) else 0.0
        for r in range(n_regimes)
    ])
    order = np.argsort(mean_dd)
    regime_map = {old: new for new, old in enumerate(order)}
    regimes = np.array([regime_map[int(r)] for r in raw], dtype=np.int8)
    counts = {r: int((regimes == r).sum()) for r in range(n_regimes)}
    print(f"  HMM regimes causal (0=bull,1=sideways,2=bear): {counts}")
    return regimes


@dataclass
class DatasetBundle:
    raw: pd.DataFrame
    raw_aligned: pd.DataFrame
    dates_all: pd.DatetimeIndex
    close_prices: pd.DataFrame
    turbulence: pd.Series
    turb_aligned: pd.Series
    threshold: float
    features_all: np.ndarray
    feature_names: List[str]
    close_all: np.ndarray
    atr_all: np.ndarray
    features_train: np.ndarray
    close_train: np.ndarray
    turb_train: np.ndarray
    atr_train: np.ndarray
    features_val: np.ndarray
    close_val: np.ndarray
    turb_val: np.ndarray
    atr_val: np.ndarray
    features_test: np.ndarray
    close_test: np.ndarray
    turb_test: np.ndarray
    atr_test: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    calm_mask: np.ndarray
    test_gap_days: pd.Series
    feature_dim: int
    tickers: List[str]
    atr_indices: List[int]
    config: DataConfig
    regime_all: Optional[np.ndarray]
    regime_train: Optional[np.ndarray]
    regime_val: Optional[np.ndarray]
    regime_test: Optional[np.ndarray]
    rvi_all: Optional[np.ndarray]
    rvi_train: Optional[np.ndarray]
    rvi_val: Optional[np.ndarray]
    rvi_test: Optional[np.ndarray]


def prepare_datasets(config: DataConfig = DEFAULT_CONFIG) -> DatasetBundle:
    raw = load_or_download(config.tickers, config.train_start, config.test_end, config.data_file)
    close_cols = [f"{t}_close" for t in config.tickers]
    raw = raw.dropna(subset=close_cols).sort_index()
    print(f"Загружено: {raw.shape[0]} дней, {raw.shape[1]} колонок")

    close_prices = extract_close_prices(raw, config.tickers)
    print("Считаем turbulence index...")
    turbulence = compute_turbulence(close_prices, lookback=config.turbulence_lookback)
    train_turb = turbulence[config.train_start : config.train_end].dropna()
    if len(train_turb) == 0:
        raise ValueError("Train turbulence series is empty; check DataConfig dates.")
    turb_cap = float(np.percentile(train_turb, 99))
    turbulence = turbulence.clip(upper=turb_cap)
    threshold = float(np.percentile(turbulence[config.train_start : config.train_end].dropna(), config.turbulence_percentile))
    print(f"Turbulence cap train 99%={turb_cap:.2f}; threshold {config.turbulence_percentile:.0f}%={threshold:.2f}")

    common_idx = raw.index.intersection(turbulence.index)
    raw_aligned = raw.loc[common_idx].copy()
    turb_aligned = turbulence.loc[common_idx].copy()
    dates_all = raw_aligned.index

    train_mask = ((dates_all >= config.train_start) & (dates_all <= config.train_end)).astype(bool)
    val_mask = ((dates_all >= config.val_start) & (dates_all <= config.val_end)).astype(bool)
    test_mask = ((dates_all >= config.test_start) & (dates_all <= config.test_end)).astype(bool)
    calm_mask = ((dates_all >= config.calm_start) & (dates_all <= config.calm_end)).astype(bool)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            f"Empty split detected: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}. "
            "Check date ranges and cache coverage."
        )

    print("Вычисляем technical/cross-asset features...")
    features_all, feature_names = compute_features(raw_aligned, config.tickers, fit_mask=train_mask)
    atr_all = compute_raw_atr_values(raw_aligned, config.tickers)
    close_all = extract_close_prices(raw_aligned, config.tickers).values.astype(np.float32)

    if config.use_hmm:
        print("Вычисляем causal HMM volatility regimes...")
        regime_all = compute_regimes(extract_close_prices(raw_aligned, config.tickers), train_mask=train_mask, n_regimes=config.n_regimes)
    else:
        regime_all = None
        print("HMM отключён (use_hmm=False)")

    if config.use_rvi:
        print("Загружаем RVI...")
        rvi_series = download_rvi(config.train_start, config.test_end, config.rvi_file)
        if rvi_series.empty:
            market_ret = np.log(close_prices.iloc[:, 0] / close_prices.iloc[:, 0].shift(1))
            rvi_series = market_ret.rolling(20, min_periods=5).std().rename("rvi")
            print("  RVI proxy: rolling 20-day vol")
        rvi_all = _normalize_series_by_train(rvi_series, train_mask, dates_all)
    else:
        rvi_all = None
        print("RVI отключён (use_rvi=False)")

    features_train = features_all[train_mask]
    close_train = close_all[train_mask]
    turb_train = turb_aligned.values.astype(np.float32)[train_mask]
    atr_train = atr_all[train_mask]

    features_val = features_all[val_mask]
    close_val = close_all[val_mask]
    turb_val = turb_aligned.values.astype(np.float32)[val_mask]
    atr_val = atr_all[val_mask]

    features_test = features_all[test_mask]
    close_test = close_all[test_mask]
    turb_test = turb_aligned.values.astype(np.float32)[test_mask]
    atr_test = atr_all[test_mask]

    regime_train = regime_all[train_mask] if regime_all is not None else None
    regime_val = regime_all[val_mask] if regime_all is not None else None
    regime_test = regime_all[test_mask] if regime_all is not None else None
    rvi_train = rvi_all[train_mask] if rvi_all is not None else None
    rvi_val = rvi_all[val_mask] if rvi_all is not None else None
    rvi_test = rvi_all[test_mask] if rvi_all is not None else None

    test_dates = dates_all[test_mask]
    gaps = pd.Series(test_dates).diff().dt.days
    large_gaps = gaps[gaps > 5]
    if len(large_gaps) > 0:
        print(f"Большие пропуски в test (>5 дней): {large_gaps.tolist()}")
    else:
        print("Пропусков в test данных >5 дней нет.")

    atr_indices = [feature_names.index(f"{ticker}_atr_ratio") for ticker in config.tickers if f"{ticker}_atr_ratio" in feature_names]
    print(f"Train: {features_train.shape[0]} | Val: {features_val.shape[0]} | Test: {features_test.shape[0]}")
    print(f"Feature dim: {features_train.shape[1]} | ATR raw matrix: {atr_all.shape}")

    return DatasetBundle(
        raw=raw,
        raw_aligned=raw_aligned,
        dates_all=dates_all,
        close_prices=close_prices,
        turbulence=turbulence,
        turb_aligned=turb_aligned,
        threshold=threshold,
        features_all=features_all,
        feature_names=feature_names,
        close_all=close_all,
        atr_all=atr_all,
        features_train=features_train,
        close_train=close_train,
        turb_train=turb_train,
        atr_train=atr_train,
        features_val=features_val,
        close_val=close_val,
        turb_val=turb_val,
        atr_val=atr_val,
        features_test=features_test,
        close_test=close_test,
        turb_test=turb_test,
        atr_test=atr_test,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        calm_mask=calm_mask,
        test_gap_days=large_gaps,
        feature_dim=features_train.shape[1],
        tickers=config.tickers,
        atr_indices=atr_indices,
        config=config,
        regime_all=regime_all,
        regime_train=regime_train,
        regime_val=regime_val,
        regime_test=regime_test,
        rvi_all=rvi_all,
        rvi_train=rvi_train,
        rvi_val=rvi_val,
        rvi_test=rvi_test,
    )
