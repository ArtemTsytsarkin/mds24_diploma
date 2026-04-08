from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import apimoex
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

warnings.filterwarnings("ignore")


@dataclass
class DataConfig:
    tickers: List[str]
    train_start: str = "2009-01-01"
    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2024-01-01"
    calm_start: str = "2013-01-01"
    calm_end: str = "2019-12-31"
    data_file: str = "moex_data.csv"
    turbulence_lookback: int = 252
    turbulence_percentile: float = 75.0


DEFAULT_CONFIG = DataConfig(
    tickers=["SBER", "GAZP", "LKOH", "YNDX", "GMKN"],
)


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
    df = df.set_index("TRADEDATE")
    df.columns = [c.lower() for c in df.columns]
    df = df.add_prefix(f"{ticker}_")
    return df.replace(0, np.nan)


def load_or_download(tickers: List[str], start: str, end: str, cache_file: str) -> pd.DataFrame:
    if os.path.exists(cache_file):
        print(f"Загружаем из кэша: {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print("Скачиваем данные с MOEX ISS...")
    frames = []
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        frames.append(download_moex_ohlcv(ticker, start, end))
        print("OK")

    df = pd.concat(frames, axis=1).ffill().bfill()
    df.to_csv(cache_file)
    print(f"Сохранено в {cache_file}")
    return df


def extract_close_prices(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    return df[[f"{ticker}_close" for ticker in tickers]].copy()


def compute_turbulence(close_prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    returns = close_prices.pct_change().dropna()
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


def compute_features(df: pd.DataFrame, tickers: List[str], fit_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Объединяет логику из `xlstm_moex_short_v2.ipynb` и `xlstm_architecture_comparison.ipynb`.
    Сохраняет:
    - базовые OHLCV признаки
    - технические индикаторы (trend, ADX, RSI, MACD, MOM, Bollinger pct, ATR, OBV, MFI)
    - нормализацию только по train для устранения look-ahead bias
    """
    parts = []
    for ticker in tickers:
        close = df[f"{ticker}_close"]
        high = df[f"{ticker}_high"]
        low = df[f"{ticker}_low"]
        volume = df[f"{ticker}_volume"]

        sub = df[[f"{ticker}_{c}" for c in ["open", "high", "low", "close", "volume"]]].copy()
        sub[f"{ticker}_trend"] = (ta.ema(close, length=10) - ta.ema(close, length=30)) / (close.abs() + 1e-8)
        sub[f"{ticker}_adx"] = ta.adx(high, low, close, length=14)["ADX_14"] / 100
        sub[f"{ticker}_rsi"] = ta.rsi(close, length=14) / 100

        macd = ta.macd(close)
        sub[f"{ticker}_macd"] = macd["MACD_12_26_9"] / (close.abs() + 1e-8)
        sub[f"{ticker}_mom"] = ta.mom(close, length=10) / (close.abs() + 1e-8)

        bb = ta.bbands(close, length=20)
        sub[f"{ticker}_bb_pct"] = (
            (close - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"] + 1e-8)
        )
        sub[f"{ticker}_atr"] = ta.atr(high, low, close, length=14) / (close.abs() + 1e-8)
        sub[f"{ticker}_obv"] = ta.obv(close, volume)
        sub[f"{ticker}_mfi"] = ta.mfi(high, low, close, volume, length=14) / 100
        sub = sub.fillna(0.0)

        if fit_mask is not None:
            mean = sub[fit_mask].mean()
            std = sub[fit_mask].std() + 1e-8
        else:
            mean = sub.mean()
            std = sub.std() + 1e-8

        sub = (sub - mean) / std
        parts.append(sub.values)

    return np.concatenate(parts, axis=1).astype(np.float32)


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
    close_all: np.ndarray
    features_train: np.ndarray
    close_train: np.ndarray
    turb_train: np.ndarray
    features_test: np.ndarray
    close_test: np.ndarray
    turb_test: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    calm_mask: np.ndarray
    test_gap_days: pd.Series
    feature_dim: int
    tickers: List[str]
    config: DataConfig


def prepare_datasets(config: DataConfig = DEFAULT_CONFIG) -> DatasetBundle:
    raw = load_or_download(config.tickers, config.train_start, config.test_end, config.data_file)
    print(f"Загружено: {raw.shape[0]} дней, {raw.shape[1]} колонок")

    close_prices = extract_close_prices(raw, config.tickers)
    print("Считаем turbulence index (~1–2 мин)...")
    turbulence = compute_turbulence(close_prices, lookback=config.turbulence_lookback)

    threshold = float(
        np.percentile(turbulence[config.train_start : config.train_end].dropna(), config.turbulence_percentile)
    )
    print(f"Turbulence threshold ({config.turbulence_percentile:.0f}th pct): {threshold:.2f}")

    common_idx = raw.index.intersection(turbulence.dropna().index)
    raw_aligned = raw.loc[common_idx]
    turb_aligned = turbulence.loc[common_idx]
    dates_all = raw_aligned.index

    train_mask = ((dates_all >= config.train_start) & (dates_all <= config.train_end)).astype(bool)
    test_mask = ((dates_all >= config.test_start) & (dates_all <= config.test_end)).astype(bool)
    calm_mask = ((dates_all >= config.calm_start) & (dates_all <= config.calm_end)).astype(bool)

    print("Вычисляем технические индикаторы...")
    features_all = compute_features(raw_aligned, config.tickers, fit_mask=train_mask)
    close_all = close_prices.loc[common_idx].values.astype(np.float32)

    features_train = features_all[train_mask]
    close_train = close_all[train_mask]
    turb_train = turb_aligned[train_mask].values.astype(np.float32)

    features_test = features_all[test_mask]
    close_test = close_all[test_mask]
    turb_test = turb_aligned[test_mask].values.astype(np.float32)

    test_dates = dates_all[test_mask]
    gaps = pd.Series(test_dates).diff().dt.days
    large_gaps = gaps[gaps > 5]
    if len(large_gaps) > 0:
        print("\nБольшие пропуски в тестовых данных (>5 дней):")
        print(large_gaps)
    else:
        print("\nПропусков в тестовых данных нет.")

    print(f"Train: {features_train.shape[0]} дней")
    print(f"Test:  {features_test.shape[0]} дней")
    print(f"Feature dim: {features_train.shape[1]}")

    return DatasetBundle(
        raw=raw,
        raw_aligned=raw_aligned,
        dates_all=dates_all,
        close_prices=close_prices,
        turbulence=turbulence,
        turb_aligned=turb_aligned,
        threshold=threshold,
        features_all=features_all,
        close_all=close_all,
        features_train=features_train,
        close_train=close_train,
        turb_train=turb_train,
        features_test=features_test,
        close_test=close_test,
        turb_test=turb_test,
        train_mask=train_mask,
        test_mask=test_mask,
        calm_mask=calm_mask,
        test_gap_days=large_gaps,
        feature_dim=features_train.shape[1],
        tickers=config.tickers,
        config=config,
    )
