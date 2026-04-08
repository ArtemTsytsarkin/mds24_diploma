# MOEX xLSTM RL pipeline

Готовый проект, собранный из твоих notebook'ов:

- `xlstm_moex_trading_01.ipynb`
- `xlstm_moex_short_full.ipynb`
- `xlstm_moex_short_v2.ipynb`
- `xlstm_architecture_comparison.ipynb`
- `notebookde920ba92f.ipynb`

## Что сохранено из тетрадок

В проект перенесена вся основная логика, а не переписана заново с нуля:

- загрузка OHLCV с MOEX через `apimoex`
- turbulence index
- short environment
- `margin_ratio=0.25` из improved short v2
- reward shaping для short-среды
- auto-close short при движении цены против позиции > 2%
- нормализация признаков только по train, чтобы убрать look-ahead bias
- технические индикаторы из `xlstm_moex_short_v2.ipynb` и `xlstm_architecture_comparison.ipynb`
- curriculum learning в 2 этапа: спокойный рынок -> полный период
- baseline `LSTM`
- `xLSTM base`
- `xLSTM + attention`
- `xLSTM large`
- единые метрики сравнения для всех моделей

## Структура

- `data.py` — загрузка данных и признаки
- `env.py` — trading env
- `models.py` — LSTM/xLSTM архитектуры
- `train.py` — обучение одной модели
- `experiments.py` — полный прогон сравнения
- `requirements.txt` — зависимости
- `notebooks/` — исходные тетрадки без изменений

## Установка

```bash
pip install -r requirements.txt
```

## Обучить одну модель

```bash
python train.py --model xlstm_base
python train.py --model lstm
python train.py --model xlstm_attn
python train.py --model xlstm_large
```

## Сравнить все модели

```bash
python experiments.py
```

## Что считается в сравнении

- `CR` — Cumulative Return
- `MER` — Max Earning Rate
- `MPB` — Max PullBack
- `APPT` — Avg Profit Per Trade
- `SR` — Sharpe Ratio
- `WinRate`
- `ProfitFactor`
- `Trades`
- `FinalValue`

## Замечания

1. `xlstm_large` может не поместиться в память на слабой GPU.
2. `sLSTM` включается только если `CUDA Compute Capability >= 8.0`.
3. Для честного сравнения все модели обучаются на одинаковых split'ах и одинаковых гиперпараметрах PPO.
4. Исходные тетрадки сохранены в папке `notebooks/`, чтобы ничего из исходников не потерять.
