# Прогнозирование нагрузки на электросеть

Проект по анализу временных рядов и прогнозированию почасового потребления
электроэнергии. В работе сравниваются три подхода: **ARIMA**,
**Prophet** и **NeuralProphet**.

## Авторы

- Кунёв Антон
- Уколов Вячеслав
- Группа: **33ИС**

## Цель работы

Цель проекта — построить модель, которая по историческим данным прогнозирует
нагрузку на электросеть на следующие 7 суток (168 часов).

В ходе работы выполняются:

1. загрузка и подготовка исходного набора данных;
2. исследовательский анализ временного ряда;
3. разделение данных на обучающую и тестовую выборки;
4. обучение моделей ARIMA, Prophet и NeuralProphet;
5. подбор гиперпараметров;
6. сравнение моделей по метрикам RMSE, MAE и MAPE;
7. сохранение лучшей модели и построение прогноза.

## Данные

Используется открытый набор данных
[Electricity Load Diagrams 2011–2014](https://archive.ics.uci.edu/dataset/321/electricityloadDiagrams20112014)
из репозитория UCI Machine Learning Repository.

Исходный файл `LD2011_2014.txt` имеет большой размер и поэтому не хранится
в GitHub-репозитории.

**Ссылка на датасет в Google Drive:** https://drive.google.com/file/d/1X0xBaVI3WeAdh_yxsT0zcTEDDKqt-x5A/view?usp=sharing

Также датасет можно загрузить автоматически:

```bash
python scripts/download_data.py
```

После загрузки файл должен находиться по пути:

```text
data/LD2011_2014.txt
```

В основном эксперименте используются данные за 2012 год. Исходные
15-минутные измерения объединяются в почасовой временной ряд, а значения всех
клиентов суммируются. Пропуски заполняются линейной интерполяцией.

## Используемые модели

| Модель | Настраиваемые параметры |
|---|---|
| ARIMA | порядок `(p, d, q)` |
| Prophet | `changepoint_prior_scale`, режим сезонности |
| NeuralProphet | скорость обучения, число эпох, количество лагов |

Для отслеживания параметров и метрик экспериментов используется **MLflow**.

## Результаты

Тестовая выборка включает последние 168 часов временного ряда.

| Модель | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Prophet, multiplicative | 126295.65 | 74140.41 | 13.33% |
| NeuralProphet, 100 эпох, 72 лага | 130224.13 | 81088.10 | 14.19% |
| NeuralProphet, 50 эпох, 24 лага | 132569.33 | 74613.01 | 12.86% |
| ARIMA (1, 1, 1) | 460016.96 | 404683.05 | 53.10% |

По основной метрике RMSE лучший результат показала модель **Prophet** с
мультипликативной сезонностью и `changepoint_prior_scale=0.05`.

Полная таблица находится в
[`models/experiment_results.csv`](models/experiment_results.csv).

## Структура проекта

```text
energy-load-forecasting/
├── data/                              # исходный датасет
├── models/
│   └── experiment_results.csv         # результаты экспериментов
├── notebooks/
│   └── energy_load_forecasting.ipynb  # основной ноутбук
├── scripts/
│   ├── download_data.py               # загрузка данных UCI
│   └── run_notebook.py                # автоматический запуск ноутбука
├── REPORT.md                           # отчёт по проекту
├── requirements.txt                    # зависимости Python
└── README.md
```

## Требования

- Python 3.10–3.12;
- не менее 4 ГБ оперативной памяти;
- Jupyter Notebook, JupyterLab или VS Code.

## Установка

Клонируйте репозиторий и перейдите в его каталог:

```bash
git clone https://github.com/MiniMaryo/energy-load-forecasting.git
cd energy-load-forecasting
```

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте его в Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Установите зависимости:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Загрузите датасет скриптом или поместите скачанный из Google Drive файл в
каталог `data`:

```bash
python scripts/download_data.py
```

## Запуск

Откройте основной ноутбук:

```bash
jupyter notebook notebooks/energy_load_forecasting.ipynb
```

Затем выполните все ячейки командой **Run All**.

Ноутбук также можно выполнить автоматически:

```bash
python scripts/run_notebook.py
```

Для просмотра сохранённых экспериментов MLflow:

```bash
mlflow ui --backend-store-uri mlruns
```

После запуска интерфейс будет доступен по адресу
<http://127.0.0.1:5000>.

## Основные библиотеки

- `pandas`, `numpy` — обработка данных;
- `matplotlib`, `seaborn` — визуализация;
- `statsmodels`, `pmdarima` — модель ARIMA;
- `prophet` — модель Prophet;
- `neuralprophet` — нейросетевая модель NeuralProphet;
- `scikit-learn` — расчёт метрик;
- `mlflow` — журналирование экспериментов.

## Воспроизводимость

В ноутбуке зафиксирован `RANDOM_SEED = 42`. Параметры и метрики запусков
записываются в MLflow, а итоговые результаты сохраняются в каталоге `models`.
