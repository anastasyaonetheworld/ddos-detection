# DDoS Detection for IoT / Edge

Обнаружение DDoS-атак (бинарная классификация: атака / нормальный трафик) для слабых устройств — IoT, IIoT, edge-шлюзы. Работает на CPU, GPU не нужен.

## Модели

| Файл | Модель | Суть | Результаты |
|---|---|---|---|
| `simple_mlp.ipynb` | **Base MLP** (простая) | 4-слойный перцептрон, обучается отдельно на каждом датасете | KDD99 — 99.96%, CIC-DDoS2019 — ~95%, Edge-IIoTset — ~97%; ~0.1 мс на пример |
| `hybrid_transfer_cnn.ipynb` | **Hybrid Transfer CNN** (сложная) | Лёгкая CNN + transfer learning (KDD → CIC-DDoS2019 → Edge-IIoTset), CBAM-attention, ~44k параметров | Edge-IIoTset — 99.86% (F1 99.78%), CIC-DDoS2019 — ~95%, KDD — ~88%; ~0.57 мс на пример |

## Датасеты

В репозиторий не входят, скачиваются отдельно:
- KDD Cup 1999
- CIC-DDoS2019
- Edge-IIoTset

Путь к данным нужно указать в ноутбуке (ячейка 3).

## Зависимости

```
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm joblib
```

## Источник

Ноутбуки взяты из [MickeyKas/DDoSUsingMLP](https://github.com/MickeyKas/DDoSUsingMLP)
(авторы: Mikiyas Alemayehu, Mohamed Chahine Ghanem; статья *Edge-Optimized Deep and Transfer Learning for Efficient DDoS Detection in IIoT Networks*), лицензия MIT.
Оригинальные имена файлов: `DLDDoS.ipynb` → `simple_mlp.ipynb`, `Transfer.ipynb` → `hybrid_transfer_cnn.ipynb`.
