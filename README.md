# Глубокое обучение · ИТМО

Домашние задания по курсу глубокого обучения, направление «Инженерия ИИ», 2 курс.

**Автор:** Виктор Васькин

| ДЗ | Тема | Статус |
|---|---|---|
| [ДЗ-1](hw1/) | Свёртки, обучение глубоких сетей, adversarial robustness | ✅ сдано, основная часть + бонусы 7.1–7.5 |
| [ДЗ-2](hw2/) | NER на русском: ruBert-base + NEREL | ✅ сдано, основная часть + бонусы 6.2-6.5 |

## Структура

```
.
├── hw1/
│   ├── hw1.ipynb      # ноутбук с решением и outputs финального прогона
│   ├── artifacts/     # metrics.json и checkpoints моделей
│   └── assets/        # графики для README
├── hw2/
│   ├── hw2_nlp.ipynb  # NER, ноутбук с outputs финального прогона
│   ├── artifacts/     # metrics.json и bonus.json
│   └── assets/        # графики для README
└── requirements.txt
```

## Запуск

```bash
python -m pip install -r requirements.txt
jupyter notebook hw1/hw1.ipynb
```

Датасеты (CIFAR-10, FashionMNIST) скачиваются автоматически в `./data` при первом запуске. Для разделов с обучением нужен GPU, финальные прогоны делались в Google Colab на Tesla T4.
