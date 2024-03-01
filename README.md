# Алгоритм отрисовки зоны для движения
Этот алгоритм описывает как из кадра видеопотока строиться зона для движения.
## Оглавление
- [Концептуальное описание](#общее)
- [1. Сегментация кадра](#сегментация)
- [2. Тепловая карта](#heatmap)
- [3. Зона для движения](#зона)

## Концептуальное описание
Вся система состоит из следующих частей:
1) Модуль обучения нейронной сети (data_preprocessing.py). Занимается предобработкой данных, созданием, настройкой и обучением нейронной сети для сегментации. Модуль настроен на создание и обучение нейронной сети U-net.
2) Нейронная сеть. Подходит любая нейронная сеть которая может выполнить многоклассовую попиксельную сегментацию входного кадра (т.е. используемая нейронная сеть не обязательно должна являться результатом работы модуля обучения нейронной сети (data_preprocessing.py)).
3) Входной видеопоток. Представляет собой любое видео, будь то записанное в формате, например, mp4 или потоковое. Поддерживается любое разрешение и любая частота кадров.
4) Обработчик видео потока. Принимает на вход нейросеть для сегментации и само видео. Покадрово обрабатывает входной видеопоток и на выходе даёт выходной видеопоток с размеченной зоной для движения автомобиля.
![Общая схема работы программного модуля](1.jpg)

## 1. Сегментация кадра

## 2. Тепловая карта

## 3. Зона для движения
