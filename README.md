# mlops-model
Repository holds everything necessary for model training and inference 

## Підготовка оточення

`dvc repro prepare-env -f`

## Тренування моделі

`dvc repro train-model -f`

## Інференс моделі

`dvc repro run-model -f`

Далі застосунок починає трекати папку `ray/ray-head/input`

І переміщує розпізнані зображення в `ray/ray-head/output`
