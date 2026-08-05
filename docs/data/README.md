Здесь лежит `anomalies.json`, который генерирует `src/export_web.py`.

Файл `anomalies.json.example` — синтетический пример той же структуры
(случайное поле с трендом), чтобы карту можно было открыть и проверить
вёрстку до того, как отработает загрузка данных. Для проверки:

    cp anomalies.json.example anomalies.json
    cd ../.. && python3 -m http.server 8000    # затем http://localhost:8000/docs/

Реальные данные перезапишут файл при следующем запуске export_web.py.
