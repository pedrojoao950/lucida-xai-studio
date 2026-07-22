# LÚCIDA Science API

Serviço Python para treino científico do Explainable AI Studio.

## Iniciar

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Documentação interativa: `http://127.0.0.1:8000/docs`.

O endpoint `POST /v1/train` recebe um CSV multipart e executa os cinco algoritmos sob o mesmo protocolo. As transformações são ajustadas dentro de cada fold para impedir leakage.
