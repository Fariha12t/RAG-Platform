.PHONY: install test seed api ui eval docker
install:  ; pip install -r requirements-dev.txt
test:     ; python -m pytest -q
seed:     ; python -m scripts.seed_demo
api:      ; uvicorn app.api.main:app --reload --port 8000
ui:       ; streamlit run ui/streamlit_app.py
eval:     ; python -m evaluation.run_eval
docker:   ; docker compose up --build
