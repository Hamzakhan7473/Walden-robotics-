PYTHON ?= python

.PHONY: data outputs app test

data:
	$(PYTHON) data/generate_data.py

outputs: data
	$(PYTHON) scripts/export_outputs.py

app: data
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest
