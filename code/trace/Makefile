PYTHON ?= python3
TEMPLATE ?= ../www2027-extended-results-template.csv
MEASURED ?= ../www2027-extended-results-measured.csv
RAW_DIR ?= ../experiment-raw
RUNNER ?=

.PHONY: template smoke analyze require-complete compile icde-figures icde-figure-workbooks

template:
	$(PYTHON) build_template.py $(TEMPLATE)

compile:
	$(PYTHON) -m py_compile *.py

icde-figures:
	MPLBACKEND=Agg XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
		$(PYTHON) generate_icde_simulation_figures.py

icde-figure-workbooks:
	$(PYTHON) build_icde_figure_workbooks.py

smoke:
	rm -f /tmp/edgae-smoke.csv
	rm -rf /tmp/edgae-smoke-raw
	$(PYTHON) run_extended_eval.py --simulate \
		--blocks cache-stability --datasets web-google \
		--controls K=1 --seeds 1 \
		--output /tmp/edgae-smoke.csv --raw-dir /tmp/edgae-smoke-raw
	$(PYTHON) analyze_extended_results.py /tmp/edgae-smoke.csv \
		--include-invalid --bootstrap-samples 100

analyze:
	$(PYTHON) analyze_extended_results.py $(MEASURED)

require-complete:
	$(PYTHON) analyze_extended_results.py $(MEASURED) --require-complete

run-real:
	@test -n "$(RUNNER)" || { echo "RUNNER=/path/to/runner.py is required"; exit 2; }
	$(PYTHON) run_extended_eval.py --runner-command "$(RUNNER)" \
		--template $(TEMPLATE) --output $(MEASURED) --raw-dir $(RAW_DIR)
