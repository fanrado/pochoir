# PLAN_for_production.md §3 — single entry point for the test-full-3d-pixel
# workflow. Runs pytest and the pipeline, capturing artifacts under a fresh
# test/review/<date>_<slug>/ folder with a NOTES.md stub.
#
# Iteration defaults use the reduced z-extents from the plan (300 instead of
# 500). Release validation overrides them on the command line:
#
#   make test-pixel SLUG=release-validation \
#        POCHOIR_DRIFT_SHAPE=44,44,500 POCHOIR_WEIGHT_SHAPE=220,220,500
#
# Assumes the venv at pochoir/env/ is already activated (per PLAN §"Python
# environment").

POCHOIR_DRIFT_SHAPE  ?= 44,44,300
POCHOIR_WEIGHT_SHAPE ?= 220,220,300
SLUG                 ?= iteration
DATE                 := $(shell date +%Y-%m-%d)
REVIEW_DIR           := test/review/$(DATE)_$(SLUG)

.PHONY: test-pixel pytest pipeline

test-pixel: pytest pipeline
	@echo "Artifacts: $(REVIEW_DIR)"

pytest:
	@mkdir -p $(REVIEW_DIR)
	@echo "=== pytest -> $(REVIEW_DIR)/pytest.log ==="
	cd test && python -m pytest -v 2>&1 | tee ../$(REVIEW_DIR)/pytest.log

pipeline:
	@mkdir -p $(REVIEW_DIR)/outputs
	@echo "=== test-full-3d-pixel.sh -> $(REVIEW_DIR)/pipeline.log ==="
	@echo "    POCHOIR_DRIFT_SHAPE=$(POCHOIR_DRIFT_SHAPE)"
	@echo "    POCHOIR_WEIGHT_SHAPE=$(POCHOIR_WEIGHT_SHAPE)"
	cd test && POCHOIR_DRIFT_SHAPE=$(POCHOIR_DRIFT_SHAPE) \
	           POCHOIR_WEIGHT_SHAPE=$(POCHOIR_WEIGHT_SHAPE) \
	           bash test-full-3d-pixel.sh 2>&1 | tee ../$(REVIEW_DIR)/pipeline.log
	@if [ ! -f $(REVIEW_DIR)/NOTES.md ]; then \
	  printf '%s\n' \
	    "# $(SLUG) — $(DATE)" "" \
	    "## What changed" "- " "" \
	    "## Why" "- " "" \
	    "## Tests added / updated" "- " "" \
	    "## How to reproduce" \
	    "- \`make test-pixel SLUG=$(SLUG) POCHOIR_DRIFT_SHAPE=$(POCHOIR_DRIFT_SHAPE) POCHOIR_WEIGHT_SHAPE=$(POCHOIR_WEIGHT_SHAPE)\`" "" \
	    "## Result" \
	    "- pytest: see pytest.log" \
	    "- pipeline: see pipeline.log" "" \
	    "## Follow-ups" "- " \
	    > $(REVIEW_DIR)/NOTES.md ; \
	fi
