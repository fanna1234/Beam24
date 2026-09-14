.PHONY: check lint ci smoke configure gpu-build

PYTHON ?= $(if $(BEAM24_PYTHON),$(BEAM24_PYTHON),$(if $(PYTHON_BIN),$(PYTHON_BIN),python3))

check:
	"$(PYTHON)" -m compileall -q src scripts tests artifact/scripts
	"$(PYTHON)" scripts/check_baseline_manifest.py
	"$(PYTHON)" scripts/check_evaluation_matrix.py
	"$(PYTHON)" scripts/check_reference_anchors.py
	"$(PYTHON)" scripts/check_claim_framing.py
	"$(PYTHON)" -m unittest discover -s tests -v
	@if command -v shellcheck >/dev/null 2>&1; then \
		$(MAKE) lint; \
	else \
		echo "shellcheck unavailable: lint skipped (required by make ci)"; \
	fi

lint:
	shellcheck artifact/*.sh artifact/scripts/*.sh

ci:
	@command -v shellcheck >/dev/null 2>&1 || { echo "shellcheck required for make ci" >&2; exit 3; }
	$(MAKE) check

smoke:
	./artifact/reproduce.sh smoke

configure:
	cmake --preset cpu-check

gpu-build:
	cmake --preset sm120
	cmake --build --preset sm120
