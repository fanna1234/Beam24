.PHONY: check lint ci smoke configure gpu-build

check:
	python3 -m compileall -q src scripts tests artifact/scripts
	python3 scripts/check_baseline_manifest.py
	python3 scripts/check_evaluation_matrix.py
	python3 scripts/check_reference_anchors.py
	python3 scripts/check_claim_framing.py
	python3 -m unittest discover -s tests -v
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
