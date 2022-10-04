.PHONY: test
test:
	python -m pytest -s -vv tests

.PHONY: bump2version
bump2version:
	bump2version $(PART)
