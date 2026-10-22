.PHONY: test bench serve web web-build

test:
	python3 -m pytest tests -q

bench:
	python3 -m nano_infer.benchmark

serve:
	python3 -m nano_infer.server

web:
	cd web && npm run dev -- -p 43123

web-build:
	cd web && npm run build
