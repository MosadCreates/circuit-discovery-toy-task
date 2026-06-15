.PHONY: train patch visualize figures report test serve-docs clean setup setup-linux

train:
	python src/training/train.py --config configs/training/default.yaml

patch:
	python src/patching/residual_stream_patch.py --config configs/patching/default.yaml
	python src/patching/head_patch.py --config configs/patching/default.yaml
	python src/patching/mlp_patch.py --config configs/patching/default.yaml
	python src/patching/path_patch.py --config configs/patching/default.yaml

visualize:
	python src/data/visualize_dataset.py --config configs/training/default.yaml
	python src/analysis/attention_patterns.py
	python src/analysis/head_roles.py
	python src/analysis/attention_variation.py
	python src/analysis/fourier.py
	python src/analysis/embedding_analysis.py
	python src/analysis/neuron_analysis.py
	python src/analysis/logit_fourier.py
	python src/analysis/logit_lens.py
	python src/analysis/direct_logit_attribution.py
	python src/analysis/weight_analysis.py
	python src/analysis/grokking_dynamics.py
	python src/analysis/minimal_circuit.py

# Copy all result figures to docs/figures/ with Figure_N naming convention.
# Each script saves to its own output directory; this target unifies them.
figures:
	@mkdir -p docs/figures
ifeq ($(OS),Windows_NT)
	powershell -Command "\
		$$src = 'results'; \
		$$dst = 'docs/figures'; \
		$$( \
			'grokking_dynamics.png',                       'Figure_1_grokking_dynamics.png'; \
			'fourier/embedding_fourier.png',               'Figure_2_embedding_fourier.png'; \
			'fourier/neuron_2d_fourier_top20.png',         'Figure_3_neuron_2d_fourier.png'; \
			'attention/attention_summary.png',             'Figure_4_attention_summary.png'; \
			'patching/residual_stream_patch.png',          'Figure_5_residual_stream_patch.png'; \
			'patching/head_patch.png',                     'Figure_6_head_patch.png'; \
			'patching/neuron_patch.png',                   'Figure_7_neuron_patch.png'; \
			'direct_logit_attribution.png',                'Figure_8_direct_logit_attribution.png'; \
			'circuit_accuracy_vs_components.png',          'Figure_9_circuit_accuracy.png'; \
		) | ForEach-Object { \
			$$srcFile = Join-Path $$src $$_.Item1; \
			$$dstFile = Join-Path $$dst $$_.Item2; \
			if (Test-Path $$srcFile) { Copy-Item $$srcFile $$dstFile -Force; Write-Output \"  Copied $$srcFile -> $$dstFile\"; } \
			else { Write-Output \"  MISSING: $$srcFile\"; } \
		}"
else
	@for pair in \
		"results/grokking_dynamics.png:docs/figures/Figure_1_grokking_dynamics.png" \
		"results/fourier/embedding_fourier.png:docs/figures/Figure_2_embedding_fourier.png" \
		"results/fourier/neuron_2d_fourier_top20.png:docs/figures/Figure_3_neuron_2d_fourier.png" \
		"results/attention/attention_summary.png:docs/figures/Figure_4_attention_summary.png" \
		"results/patching/residual_stream_patch.png:docs/figures/Figure_5_residual_stream_patch.png" \
		"results/patching/head_patch.png:docs/figures/Figure_6_head_patch.png" \
		"results/patching/neuron_patch.png:docs/figures/Figure_7_neuron_patch.png" \
		"results/direct_logit_attribution.png:docs/figures/Figure_8_direct_logit_attribution.png" \
		"results/circuit_accuracy_vs_components.png:docs/figures/Figure_9_circuit_accuracy.png" \
	; do \
		src=$${pair%:*}; dst=$${pair#*:}; \
		if [ -f "$$src" ]; then cp "$$src" "$$dst"; echo "  Copied $$src -> $$dst"; \
		else echo "  MISSING: $$src"; fi; \
	done

report: figures
	mkdocs build --clean -d site

test:
	pytest tests/ -v --tb=short

serve-docs: figures
	mkdocs serve

clean:
ifeq ($(OS),Windows_NT)
	@if exist ".venv" rmdir /S /Q .venv
	@if exist "results\checkpoints" rmdir /S /Q results\checkpoints\* 2>nul || echo ok
	@if exist "results\figures" rmdir /S /Q results\figures\* 2>nul || echo ok
	@if exist "results\fourier" rmdir /S /Q results\fourier 2>nul || echo ok
	@if exist "results\attention" rmdir /S /Q results\attention 2>nul || echo ok
	@if exist "results\patching" rmdir /S /Q results\patching 2>nul || echo ok
	@powershell -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
	rm -rf .venv
	rm -rf results/checkpoints/*
	rm -rf results/fourier/*
	rm -rf results/attention/*
	rm -rf results/patching/*
	rm -rf results/figures/*
	rm -rf site
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
endif

setup:
	powershell -ExecutionPolicy Bypass -File setup.ps1

setup-cpu:
	powershell -ExecutionPolicy Bypass -File setup.ps1 -CPU

setup-linux:
	./setup.sh
