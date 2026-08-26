SHELL := bash

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

Q ?= @
ifeq ($(V),1)
Q :=
endif

define msg
	@printf "  %-8s  %s\n" "$(1)" "$(2)"
endef

comma := ,

# KuiperBench is checked separately from Kuiper. The default is fixed so local
# and CI builds use the same verified library and compiler suite. An explicit
# KUIPER_HOME is treated as user-managed and is never overwritten.
ifeq ($(origin KUIPER_HOME),undefined)
KUIPER_HOME := $(CURDIR)/.kuiper
KUIPER_HOME_MANAGED := 1
else
KUIPER_HOME_MANAGED := 0
endif
KUIPER_NIGHTLY ?= 2026-08-25
ifeq ($(KUIPER_HOME_MANAGED),1)
# Encoding the pin in the target makes a nightly change invalidate the
# installed package. The install recipe also removes outputs checked against
# the previous package before allowing dependent work to start.
KUIPER_MARKER := $(KUIPER_HOME)/.kbench-nightly-$(KUIPER_NIGHTLY)
else
KUIPER_MARKER := $(KUIPER_HOME)/.packaged
endif
FSTAR_EXE := $(KUIPER_HOME)/inst/bin/fstar.exe
KRML_EXE := $(KUIPER_HOME)/inst/bin/krml
PLUGIN_SOURCE := $(KUIPER_HOME)/extraction/dune/_build/default/kuiper_extr.cmxs

# Generated CUDA is checked in, so its formatter is part of the reproducible
# generated-code path. Download one exact clang-format release instead of using
# whichever version happens to be installed on the host.
TOOLS_DIR := $(CURDIR)/.tools
CLANG_FORMAT_VERSION := 19.1.7
CLANG_FORMAT := $(TOOLS_DIR)/clang-format-$(CLANG_FORMAT_VERSION)/bin/clang-format
CLANG_FORMAT_FLAGS := --Werror --fail-on-incomplete-format \
	--style=file:$(CURDIR)/.clang-format
# Karamel starts generated files with multiple blank lines, which clang-format
# intentionally preserves. Canonical output retains exactly one.
NORMALIZE_LEADING_BLANKS := awk 'BEGIN { print "" } NF || seen { seen = 1; print }'
# F* interprets dots in directory components of --load_cmxs as module-name
# separators. Stage a dot-free local symlink so extraction works with the
# default .kuiper package directory.
PLUGIN = $(CURDIR)/$(OUTDIR)/kuiper_extr

export KUIPER_HOME
export FSTAR_EXE
export KRML_EXE

OUTDIR := obj
CACHEDIR := $(OUTDIR)
ROOTS := $(shell find src -type f \( -name '*.fst' -o -name '*.fsti' \) | sort)
CHECKED := $(foreach f,$(ROOTS),$(OUTDIR)/$(notdir $(f)).checked)

FSTAR_FLAGS :=
FSTAR_FLAGS += --include $(KUIPER_HOME)/src
FSTAR_FLAGS += --include $(KUIPER_HOME)/obj
FSTAR_FLAGS += --include $(CURDIR)/src
FSTAR_FLAGS += --include $(CURDIR)/$(CACHEDIR)
FSTAR_FLAGS += --cache_dir $(CACHEDIR)
FSTAR_FLAGS += --odir $(OUTDIR)
FSTAR_FLAGS += --warn_error -274
FSTAR_FLAGS += --warn_error -291
FSTAR_FLAGS += --warn_error -249-321
FSTAR_FLAGS += --warn_error @242@250
FSTAR_FLAGS += --warn_error -288
FSTAR_FLAGS += --warn_error -271
FSTAR_FLAGS += --z3version 4.13.3
FSTAR_FLAGS += --ext kuiper
FSTAR_FLAGS += --ext __unrefine
FSTAR_FLAGS += --ext no_krml_private
FSTAR_FLAGS += --ext context_pruning_no_ambients
FSTAR_FLAGS += --ext freshen
ifneq ($(filter-out 0 1,$(strip $(ADMIT))),)
$(error ADMIT must be unset, 0, or 1)
endif
ifeq ($(strip $(ADMIT)),1)
FSTAR_FLAGS += --admit_smt_queries true
endif
FSTAR_FLAGS += $(O)

FSTAR := $(FSTAR_EXE) $(if $(V),,--silent) $(FSTAR_FLAGS)

KRML_FLAGS :=
KRML_FLAGS += -add-early-include '<kuiper.h>'
KRML_FLAGS += -add-early-include '<kbench.h>'
KRML_FLAGS += -fc++-compat -fcast-allocations
KRML_FLAGS += -skip-compilation -skip-makefiles
KRML_FLAGS += -faggressive-inlining -fauto-for-loops -fnoshort-enums
KRML_FLAGS += -cuda -dbacktrace
KRML_FLAGS += $(if $(V),-verbose,-silent)
KRML_FLAGS += -drop Prims -minimal -header /dev/null
KRML_FLAGS += -warn-error @6 -warn-error -2@4-10@18
KRML_FLAGS += $(KO)
KRML := $(KRML_EXE) $(KRML_FLAGS)

PACKAGE_EXTRACT := \
	$(KUIPER_HOME)/src/klas/Klas.GEMM.Naive1.fst \
	$(KUIPER_HOME)/src/klas/Klas.GEMM.Naive2.fst \
	$(KUIPER_HOME)/src/klas/Klas.GEMM.Naive3.fst \
	$(KUIPER_HOME)/src/klas/Klas.RowScale.fst \
	$(KUIPER_HOME)/src/klas/Klas.RowSoftmax.fst

EXTRACT := $(shell find src/kernelbench -mindepth 3 -maxdepth 3 -name '*.fst' | sort)
EXTRACT += $(wildcard src/examples/*.fst)
EXTRACT += $(wildcard src/klas/*.fst)
EXTRACT += $(PACKAGE_EXTRACT)
EXTRACT_MODULES := $(subst .,_,$(basename $(notdir $(EXTRACT))))
EXTRACT_CU := $(addprefix $(OUTDIR)/,$(addsuffix .cu,$(EXTRACT_MODULES)))
EXTRACT_H := $(addprefix $(OUTDIR)/,$(addsuffix .h,$(EXTRACT_MODULES)))

.PHONY: all verify extract-all install-kuiper install-clang-format prepare clean dist
all: verify extract-all

verify: $(CHECKED)

extract-all: $(EXTRACT_CU) $(EXTRACT_H)

install-kuiper: $(KUIPER_MARKER)
install-clang-format: $(CLANG_FORMAT)
prepare: $(KUIPER_MARKER) $(CLANG_FORMAT)

ifeq ($(KUIPER_HOME_MANAGED),1)
$(KUIPER_MARKER):
	$(call msg,INSTALL,Kuiper nightly $(KUIPER_NIGHTLY))
	$(Q)./scripts/install-kuiper.sh --nightly --version $(KUIPER_NIGHTLY) --dest $(KUIPER_HOME) --no-link
	$(Q)test -f $(KUIPER_HOME)/.packaged
	$(Q)test -x $(FSTAR_EXE)
	$(Q)test -x $(KRML_EXE)
	$(Q)test -f $(PLUGIN_SOURCE)
	$(Q)rm -rf $(OUTDIR)
	$(Q)rm -f .depend
	$(Q)touch $@
else
$(KUIPER_MARKER):
	$(error KUIPER_HOME does not contain a packaged Kuiper installation: $(KUIPER_HOME))
endif

$(CLANG_FORMAT): scripts/install-clang-format.sh
	$(call msg,INSTALL,clang-format $(CLANG_FORMAT_VERSION))
	$(Q)CLANG_FORMAT_VERSION=$(CLANG_FORMAT_VERSION) \
		./scripts/install-clang-format.sh $(TOOLS_DIR)/clang-format-$(CLANG_FORMAT_VERSION)

# These files appear as ordinary prerequisites later in the graph. On a fresh
# checkout their rule first materializes the package, then validates its shape.
$(FSTAR_EXE) $(KRML_EXE) $(PLUGIN_SOURCE) $(PACKAGE_EXTRACT): | $(KUIPER_MARKER)
	$(Q)test -e $@

ifeq ($(filter clean echo-fstar echo-krml install-kuiper prepare kb-venv,$(MAKECMDGOALS)),)
-include .depend
endif

.depend: $(ROOTS) $(KUIPER_MARKER)
	$(call msg,DEPEND,$@)
	$(Q)mkdir -p $(OUTDIR)
	$(Q)$(FSTAR) --codegen krml \
		--already_cached 'FStar,LowStar,Prims,Pulse,PulseCore' \
		--dep full $(ROOTS) -o $@.tmp
	$(Q)mv $@.tmp $@

$(OUTDIR)/%.checked: | $(KUIPER_MARKER)
	$(call msg,CHECK,$<)
	$(Q)mkdir -p $(OUTDIR)
	$(Q)$(FSTAR) --already_cached '*' -c $< -o $@
	$(Q)touch -c $@

$(OUTDIR)/%.krml: MOD = $(subst _,.,$(basename $(notdir $@)))
# Package modules are not part of the local dependency graph, so provide their
# extraction source explicitly. Their dependencies resolve from package cache.
$(OUTDIR)/Klas_GEMM_Naive1.krml: $(KUIPER_HOME)/src/klas/Klas.GEMM.Naive1.fst
$(OUTDIR)/Klas_GEMM_Naive2.krml: $(KUIPER_HOME)/src/klas/Klas.GEMM.Naive2.fst
$(OUTDIR)/Klas_GEMM_Naive3.krml: $(KUIPER_HOME)/src/klas/Klas.GEMM.Naive3.fst
$(OUTDIR)/Klas_RowScale.krml: $(KUIPER_HOME)/src/klas/Klas.RowScale.fst
$(OUTDIR)/Klas_RowSoftmax.krml: $(KUIPER_HOME)/src/klas/Klas.RowSoftmax.fst

$(OUTDIR)/kuiper_extr.cmxs: $(PLUGIN_SOURCE) | $(KUIPER_MARKER)
	$(Q)mkdir -p $(OUTDIR)
	$(Q)ln -sf $(PLUGIN_SOURCE) $@

$(OUTDIR)/%.krml: | $(KUIPER_MARKER) $(OUTDIR)/kuiper_extr.cmxs
	$(call msg,EXTRACT,$(MOD))
	$(Q)$(FSTAR) --codegen krml --load_cmxs $(PLUGIN) \
		--extract "-*,+$(MOD),+Kuiper$(if $(findstring Kuiper.KB.SDPA,$(MOD)),$(comma)+Klas)" \
		-o $@ $<

$(OUTDIR)/pre/%.cu $(OUTDIR)/pre/%.h &: MOD = $(subst _,.,$(basename $(notdir $<)))
$(OUTDIR)/pre/%.cu $(OUTDIR)/pre/%.h &: $(OUTDIR)/%.krml $(KRML_EXE)
	$(call msg,KRML,$(MOD))
	$(Q)mkdir -p $(OUTDIR)/pre
	$(Q)$(KRML) -bundle "$(MOD)=*" -tmpdir $(OUTDIR)/pre/ $<

$(OUTDIR)/%.cu: $(OUTDIR)/pre/%.cu $(KUIPER_HOME)/scripts/fixup.sed $(CLANG_FORMAT) .clang-format
	$(call msg,FIXUP,$@)
	$(Q)sed -f $(KUIPER_HOME)/scripts/fixup.sed $< | \
		$(CLANG_FORMAT) $(CLANG_FORMAT_FLAGS) --assume-filename=$@ | \
		$(NORMALIZE_LEADING_BLANKS) > $@

$(OUTDIR)/%.h: $(OUTDIR)/pre/%.h $(KUIPER_HOME)/scripts/fixup.sed $(CLANG_FORMAT) .clang-format
	$(call msg,FIXUP,$@)
	$(Q)sed -f $(KUIPER_HOME)/scripts/fixup.sed $< | \
		$(CLANG_FORMAT) $(CLANG_FORMAT_FLAGS) --assume-filename=$@ | \
		$(NORMALIZE_LEADING_BLANKS) > $@

dist: extract-all
	$(Q)./scripts/update-dist.sh

clean:
	$(Q)rm -rf $(OUTDIR) .depend

.PHONY: echo-fstar echo-krml
echo-fstar: $(KUIPER_MARKER)
	@echo $(FSTAR)

echo-krml: $(KUIPER_MARKER)
	@echo $(KRML)

# KernelBench integration tests. Verification/extraction is deliberately a
# prerequisite so the Python extensions consume current generated CUDA.
KUIPER_VENV := $(CURDIR)/.venv
KUIPER_VENV_PY := $(KUIPER_VENV)/bin/python
KUIPER_VENV_STAMP := $(KUIPER_VENV)/.deps-installed
KUIPER_VENV_BASE_PY ?= python3

$(KUIPER_VENV_PY):
	$(call msg,VENV,$(KUIPER_VENV))
	$(Q)$(KUIPER_VENV_BASE_PY) -m venv $(KUIPER_VENV)
	$(Q)$(KUIPER_VENV_PY) -m pip install --quiet --upgrade pip wheel

$(KUIPER_VENV_STAMP): $(KUIPER_VENV_PY) src/kernelbench/requirements.txt
	$(call msg,PIP,src/kernelbench/requirements.txt)
	$(Q)$(KUIPER_VENV_PY) -m pip install --quiet -r src/kernelbench/requirements.txt
	$(Q)touch $@

.PHONY: kb-venv test-kb-all
kb-venv: $(KUIPER_VENV_STAMP)

test-kb-%: extract-all $(KUIPER_VENV_STAMP)
	@test -f src/kernelbench/level1/challenge$*/run.sh || \
		{ echo "challenge $* has no run.sh"; exit 1; }
	@KUIPER_PYTHON=$(KUIPER_VENV_PY) bash src/kernelbench/level1/challenge$*/run.sh

test-kb-all: extract-all $(KUIPER_VENV_STAMP)
	@KUIPER_PYTHON=$(KUIPER_VENV_PY) $(KUIPER_VENV_PY) src/kernelbench/run_test.py

.PHONY: lint list-admits
lint:
	@! git grep -n '[[:blank:]]$$' -- 'src/*.fst' 'src/*.fsti' 'src/**/*.fst' 'src/**/*.fsti'

list-admits:
	@git ls-files -z -- 'src/*.fst' 'src/*.fsti' 'src/**/*.fst' 'src/**/*.fsti' | \
		xargs -0 python3 scripts/list-admits.py
