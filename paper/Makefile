TECTONIC ?= tectonic
OUT ?= build

.PHONY: all clean

all:
	mkdir -p $(OUT)
	$(TECTONIC) -X compile main.tex --outdir $(OUT) --keep-logs

clean:
	rm -rf $(OUT)
