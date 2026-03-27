TAILWIND_VERSION := 4.2.2
TAILWIND_BIN_DIR := bin
TAILWIND_BIN := $(TAILWIND_BIN_DIR)/tailwindcss
TAILWIND_URL := https://github.com/tailwindlabs/tailwindcss/releases/download/v$(TAILWIND_VERSION)/tailwindcss-linux-x64

INPUT_CSS  := src/dashboard/static/input.css
OUTPUT_CSS := src/dashboard/static/tailwind.css

.PHONY: css css-watch clean

$(TAILWIND_BIN):
	mkdir -p $(TAILWIND_BIN_DIR)
	curl -fsSL $(TAILWIND_URL) -o $(TAILWIND_BIN) || (rm -f $(TAILWIND_BIN); exit 1)
	chmod +x $(TAILWIND_BIN)

css: $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --minify

css-watch: $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --watch

clean:
	rm -rf $(TAILWIND_BIN_DIR) $(OUTPUT_CSS)
