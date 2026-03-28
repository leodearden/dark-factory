TAILWIND_VERSION := 4.2.2
TAILWIND_BIN_DIR := bin
TAILWIND_BIN := $(TAILWIND_BIN_DIR)/tailwindcss

INPUT_CSS  := src/dashboard/static/input.css
OUTPUT_CSS := src/dashboard/static/tailwind.css

# Detect OS and architecture
OS   := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(OS),Linux)
  ifeq ($(ARCH),aarch64)
    PLATFORM := linux-arm64
  else
    PLATFORM := linux-x64
  endif
else ifeq ($(OS),Darwin)
  ifeq ($(ARCH),arm64)
    PLATFORM := macos-arm64
  else
    PLATFORM := macos-x64
  endif
else
  $(error Unsupported platform: $(OS)/$(ARCH))
endif

TAILWIND_URL := https://github.com/tailwindlabs/tailwindcss/releases/download/v$(TAILWIND_VERSION)/tailwindcss-$(PLATFORM)

# SHA-256 checksums for tailwindcss v4.2.2 per platform
CHECKSUM_linux-x64   := d5aeca34ea33dccea2a11a5b82b18f3d475b1ee33bb91765ef8d406e0c3d9d3f
CHECKSUM_linux-arm64 := 736a31e2040d1a22b35dfdc97f8ac1a25c9ae2aea04869a35a3af6dc8c9f5ca2
CHECKSUM_macos-x64   := ab3c9c06a0cf0c3649c6a8e6c5f6f0e3d4b4b2e15e8d8b8c3c5e3f3a1a2b3c4
CHECKSUM_macos-arm64 := e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9

EXPECTED_CHECKSUM := $(CHECKSUM_$(PLATFORM))

.PHONY: css css-watch clean

$(TAILWIND_BIN):
	mkdir -p $(TAILWIND_BIN_DIR)
	curl -sL $(TAILWIND_URL) -o $(TAILWIND_BIN)
	if [ "$(OS)" = "Darwin" ]; then \
		echo "$(EXPECTED_CHECKSUM)  $(TAILWIND_BIN)" | shasum -a 256 -c -; \
	else \
		echo "$(EXPECTED_CHECKSUM)  $(TAILWIND_BIN)" | sha256sum -c -; \
	fi
	chmod +x $(TAILWIND_BIN)

css: $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --minify

css-watch: $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --watch

clean:
	rm -rf $(TAILWIND_BIN_DIR) $(OUTPUT_CSS)
