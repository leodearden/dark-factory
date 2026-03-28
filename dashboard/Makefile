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
# Verified by downloading the actual release artifacts from GitHub and computing sha256sum.
CHECKSUM_linux-x64   := 4ab84f2b496c402d3ec4fd25e0e5559fe1184d886dadae8fb4438344ec044c22
CHECKSUM_linux-arm64 := ad627e77b496cccada4a6e26eafff698ef0829081e575a4baf3af8524bb00747
CHECKSUM_macos-x64   := 98e34c6abd00a75a74ea2d20acf9e284241d13023133076d220c6f3ca419d920
CHECKSUM_macos-arm64 := 2ce66b7c8101ef1245a07d1e7abb4beb35bf512fd3beecba1cdfb327580d1252

EXPECTED_CHECKSUM := $(CHECKSUM_$(PLATFORM))

.DELETE_ON_ERROR:

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
