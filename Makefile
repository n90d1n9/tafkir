.PHONY: error-codes product-boundaries ci

error-codes:
	./scripts/generate-error-codes.sh

product-boundaries:
	./scripts/check-product-boundaries.sh

ci: product-boundaries
	CI=true ./gradlew --no-daemon -x test :spi:tafkir-spi:build
