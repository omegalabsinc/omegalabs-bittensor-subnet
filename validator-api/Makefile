PROJECT_NAME := validator-api
MEMORY_LIMIT := 14g

default: build sh

build:
	docker build -f Dockerfile -t $(PROJECT_NAME):latest .

sh:
	docker run --memory=$(MEMORY_LIMIT) --rm \
		--network="host" \
		--gpus all \
		--detach \
		-v $(shell pwd):/app \
		-v $(shell pwd)/cache:/root/.cache/ \
		-it \
		--name $(PROJECT_NAME)-sh \
		--entrypoint bash \
		$(PROJECT_NAME):latest
	docker attach $(PROJECT_NAME)-sh

run:
	docker run --memory=$(MEMORY_LIMIT) --rm \
		--network="host" \
		--gpus all \
		--detach \
		-v $(shell pwd):/app \
		-v $(shell pwd)/cache:/root/.cache/ \
		-it \
		--name $(PROJECT_NAME) \
		$(PROJECT_NAME):latest
	docker attach $(PROJECT_NAME)
