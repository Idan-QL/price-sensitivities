# run commands:
# Run all targets with image name "my-project":
# $: make TAG=staging PROJ=my-project
# $: make all TAG=staging PROJ=my-project
# Build and push to ECR only (with default image name):
# $: make push_eu TAG=production
# Only build image for US (with default image name):
# $: make build_us TAG=production

SHELL=/bin/bash

# Define the possible values for the env and dc arguments
#ALLOWED_TAG_VALUES := staging production

# Validate the env and dc arguments
#ifeq (,$(filter $(TAG),$(ALLOWED_TAG_VALUES)))
#  $(error Invalid TAG argument. Allowed values: $(ALLOWED_TAG_VALUES))
#endif

AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)
PW_US := $(shell aws ecr get-login-password --region us-east-1)
PW_EU := $(shell aws ecr get-login-password --region eu-central-1)

# Define the default value for the project name
PROJ ?= elasticity

all: build_us push_us build_eu push_eu

build_us:
	@echo
	@echo "Building docker image $(PROJ):$(TAG) for US data center"
	DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --ssh default -t $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$(PROJ):$(TAG) .

.PHONY: push_us
push_us: build_us
	@echo
	@echo "Pushing image $(PROJ):$(TAG) to AWS ECR (US)"
	@echo "$(PW_US)" | docker login -u AWS $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com --password-stdin
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.us-east-1.amazonaws.com/$(PROJ):$(TAG)

build_eu:
	@echo
	@echo "Building docker image $(PROJ):$(TAG) for EU data center"
	DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --ssh default -t $(AWS_ACCOUNT_ID).dkr.ecr.eu-central-1.amazonaws.com/$(PROJ):$(TAG) .

.PHONY: push_eu
push_eu: build_eu
	@echo
	@echo "Pushing image $(PROJ):$(TAG) to AWS ECR (EU)"
	@echo "$(PW_EU)" | docker login -u AWS $(AWS_ACCOUNT_ID).dkr.ecr.eu-central-1.amazonaws.com --password-stdin
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.eu-central-1.amazonaws.com/$(PROJ):$(TAG)

.SILENT: all
