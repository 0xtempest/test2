.PHONY: env

make_env:
	python3 -m venv vllm_inference_env

make install:
	pip3 install -r requirements.txt

