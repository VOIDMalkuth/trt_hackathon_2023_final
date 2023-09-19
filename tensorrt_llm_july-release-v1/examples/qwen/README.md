# Qwen-7B

This document shows how to build and run a Qwen-7B model in TensorRT-LLM on a single GPU.

## Run Original Qwen-7B

Run `git clone https://huggingface.co/Qwen/Qwen-7B` in current folder, note that git lfs is used for model files.

Make sure requirements are installed by running `pip install -r requirements.txt` in `Qwen-7B` folder.

Then run script with `python run_original.py`.

## Run trt version of Qwen-7B

### Build TRT engine

```bash
# under tensorrt_llm_july-release-v1/examples/qwen
# make sure Qwen-7B folder exists with weight
python build.py
```

### Run Qwen-7B with TRT engine

```bash
# under tensorrt_llm_july-release-v1/examples/qwen
# make sure Qwen-7B folder exists with weight and qwen_trt_engine folder exists with engine

# with no --input_text arguement, the script will use default input sentence as in run_original.py
python run.py --max_output_len=128
```

## Run summary test on CNN Dailymail Dataset

```bash
# under tensorrt_llm_july-release-v1/examples/qwen
# make sure Qwen-7B folder exists with weight and qwen_trt_engine folder exists with engine

# don't run both tests in a single run as the gpu memory is limited

python summarize.py --test_hf
python summarize.py --test_trt_llm
```

If network connection is limited, you may download required dataset and model and then run with offline mode.
