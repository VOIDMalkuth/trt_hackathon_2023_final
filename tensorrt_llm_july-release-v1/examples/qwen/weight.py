import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def load_from_hf_qwen(tensorrt_llm_qwen,
                       hf_qwen,
                       rank=0,
                       tensor_parallel=1,
                       dtype="float32",
                       multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_qwen.named_parameters())
    print(model_params.keys())

    # qkv is 'transformer.h.{l}.attn.c_attn.weight/bias'
    # qkv_proj is 'transformer.h.{l}.attn.c_proj.weight'
    for l in range(hf_qwen.config.num_hidden_layers):
        weight_name = f'transformer.h.{l}.attn.c_attn.weight'
        bias_name = f'transformer.h.{l}.attn.c_attn.bias'
        qkv_weight = model_params[weight_name]
        qkv_bias = model_params[bias_name]
        if multi_query_mode:
            qkv_weight = list(torch.chunk(qkv_weight, 3))
            qkv_bias = list(torch.chunk(qkv_bias, 3))

        model_params[weight_name] = qkv_weight
        model_params[bias_name] = qkv_bias

    torch_dtype = str_dtype_to_torch(dtype)
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        
        if 'transformer.wte.weight' in k:
            tensorrt_llm_qwen.wte.weight.value = v
        elif 'transformer.ln_f.weight' in k:
            tensorrt_llm_qwen.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            tensorrt_llm_qwen.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                print("Unknown weight name: " + str(k))
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_qwen.num_layers:
                print("Weight idx exceeded range, name: " + str(k))
                continue
            if 'ln_1.weight' in k:
                tensorrt_llm_qwen.layers[idx].ln_1.weight.value = v
            elif 'ln_2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].ln_2.weight
                dst.value = v
            elif 'attn.c_attn.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attn.qkv.weight
                if multi_query_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], tensor_parallel, rank)
                    wk = split(v[1], tensor_parallel, rank)
                    wv = split(v[2], tensor_parallel, rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
                                              model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[idx].attn.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_attn.bias' in k:
                dst = tensorrt_llm_qwen.layers[idx].attn.qkv.bias
                if multi_query_mode:
                    assert False
                else:
                    q_emb = v.shape[0] // 3
                    v = v.reshape(3, q_emb)
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // tensor_parallel))
                
                # bias don't care about use_weight_only:
                dst.value = np.ascontiguousarray(split_v)
            elif 'attn.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].attn.dense.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[idx].attn.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)

                # set bias to 0
                dst = tensorrt_llm_qwen.layers[idx].attn.dense.bias
                v = np.zeros(v.shape[:-1], dtype=dtype)
                split_v = split(v, tensor_parallel, rank, dim=1)
                # bias don't care about use_weight_only:
                dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w1.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.w1.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[idx].mlp.w1.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.c_proj.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.c_proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[idx].mlp.c_proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.w2.weight' in k:
                dst = tensorrt_llm_qwen.layers[idx].mlp.w2.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_qwen.layers[idx].mlp.w2.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            else:
                print("Unknown weight " + str(k))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return
