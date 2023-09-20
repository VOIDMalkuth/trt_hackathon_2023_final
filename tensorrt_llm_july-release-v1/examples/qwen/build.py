import argparse
import json
import os
import time
from pathlib import Path

import tensorrt as trt
import torch
import torch.multiprocessing as mp

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.models import weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.quantization import QuantMode

from transformers import AutoModelForCausalLM

from weight import load_from_hf_qwen  # isort:skip

MODEL_NAME = "Qwen"

import onnx
import tensorrt as trt
from onnx import TensorProto, helper


def trt_dtype_to_onnx(dtype):
    if dtype == trt.float16:
        return TensorProto.DataType.FLOAT16
    elif dtype == trt.float32:
        return TensorProto.DataType.FLOAT
    elif dtype == trt.int32:
        return TensorProto.DataType.INT32
    else:
        raise TypeError("%s is not supported" % dtype)


def to_onnx(network, path):
    inputs = []
    for i in range(network.num_inputs):
        network_input = network.get_input(i)
        inputs.append(
            helper.make_tensor_value_info(
                network_input.name, trt_dtype_to_onnx(network_input.dtype),
                list(network_input.shape)))

    outputs = []
    for i in range(network.num_outputs):
        network_output = network.get_output(i)
        outputs.append(
            helper.make_tensor_value_info(
                network_output.name, trt_dtype_to_onnx(network_output.dtype),
                list(network_output.shape)))

    nodes = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_inputs = []
        for j in range(layer.num_inputs):
            ipt = layer.get_input(j)
            if ipt is not None:
                layer_inputs.append(layer.get_input(j).name)
        layer_outputs = [
            layer.get_output(j).name for j in range(layer.num_outputs)
        ]
        nodes.append(
            helper.make_node(str(layer.type),
                             name=layer.name,
                             inputs=layer_inputs,
                             outputs=layer_outputs,
                             domain="com.nvidia"))

    onnx_model = helper.make_model(helper.make_graph(nodes,
                                                     'attention',
                                                     inputs,
                                                     outputs,
                                                     initializer=None),
                                   producer_name='NVIDIA')
    onnx.save(onnx_model, path)


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')

def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, multi_query_mode, configs):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param configs: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(configs.dtype)

    # Initialize Module
    tensorrt_llm_qwen = tensorrt_llm.models.QwenForCausalLM(
        num_layers=configs.n_layer,
        num_heads=configs.n_head,
        hidden_size=configs.n_embd,
        vocab_size=configs.vocab_size,
        max_position_embeddings=configs.n_positions,
        dtype=kv_dtype,
        mlp_hidden_size=configs.inter_size,
        neox_rotary_style=True,
        multi_query_mode=multi_query_mode,
        tensor_parallel=configs.world_size,  # TP only
        tensor_parallel_group=list(range(configs.world_size)))
    if configs.use_weight_only and configs.weight_only_precision == 'int8':
        tensorrt_llm_qwen = weight_only_quantize(tensorrt_llm_qwen,
                                                  QuantMode.use_weight_only())
    elif configs.use_weight_only and configs.weight_only_precision == 'int4':
        tensorrt_llm_qwen = weight_only_quantize(
            tensorrt_llm_qwen,
            QuantMode.use_weight_only(use_int4_weights=True))
    if configs.model_dir is not None and config.load_weight:
        logger.info(f'Loading HF Qwen-7B ... from {configs.model_dir}')
        tik = time.time()
        hf_qwen = AutoModelForCausalLM.from_pretrained(
            configs.model_dir,
            device_map="cpu",  # Load to CPU memory
            fp16=True, trust_remote_code=True)
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF Qwen loaded. Total time: {t}')
        load_from_hf_qwen(tensorrt_llm_qwen,
                           hf_qwen,
                           rank,
                           configs.world_size,
                           dtype=configs.dtype,
                           multi_query_mode=multi_query_mode)
        del hf_qwen

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if configs.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=configs.use_gpt_attention_plugin)
    if configs.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=configs.use_gemm_plugin)
    if configs.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype='float16')
    if configs.world_size > 1:
        network.plugin_config.set_nccl_plugin(configs.dtype)
    if configs.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_qwen.named_parameters())

        # Forward
        inputs = tensorrt_llm_qwen.prepare_inputs(configs.max_batch_size,
                                                   configs.max_input_len,
                                                   configs.max_output_len, True,
                                                   configs.max_beam_width)
        tensorrt_llm_qwen(*inputs)
        if configs.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_qwen.named_network_outputs():
                print(k, v)
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = kv_dtype
        if configs.visualize:
            model_path = os.path.join(configs.output_dir, 'test.onnx')
            to_onnx(network.trt_network, model_path)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(configs.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine

def build(rank, configs):
    torch.cuda.set_device(rank % configs.gpus_per_node)
    tensorrt_llm.logger.set_level(configs.log_level)
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir)
    multi_query_mode = (configs.n_kv_head
                        is not None) and (configs.n_kv_head != configs.n_head)

    # when doing serializing build, all ranks share one engine
    builder = Builder()

    cache = None
    for cur_rank in range(configs.world_size):
        # skip other ranks if parallel_build is enabled
        if configs.parallel_build and cur_rank != rank:
            continue
        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=configs.dtype,
            timing_cache=configs.timing_cache if cache is None else cache,
            tensor_parallel=configs.world_size,  # TP only
            parallel_build=configs.parallel_build,
            num_layers=configs.n_layer,
            num_heads=configs.n_head,
            hidden_size=configs.n_embd,
            vocab_size=configs.vocab_size,
            hidden_act=configs.hidden_act,
            max_position_embeddings=configs.n_positions,
            max_batch_size=configs.max_batch_size,
            max_input_len=configs.max_input_len,
            max_output_len=configs.max_output_len,
            int8=configs.quant_mode.has_act_and_weight_quant(),
            opt_level=configs.builder_opt,
            multi_query_mode=multi_query_mode)
        engine_name = get_engine_name(MODEL_NAME, configs.dtype, configs.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, multi_query_mode, configs)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not configs.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(configs.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(
            builder_config, os.path.join(configs.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."

class QwenConfig(object):

    def __init__(self):
        self.builder_opt = 5
        self.dtype = "float16"
        self.enable_debug_output = False
        self.gpus_per_node = 1
        self.inter_size = 22016
        self.log_level = 'info'
        self.max_batch_size = 1
        self.max_beam_width = 1
        self.max_input_len = 4096 # 8192
        self.max_output_len = 1024 # 2048
        self.model_dir = "Qwen-7B"
        self.n_embd = 4096
        self.n_head = 32
        self.n_kv_head = None
        self.n_layer = 32
        self.n_positions = 8192
        self.output_dir = 'qwen_trt_engine'
        self.parallel_build = False
        self.quant_mode = QuantMode(0)
        self.remove_input_padding = False
        self.timing_cache = True
        self.use_gemm_plugin = "float16"
        self.use_gpt_attention_plugin = "float16"
        self.use_weight_only = False
        self.weight_only_precision = 'int8'
        self.visualize = False
        self.vocab_size = 151936
        self.hidden_act = "silu"
        self.world_size = 1
        self.load_weight = True

class QwenConfigWeightOnlyInt8(QwenConfig):
    def __init__(self):
        super().__init__()
        self.use_weight_only = True
        self.weight_only_precision = 'int8'

class QwenConfigWeightOnlyInt4(QwenConfig):
    def __init__(self):
        super().__init__()
        self.use_weight_only = True
        self.weight_only_precision = 'int4'

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python build.py --use-fp16/--use-int8-weightonly/--use-int4-weightonly")
        exit(-1)

    if sys.argv[1] == "--use-fp16":
        config = QwenConfig()
    elif sys.argv[1] == "--use-int8-weightonly":
        config = QwenConfigWeightOnlyInt8()
    elif sys.argv[1] == "--use-int4-weightonly":
        config = QwenConfigWeightOnlyInt4()
    else:
        print("Usage: python build.py --use-fp16/--use-int8-weightonly/--use-int4-weightonly")
        exit(-1)
    
    logger.set_level(config.log_level)
    tik = time.time()
    if config.parallel_build and config.world_size > 1 and \
            torch.cuda.device_count() >= config.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {config.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=config.world_size, args=(config, ))
    else:
        config.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        build(0, config)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {config.world_size} engines: {t}')
        
        