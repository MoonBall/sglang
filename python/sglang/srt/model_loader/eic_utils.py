from typing import List, Dict, Optional

import eic
import torch
import os
import yaml
import time
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)

dtype_to_int = {
    torch.bool: 0,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 2,
    torch.float16: 3,
    torch.bfloat16: 4,
    torch.float32: 5,
    torch.float64: 6,
    torch.int8: 7,
    torch.int16: 8,
    torch.int32: 9,
    torch.int64: 10,
    torch.uint8: 11,
    torch.uint16: 12,
    torch.uint32: 13,
    torch.uint64: 14,
}

int_to_dtype = {
    0: torch.bool,
    1: torch.float8_e4m3fn,
    2: torch.float8_e5m2,
    3: torch.float16,
    4: torch.bfloat16,
    5: torch.float32,
    6: torch.float64,
    7: torch.int8,
    8: torch.int16,
    9: torch.int32,
    10: torch.int64,
    11: torch.uint8,
    12: torch.uint16,
    13: torch.uint32,
    14: torch.uint64,
}


def get_int_from_dtype(dtype: torch.dtype) -> int:
    return dtype_to_int[dtype]


def get_dtype_from_int(dtype: int) -> torch.dtype:
    return int_to_dtype[dtype]


def generate_tensor_metadata(tensor: torch.Tensor) -> Dict[str, any]:
    meta : Dict[str, any] = {}
    # dtype, shape
    meta["dtype"] = get_int_from_dtype(tensor.dtype)
    meta["shape"] = tuple(tensor.shape)
    return meta


def get_tensor_from_metadata(meta: Dict[str, any]) -> torch.Tensor:
    dtype = get_dtype_from_int(meta["dtype"])
    return torch.empty(meta["shape"], dtype=dtype)


def _make_dir(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        logger.info(f"create dir '{path}' success")
    except OSError as e:
        logger.info(f"create dir '{path}' error {e}")
        exit(1)


def _get_config_key(model_path: str) -> str:
    return os.path.join(model_path, "config")


def _get_config_file_key(model_path: str, filename: str) -> str:
    return os.path.join(model_path, filename)


def upload_model_config(eic_client, model_path: str, file_list: List[str]):
    """
    upload model config to eic
    """
    file_num = len(file_list)
    file_name_total_length = 0
    base_name_list: List[str] = []
    for i, filename in enumerate(file_list):
        base_name = os.path.basename(filename)
        base_name_list.append(base_name)
        file_name_total_length += len(base_name)

    # json format: { file_name_list[]}
    config_json = {
        "file_num": file_num,
        "file_name_list": base_name_list
    }
    config_json_bytes = json.dumps(config_json).encode()
    config_buffer = torch.from_numpy(np.frombuffer(
        config_json_bytes, dtype=np.int8).copy())
    logger.info(f"start upload model config to eic, json: {config_json}")

    keys: List[str] = [_get_config_key(model_path)]
    tensors: List[torch.Tensor] = [config_buffer]

    # read file and make tensor
    for i, filename in enumerate(file_list):
        with open(filename, "rb") as fin:
            buffer = fin.read()
        tensor = torch.from_numpy(np.frombuffer(buffer, dtype=np.int8).copy())
        keys.append(_get_config_file_key(model_path, base_name_list[i]))
        tensors.append(tensor)

    batch_size = 20
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        batch_tensors = tensors[i:i + batch_size]
        eic_client.set(batch_keys, batch_tensors)
    logger.info(f"upload model config to eic done, file num: {file_num}")


def download_model_config(eic_client, model_path: str, local_path: str) -> bool:
    """
    download model config from eic
    """
    config_key = _get_config_key(model_path)
    logger.info(f"start download model config from eic, key: {config_key}")
    keys = [config_key]
    tensors = eic_client.get(keys)
    if tensors is None:
        logger.error(
            f"download model config from eic failed, key: {config_key}")
        return False
    bytes_data = tensors[0].numpy().tobytes()
    config_json = json.loads(bytes_data.decode())
    file_name_list = config_json["file_name_list"]

    data_keys: List[str] = []
    data_tensors: List[torch.Tensor] = []
    for i, filename in enumerate(file_name_list):
        data_keys.append(_get_config_file_key(model_path, filename))

    batch_size = 20
    for i in range(0, len(data_keys), batch_size):
        batch_keys = data_keys[i:i + batch_size]
        batch_tensors = eic_client.get(batch_keys)
        data_tensors.extend(batch_tensors)

    _make_dir(local_path)
    for i, tensor in enumerate(data_tensors):
        filename = file_name_list[i]
        local_path_file = os.path.join(local_path, filename)
        if os.path.exists(local_path_file) and os.path.getsize(local_path_file) == tensor.numel() * tensor.element_size():
            logger.info(
                f"model config file {local_path_file} already exists, skip")
            continue
        with open(local_path_file, "wb") as fout:
            fout.write(tensor.numpy().tobytes())
            logger.info(
                f"download model config from eic success, file: {local_path_file}")


class EICModelClient:
    """
    The remote url should start with "eic://" and only have one host-port pair
    """

    def __init__(self):
        config_file = '/sgl-workspace/config/remote-eic.yaml'
        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            AssertionError("remote_url is None")

        endpoint = remote_url[len('eic://'):]
        eic_instance_id = config.get("eic_instance_id", None)
        eic_thread_num = config.get("eic_thread_num", 6)
        eic_log_dir = config.get("eic_log_dir", None)
        eic_log_level = config.get("eic_log_level", 1)
        # tcp: 0, rdma: 2
        eic_trans_type = config.get("eic_trans_type", 2)
        eic_flag_file = config.get("eic_flag_file", None)

        logger.info(
            f"eic init start, pid: {os.getpid()}, "
            f"endpoint: {endpoint}, "
            f"eic_instance_id: {eic_instance_id}, "
            f"eic_thread_num: {eic_thread_num}, "
            f"eic_log_dir: {eic_log_dir}, "
            f"eic_log_level: {eic_log_level}, "
            f"eic_trans_type: {eic_trans_type}, "
            f"eic_flag_file: {eic_flag_file}")
        _make_dir(eic_log_dir)

        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        if eic_flag_file:
            init_option.flag_file = eic_flag_file
        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error(f"fail to init eic client, ret: {ret}")
            exit(1)

        logger.info(f"eic init success")

    @classmethod
    def instance(cls):
        if not hasattr(EICModelClient, "_instance"):
            EICModelClient._instance = EICModelClient()
        return EICModelClient._instance

    @classmethod
    def release_instance(cls):
        if not hasattr(EICModelClient, "_instance"):
            return
        del EICModelClient._instance

    def exists(self, keys: List[str]) -> bool:
        logger.debug(f"eic exists {keys}")

        keys = eic.StringVector()
        keys.extend(keys)
        exist_option = eic.ExistOption()
        status_code, exist_outcome = self.connection.mexist(keys, exist_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug(
                f"eic exists {keys} failed, status_code {status_code}")

        err_code = exist_outcome.status_codes[0]
        success = (err_code == eic.StatusCode.SUCCESS)
        if success:
            logger.debug(f"eic exists {keys} success")
        else:
            logger.debug(f"eic exists {keys} failed, err_code {err_code}")
        return success

    def get(self, keys: List[str], tensors: Optional[List[torch.Tensor]] = None) -> Optional[List[torch.Tensor]]:
        logger.debug(f"eic get {keys}")

        get_data_start_time = time.perf_counter()
        data_keys = eic.StringVector()
        data_keys.extend(keys)

        if tensors is not None:
            data_vals = eic.IOBuffers()
            for tensor in tensors:
                data_vals.append(tensor.data_ptr(), tensor.element_size()
                                 * tensor.numel(), False)
        else:
            data_vals = None

        get_option = eic.GetOption()
        get_option.ns = ""
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mget {keys} failed, status_code {status_code}")
            for i, err_code in enumerate(get_outcome.status_codes):
                if err_code != eic.StatusCode.SUCCESS:
                    logger.error(
                        f"eic mget {keys[i]} failed, err_code {err_code}")
            return None
        get_data_end_time = time.perf_counter()
        get_data_execution_time = (
            get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {keys} data cost %.2f ms",
                     get_data_execution_time * 1e3)
        if tensors is None:
            tensors = []
            for i, data_val in enumerate(data_vals):
                byte_data = data_val.encode()
                tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=np.int8).copy())
                tensors.append(tensor)

        return tensors

    def set(self, input_keys: List[str], tensors: List[torch.Tensor]) -> None:
        logger.debug(f"eic set {input_keys}")

        keys = eic.StringVector()
        vals = eic.IOBuffers()
        # hold cpu tensor in case of gc
        cpu_tensor_refs : List[torch.Tensor] = []
        # set data key & value
        keys.extend(input_keys)
        for tensor in tensors:
            cpu_tensor = tensor.cpu()
            cpu_tensor_refs.append(cpu_tensor)
            vals.append(cpu_tensor.data_ptr(), cpu_tensor.element_size()
                        * cpu_tensor.numel(), False)
        # set options
        set_option = eic.SetOption()
        set_option.ns = ""
        set_option.ttl_second = -1
        status_code, set_outcome = self.connection.mset(keys, vals, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {keys} failed, status_code {status_code}")
            for i, err_code in enumerate(set_outcome.status_codes):
                if err_code != eic.StatusCode.SUCCESS:
                    logger.error(
                        f"eic mset {keys[i]} failed, err_code {err_code}")
            raise RuntimeError(f"eic mset failed, code: {status_code}")

    def close(self) -> None:
        if self.connection:
            self.connection = None

    def allocate_zero_copy_buffer(self, length):
        return self.connection.allocate_managed_buffer(length)

    def release_zero_copy_buffer(self, ptr, length):
        return self.connection.free_managed_buffer(ptr, length)