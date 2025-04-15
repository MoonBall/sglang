from typing import Optional, List, Tuple
from sglang.srt.mem_cache.memory_pool import MemoryStateInt, KVCache, MHATokenToKVPool, MLATokenToKVPool, debug_timing, synchronized
import os
import eic
import time
import logging
import torch
import yaml
import threading

logger = logging.getLogger(__name__)
TensorPoolSize = 1024

REMOTE_EIC_YAML_ENV_VAR = "REMOTE_EIC_YAML"

class FlexibleTensorSizePool:
    def __init__(self, conn, device: str):
        self.block_pools = {}
        self.block_pool_idx = {}

        self._init = False
        self.connection = conn

        self.device = 'cpu'
        self.total_size = 0

    def make_key(self, size, dtype):
        return f"{size}_{dtype}"

    def get_item_with_shape(self, size, dtype):
        key = self.make_key(size, dtype)
        if key not in self.block_pools:
            self.block_pools[key] = [self.add_item(
                size, dtype, self.device) for _ in range(TensorPoolSize)]
            self.block_pool_idx[key] = 0
            logger.info(
                f"add new item shape {key}, total_size {self.total_size}")

        ret = self.block_pools[key][self.block_pool_idx[key]]
        self.block_pool_idx[key] = (
            self.block_pool_idx[key] + 1) % TensorPoolSize
        return ret

    def add_item(self, size, dtype, device):
        item = torch.zeros((size,), device=device, dtype=dtype)
        self.total_size += item.numel() * item.element_size()
        return item

class FlexibleKVCacheMemoryPool:
    def __init__(self, conn, device: str, kv_cache_shape, kv_cache_dtype):
        self._init = False
        self.connection = conn

        self.device = 'cpu'

        """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype

        self.max_kv_cache_num = TensorPoolSize * 2

        self.mempool = torch.zeros((self.max_kv_cache_num, ) + kv_cache_shape, dtype=kv_cache_dtype, device=device)
        self.kv_cache_idx = 0

        self.kv_cache_numel = 1
        for i in self.kv_cache_shape: self.kv_cache_numel*= i

        meminfo = eic.MemoryInfo()
        meminfo.type = eic.MemoryType.MEMORY_CUDA
        meminfo.cuda_id = 0

        vals = eic.IOBuffers()
        vals.append(self.mempool.data_ptr(), self.mempool.numel()
                    * self.mempool.element_size(), True)
        self.connection.register_memory(vals, meminfo)

        logger.info(
            f"register memory memory pool shape {self.kv_cache_shape}, dtype {self.kv_cache_dtype}, kv_cache_num {self.max_kv_cache_num}, \
device {device}, total_size {self.max_kv_cache_num * (self.mempool[0].numel() * self.mempool[0].element_size())}")

    def try_allocate_kv_cache(self, shape, dtype, count):
        if self.kv_cache_dtype != dtype or self.kv_cache_shape != shape:
            logger.error(
                f'allocate from mempool failed, self.kv_cache_shape {self.kv_cache_shape}, dtype {self.kv_cache_dtype}, require shape {shape}, dtype {dtype}')
            return None

        if count > self.max_kv_cache_num:
            logger.error(
                f'allocate from mempool failed, self.kv_cache_shape {self.kv_cache_shape}, dtype {self.kv_cache_dtype}, require count {count}, max_kv_cache_num {self.max_kv_cache_num}')
            return None

        if self.kv_cache_idx + count > self.max_kv_cache_num:
            self.kv_cache_idx = 0

        ret = self.mempool[self.kv_cache_idx:self.kv_cache_idx+count]
        self.kv_cache_idx = (self.kv_cache_idx + count) % self.max_kv_cache_num
        return ret


class EICKVClient:
    """
    The remote url should start with "eic://" and only have one host-port pair
    """

    def __init__(self, endpoint: str, kv_cache_dtype, kv_cache_shape, device="cpu"):
        if os.environ.get(REMOTE_EIC_YAML_ENV_VAR) is not None:
            logger.info(f"eic init with env var {REMOTE_EIC_YAML_ENV_VAR}")
            config_file = os.environ.get(REMOTE_EIC_YAML_ENV_VAR)
        else:
            config_file = '/sgl-workspace/config/remote-eic.yaml'
            logger.info(f"eic init with default config, config_file {config_file}")

        if os.path.exists(config_file) is False:
            logger.error(f'config file {config_file} not exists')
            exit(1)

        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            AssertionError("remote_url is None")

        endpoint = remote_url[len('eic://'):]

        logger.info(f'eic remote_url:'+ remote_url + ' endpoint: ' + endpoint)

        eic_instance_id = config.get("eic_instance_id", None)
        logger.info(f'eic instance_id: {eic_instance_id}')

        eic_thread_num = config.get("eic_thread_num", 6)
        logger.info(f'eic thread_num: {eic_thread_num}')

        eic_log_dir = config.get("eic_log_dir", None)
        logger.info(f'eic log_dir: {eic_log_dir}')

        eic_log_level = config.get("eic_log_level", 2)
        logger.info(f'eic log_level: {eic_log_level}')

        eic_trans_type = config.get("eic_trans_type", 3)
        logger.info(f'eic trans_type: {eic_trans_type}')

        eic_flag_file = config.get("eic_flag_file", None)
        logger.info(f'eic flag_file: {eic_flag_file}')

        if not os.path.exists(eic_log_dir) and not os.path.isdir(eic_log_dir):
            os.makedirs(eic_log_dir, exist_ok=True)

        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        init_option.flag_file = eic_flag_file
        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error(f"fail to init eic client, ret: {ret}")
            exit(1)

        self.device = device

        self.trans_type = eic.TransportType(eic_trans_type)
        self.shape_mem_pool = FlexibleTensorSizePool(
            self.connection, self.device)

        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_mem_pool = FlexibleKVCacheMemoryPool(
            self.connection, self.device, self.kv_cache_shape, self.kv_cache_dtype)

    def exists(self, key: str) -> bool:
        logger.debug(f"eic exists {key}")
        keys = eic.StringVector()
        keys.append(key)
        exist_option = eic.ExistOption()
        status_code, exist_outcome = self.connection.mexist(keys, exist_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug(f"eic exists {key} failed, status_code {status_code}")

        err_code = exist_outcome.status_codes[0]
        success = (err_code == eic.StatusCode.SUCCESS)
        if success:
            logger.debug(f"eic exists {key} success")
        else:
            logger.debug(f"eic exists {key} failed, err_code {err_code}")
        return success

    def get(self, keys: str) -> Optional[torch.Tensor]:
        logger.debug(f"eic get {keys}")

        # Get Data: generate data keys and vals
        get_data_start_time = time.perf_counter()
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        objs = []
        for i, key in enumerate(keys):
            dtype = self.kv_cache_dtype
            shape = self.kv_cache_shape
            logger.debug(f"get tensor shape {shape}, dtype {dtype}")

            registered = False
            item = self.kv_cache_mem_pool.try_allocate_kv_cache(shape, dtype)
            if item is None:
                obj = torch.empty(shape, dtype=dtype, device='cpu')
                logger.error("can not allocate tensor from pool")
            else:
                obj = item
                registered = True

            objs.append(obj)
            data_keys.append(key)
            data_vals.append(obj.data_ptr(), obj.element_size()
                             * obj.numel(), registered)

        # Get data: recv data buffer tensor
        get_option = eic.GetOption()
        get_option.ns = ""
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mget {keys} failed, status_code {status_code}")
            return None

        for i, err_code in enumerate(get_outcome.status_codes):
            success = (err_code == eic.StatusCode.SUCCESS)
            if success:
                logger.debug(f"eic get data {keys[i]} success")
            else:
                logger.error(f"eic get data {keys[i]} failed, err_code {err_code}")
                return None

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (
            get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {keys} data cost %.2f ms",
                     get_data_execution_time * 1e3)

        return objs

    def batch_get(self, keys: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        logger.debug(f"eic get {len(keys)}")

        # Get Data: generate data keys and vals
        get_data_start_time = time.perf_counter()
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        objs = None
        success_mask = [True for _ in range(len(keys))]
        count = len(keys)

        registered = False
        items = self.kv_cache_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count)
        if items is None:
            objs = torch.empty((count, ) + self.kv_cache_shape,
                              dtype=self.kv_cache_dtype, device='cpu')
            logger.error("can not allocate tensor from pool")
        else:
            objs = items
            registered = True

        for i, key in enumerate(keys):
            data_keys.append(key)
            data_vals.append(objs[i].data_ptr(), objs[i].element_size()
                             * objs[i].numel(), registered)

        # Get data: recv data buffer tensor
        get_option = eic.GetOption()
        get_option.ns = ""
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals)
        if status_code != eic.StatusCode.SUCCESS:
            if status_code == eic.StatusCode.PARTIAL_FAILED:
                for i, err_code in enumerate(get_outcome.status_codes):
                    success = (err_code == eic.StatusCode.SUCCESS)
                    if success:
                        logger.debug(f"eic get data {keys[i]} success")
                    else:
                        logger.error(
                            f"eic get data {keys[i]} failed, err_code {err_code}")
                        success_mask[i] = False
            else:
                logger.error(
                    f"eic mget {len(keys)} keys failed, status_code {status_code}")
                return None, []

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (
            get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {count} keys data cost %.2f us",
                     get_data_execution_time)
        return objs, success_mask

    def set(self, keys: str, obj_inputs: torch.Tensor) -> None:
        logger.debug(f"eic set {len(keys)} keys")
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()

        for key, value in zip(keys, obj_inputs):
            if not value.is_cpu:
                obj = self.shape_mem_pool.get_item_with_shape(value.numel(), value.dtype)
                if obj is not None:
                    obj = obj.reshape(value.shape)
                    obj.copy_(value)
                else:
                    obj = value.cpu()
            else:
                obj = value

            # set data key & value
            keys_vec.append(key)
            vals_vec.append(obj.data_ptr(), obj.element_size()
                        * obj.numel(), False)

        # set options
        set_option = eic.SetOption()
        set_option.ns = ""
        set_option.ttl_second = -1
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {len(keys)} failed, status_code {status_code}")
            return False
        else:
            logger.debug(f"eic mset {len(keys)} success")

        err_code = set_outcome.status_codes[0]
        if err_code == eic.StatusCode.SUCCESS:
            logger.debug(f"set data key {len(keys)} success")
            return True
        else:
            logger.error(
                f"set data key {len(keys)} failed, err_code {err_code}")
            return False

    def support_batched_get(self) -> bool:
        return True

class EICBaseTokenToKVPoolHost:

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float = 4.0,
        device: str = "cpu",
        rank: int = 0,
    ):
        assert (
            host_to_device_ratio >= 1
        ), "The eic memory should be larger than the device memory with the current protocol"
        # todo, other ways of configuring the size

        self.device_pool = device_pool
        self.host_to_device_ratio = host_to_device_ratio
        self.device = device

        self.size = int(device_pool.size * host_to_device_ratio)
        self.dtype = device_pool.store_dtype

        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros((self.size,), dtype=torch.uint8, device=self.device)
        self.free_slots = torch.arange(self.size, dtype=torch.int32)
        self.can_use_mem_size = self.size

        # A lock for synchronized() operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = rank
        self.host_ip = self._get_host_ip()
        self.split_dim = 2

    def _encode_key(self, indices):
        return [f"{self.host_ip}_{self.rank}_{index}" for index in indices.to("cpu").tolist()]

    def _get_host_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())

    # TODO: catch exception
    def get_flat_data(self, indices) -> Tuple[Optional[torch.Tensor], List[bool]]:
        logger.debug(f"get_flat_data indices {indices}")
        keys = self._encode_key(indices)
        bs = TensorPoolSize
        ret = []
        masks = []

        for i in range(0, len(keys), bs):
            key = keys[i:i+bs]
            objs, success_mask = self.eic_client.batch_get(key)
            if objs is None:
                logger.error(f"get_flat_data keys {key} failed, eic_client return none")
                return None, []
            copy_objs = objs.clone()
            ret.extend([copy_objs[i] for i in range(copy_objs.shape[0])])
            masks.extend(success_mask)

        if len(ret) == 0:
            logger.error(f"get_flat_data keys size {len(keys)} failed, eic_client return none, ret {ret}")
            return None, []

        flat_data = torch.stack(ret, dim=self.split_dim)
        return flat_data, masks

    def assign_flat_data(self, indices, flat_data):
        logger.debug(f"assign_flat_data indices {indices}")

        keys = self._encode_key(indices)
        flat_data = flat_data.to("cpu")
        values = torch.split(flat_data, 1, dim=self.split_dim)
        bs = TensorPoolSize

        for i in range(0, len(keys), bs):
            key = keys[i:i+bs]
            value = values[i:i+bs]
            ret = self.eic_client.set(key, value)
            if not ret:
                logger.error(f"assign_flat_data keys {key} failed, eic_client return none")
                return False

        return True

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        return self.assign_flat_data(indices, flat_data)

    @synchronized()
    def clear(self):
        self.mem_state.fill_(0)
        self.can_use_mem_size = self.size
        self.free_slots = torch.arange(self.size, dtype=torch.int32)

    @synchronized()
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.can_use_mem_size:
            return None

        # todo: de-fragementation
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        self.mem_state[select_index] = MemoryStateInt.RESERVED
        self.can_use_mem_size -= need_size

        return select_index

    @synchronized()
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized()
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized()
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized()
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized()
    def update_backup(self, indices: torch.Tensor):
        assert self.is_synced(indices), (
            f"The host memory slots should be in SYNCED state before turning into BACKUP. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized()
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized()
    def protect_write(self, indices: torch.Tensor):
        assert self.is_reserved(indices), (
            f"The host memory slots should be RESERVED before write operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized()
    def protect_load(self, indices: torch.Tensor):
        assert self.is_backup(indices), (
            f"The host memory slots should be in BACKUP state before load operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized()
    def complete_io(self, indices: torch.Tensor):
        assert self.is_protected(indices), (
            f"The host memory slots should be PROTECTED during I/O operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.SYNCED

    def available_size(self):
        return len(self.free_slots)

    @synchronized()
    def free(self, indices: torch.Tensor) -> int:
        self.mem_state[indices] = MemoryStateInt.IDLE
        self.free_slots = torch.concat([self.free_slots, indices])
        self.can_use_mem_size += len(indices)
        return len(indices)


class EICMHATokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float = 2.0,
        page_size: int = 1,
        device: str = "cpu",
        rank: int = 0,
    ):
        super().__init__(device_pool, host_to_device_ratio, device, rank)
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.size_per_token = (
            self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
        )
        self.kvcache_shape = (2, self.layer_num, self.head_num, self.head_dim)
        self.eic_client = EICKVClient(None, self.dtype, self.kvcache_shape, device_pool.device)



class EICMLATokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float = 2.0,
        page_size: int = 1,
        device: str = "cpu",
        rank: int = 0,
    ):
        super().__init__(device_pool, host_to_device_ratio, device, rank)
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.size_per_token = (self.kv_lora_rank + self.qk_rope_head_dim) * 1 * self.dtype.itemsize
        self.kvcache_shape = (self.layer_num, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        self.eic_client = EICKVClient(None, self.dtype, self.kvcache_shape, device_pool.device)
        self.split_dim = 1
