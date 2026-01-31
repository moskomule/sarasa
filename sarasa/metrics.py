import dataclasses
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch._utils import _get_device_module

from sarasa.config import Config
from sarasa.utils import rank


# ported from torchtitan
# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, MI325X, MI355X and Intel PVC
def get_peak_flops(device_name: str) -> float:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [line for line in result.stdout.splitlines() if "NVIDIA" in line and "H100" in line]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 2.25e15
    elif "MI355X" in device_name:
        # MI355X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
        return 2500e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


@dataclasses.dataclass(slots=True)
class DevMemStats:
    max_active_gib: float
    max_active_perc: float
    max_reserved_gib: float
    max_reserved_perc: float
    num_alloc_retries: int
    num_ooms: int


class DeviceMemoryMonitor:
    def __init__(
        self,
        device: torch.device,
    ) -> None:
        self.device = device
        try:
            _, self.total_mem = torch.accelerator.get_memory_info(self.device)
            self.device_name = _get_device_module(self.device.type).get_device_name(self.device)
        except RuntimeError:
            if self.device.type == "mps":
                self.total_mem = torch.mps.recommended_max_memory()
                self.device_name = "Apple Silicon GPU"
            else:
                raise NotImplementedError(f"Device memory monitor not implemented for device type: {self.device.type}")

        self.reset_peak_stats()
        try:
            torch.accelerator.empty_cache()
        except RuntimeError:
            logger.error(f"Failed to empty cache for device type: {self.device.type}")

    @staticmethod
    def to_gib(bytes: int) -> float:
        return bytes / (1024**3)

    def reset_peak_stats(self) -> None:
        try:
            torch.accelerator.reset_peak_memory_stats(self.device)
        except RuntimeError:
            logger.error(f"Failed to reset peak memory stats for device type: {self.device.type}")

    def get_peak_stats(self) -> DevMemStats:
        try:
            info = torch.accelerator.memory_stats(self.device)
        except RuntimeError:
            logger.error(f"Failed to get peak memory stats for device type: {self.device.type}")
            info = {}

        max_active = info.get("active_bytes.all.peak", -1)
        max_reserved = info.get("reserved_bytes.all.peak", -1)
        num_retries = info.get("num_alloc_retries", -1)
        num_ooms = info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(f"{num_retries} {self.device.type.upper()} memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {self.device.type.upper()} OOM errors thrown.")

        return DevMemStats(
            max_active_gib=self.to_gib(max_active),
            max_active_perc=max_active / self.total_mem * 100,
            max_reserved_gib=self.to_gib(max_reserved),
            max_reserved_perc=max_reserved / self.total_mem * 100,
            num_alloc_retries=num_retries,
            num_ooms=num_ooms,
        )


class BaseReporter:
    def config(
        self,
        config: dict[str, Any],
    ) -> None:
        raise NotImplementedError()

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()


class TensorboardReporter(BaseReporter):
    def __init__(
        self,
        log_dir: Path,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, max_queue=1000)

        logger.info(f"TensorBoard log is available at {log_dir}")

    def config(
        self,
        config: dict[str, Any],
    ) -> None:
        for k, v in config.items():
            self.writer.add_text(f"config/{k}", str(v))

    def log(
        self,
        metrics: dict[str, float],
        step: int,
    ) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()


class MetricsProcessor:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        flops_per_token: int,
    ) -> None:
        self.reporters = []
        if config.metrics.all_node or rank() == 0:
            if config.metrics.use_tensorboard:
                log_dir = config.output_dir / "tensorboard" if config.output_dir else Path("./tensorboard")
                self.reporters.append(TensorboardReporter(log_dir=log_dir))

        for reporter in self.reporters:
            reporter.config(config=dataclasses.asdict(config))

        self.device_mem_monitor = DeviceMemoryMonitor(device)
        self.log_freq = config.metrics.log_freq
        self.time_last_log = time.perf_counter()
        gpu_peak_flops = get_peak_flops(self.device_mem_monitor.device_name)
        logger.info(f"Detected device: {self.device_mem_monitor.device_name}, Peak FLOPS: {gpu_peak_flops}")
        self.gpu_peak_flops = gpu_peak_flops
        self.ntokens_since_last_log = 0
        self.flops_per_token = flops_per_token
        self.data_load_times: list[float] = []
        self.reset()

    def should_log(
        self,
        step: int,
    ) -> bool:
        return step == 1 or step % self.log_freq == 0

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        time_delta = time.perf_counter() - self.time_last_log
        device_mem_stats = self.device_mem_monitor.get_peak_stats()
        time_ete = time_delta / self.log_freq
        time_data_load = sum(self.data_load_times) / len(self.data_load_times) if self.data_load_times else 0.0
        time_data_load_perc = 100 * time_data_load / time_ete if time_ete > 0 else 0.0

        metrics = {
            "loss/avg": global_avg_loss,
            "loss/max": global_max_loss,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_perc,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_perc,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
            "time/end-to-end(s)": time_ete,
            "time/data_load(s)": time_data_load,
            "time/data_load(%)": time_data_load_perc,
        }

        log = (
            f"[Step {step:>10}] loss: {global_avg_loss:.4f}, memory: {device_mem_stats.max_reserved_gib:.2f} GiB, "
            f"time(s): {time_ete:.2f}sec (data load ratio: {time_data_load_perc:.1f}%)"
        )

        if extra_metrics is not None:
            metrics.update(extra_metrics)

        if self.flops_per_token > 0:
            tps = self.ntokens_since_last_log / time_delta
            mfu = 100 * self.flops_per_token * tps / self.gpu_peak_flops
            tflops = self.flops_per_token * tps / 1e12

            metrics.update({
                "throughput(tps)": tps,
                "tflops": tflops,
                "mfu(%)": mfu,
            })
            log += f", tflops: {tflops:.2f}, mfu: {mfu:.2f}%"

        for reporter in self.reporters:
            reporter.log(metrics, step)

        logger.info(log)

        self.reset()

    def reset(self) -> None:
        self.ntokens_since_last_log = 0
        self.data_load_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_mem_monitor.reset_peak_stats()

    def val_log(
        self,
        step: int,
        val_loss: float,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        for reporter in self.reporters:
            reporter.close()
