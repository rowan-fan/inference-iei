
import os
import glob
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_nvidia_gpu_count() -> int:
    """
    Get count of available NVIDIA GPUs.
    
    Returns:
        int: Number of available GPUs, 0 if none found.
    """
    gpu_path = "/proc/driver/nvidia/gpus/"
    try:
        if not os.path.exists(gpu_path):
            return 0
            
        count = len([name for name in os.listdir(gpu_path) 
                     if os.path.isdir(os.path.join(gpu_path, name))])
        
        nvidia_devices = [dev for dev in os.listdir("/dev") if dev.startswith("nvidia") and not dev.endswith("ctl")]
        if not nvidia_devices:
            # This can happen if GPU driver paths are mounted in a container but devices are not.
            return 0
            
        return count
    except (FileNotFoundError, PermissionError):
        return 0

def _get_mx_gpu_count() -> int:
    """
    Get count of available MX GPUs by checking renderD* devices.
    
    Returns:
        int: Number of available MX GPUs, 0 if none found.
    """
    try:
        render_devices = glob.glob("/dev/dri/renderD*")
        return len(render_devices)
    except Exception:
        return 0

def get_default_gpu_ids() -> list[int]:
    """
    Detects available GPUs and returns a list of their IDs.
    It first checks for NVIDIA GPUs, then for MX GPUs.
    If no GPUs are found, it assumes CPU mode and returns an empty list.

    Returns:
        list[int]: A list of GPU indices (e.g., [0, 1] for 2 GPUs) or an empty list for CPU.
    """
    nvidia_count = _get_nvidia_gpu_count()
    if nvidia_count > 0:
        logging.info(f"Found {nvidia_count} NVIDIA GPUs. Using all of them by default.")
        return list(range(nvidia_count))
    
    mx_count = _get_mx_gpu_count()
    if mx_count > 0:
        logging.info(f"Found {mx_count} MX GPUs. Using all of them by default.")
        return list(range(mx_count))
        
    logging.info("No NVIDIA or MX GPUs found. Assuming CPU mode.")
    return []

