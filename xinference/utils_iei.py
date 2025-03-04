# Copyright 2025-2025 IEI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import subprocess
import os
import asyncio
import time
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Cache for probe results
_probe_cache = {
    "result": False,
    "last_check_time": 0,
    "checking": False,
    "error": None
}

# Cache expiration time in seconds
PROBE_CACHE_TTL = 30

# Background task reference
_background_task = None

def local_probe() -> bool:
    """
    Execute the local probe.py script to check model health status.
    
    The script is executed with a 10-minute timeout to ensure it doesn't hang indefinitely.
    
    Returns:
        bool: True if the probe script executed successfully (exit code 0), False otherwise.
    """
    probe_script_path = os.environ.get("PROBE_SCRIPT_PATH", "/workspace/probe.py")
    
    # Check if the probe script exists
    if not os.path.exists(probe_script_path):
        logger.error(f"Probe script not found at {probe_script_path}")
        return False
    
    try:
        # Execute the probe script with extended timeout (10 minutes)
        logger.info(f"Executing probe script: {probe_script_path}")
        result = subprocess.run(
            ["python3", probe_script_path], 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=600  # 10 minutes timeout
        )
        
        # Check the exit code and log appropriate message
        if result.returncode == 0:
            logger.info("Probe script executed successfully")
            return True
        else:
            logger.error(f"Probe failed with exit code {result.returncode}. Stdout: {result.stdout[:200]}... Stderr: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Probe script execution timed out after 10 minutes")
        return False
    except Exception as e:
        logger.exception(f"Error executing probe script: {e}")
        return False

def _background_probe_task():
    """
    Background task that periodically runs the probe check and updates the cache.
    
    This function runs in an infinite loop and catches all exceptions to ensure
    the background task never exits unexpectedly.
    """
    global _probe_cache
    
    logger.info("Starting background probe task")
    
    while True:
        try:
            # Mark as checking and clear previous errors
            _probe_cache["checking"] = True
            _probe_cache["error"] = None
            
            # Execute probe and update cache
            start_time = time.time()
            result = local_probe()
            elapsed = time.time() - start_time
            
            _probe_cache["result"] = result
            _probe_cache["last_check_time"] = time.time()
            _probe_cache["checking"] = False
            
            logger.info(f"Background probe completed in {elapsed:.2f}s, result: {result}")
            
        except Exception as e:
            logger.exception(f"Error in background probe task: {e}")
            _probe_cache["error"] = str(e)
            _probe_cache["checking"] = False
            _probe_cache["result"] = False
        
        # Wait before next check, regardless of success or failure
        time.sleep(PROBE_CACHE_TTL)

def start_background_probe():
    """
    Start the background probe task if not already running.
    
    This function creates a daemon thread that will automatically terminate
    when the main process exits.
    """
    global _background_task
    
    # Don't start if already running
    if _background_task is not None and _background_task.is_alive():
        logger.info("Background probe task already running")
        return
    
    # Create and start a new background thread
    _background_task = threading.Thread(
        target=_background_probe_task,
        daemon=True  # Use daemon thread to auto-terminate when main process exits
    )
    _background_task.start()
    
    logger.info(f"Background probe task started with thread ID: {_background_task.ident}")

async def direct_probe_async() -> bool:
    """
    Asynchronously check model status, using cached results when available.
    
    This function is designed to be called from async contexts like FastAPI endpoints.
    It will use the cache if valid, wait if a check is in progress, or run a new check.
    
    Returns:
        bool: True if all checks pass, False otherwise.
    """
    global _probe_cache
    
    current_time = time.time()
    cache_age = current_time - _probe_cache["last_check_time"]
    
    # Use cache if valid and not currently checking
    if not _probe_cache["checking"] and cache_age < PROBE_CACHE_TTL:
        logger.info(f"Using cached probe result (age: {cache_age:.2f}s): {_probe_cache['result']}")
        return _probe_cache["result"]
    
    # Wait for ongoing check to complete
    if _probe_cache["checking"]:
        logger.info("Probe check already in progress, waiting for result")
        wait_start = time.time()
        
        # Wait up to 30 seconds for the check to complete
        for i in range(30):
            await asyncio.sleep(1)
            if not _probe_cache["checking"]:
                wait_time = time.time() - wait_start
                logger.info(f"Got probe result after waiting {wait_time:.2f}s: {_probe_cache['result']}")
                return _probe_cache["result"]
        
        # Timeout if waiting too long
        logger.error("Timeout waiting for probe check to complete")
        return False
    
    # Mark as checking and run a new probe
    _probe_cache["checking"] = True
    _probe_cache["error"] = None
    
    try:
        # Execute synchronous function in async context
        loop = asyncio.get_event_loop()
        start_time = time.time()
        result = await loop.run_in_executor(None, local_probe)
        elapsed = time.time() - start_time
        
        # Update cache
        _probe_cache["result"] = result
        _probe_cache["last_check_time"] = time.time()
        _probe_cache["checking"] = False
        
        logger.info(f"Direct async probe completed in {elapsed:.2f}s, result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Error in direct async probe: {e}")
        _probe_cache["error"] = str(e)
        _probe_cache["checking"] = False
        _probe_cache["result"] = False
        return False

def direct_probe() -> bool:
    """
    Synchronously check model status, using cached results when available.
    
    This function is designed for non-async contexts. It will use the cache if valid,
    wait if a check is in progress, or run a new check.
    
    Returns:
        bool: True if all checks pass, False otherwise.
    """
    global _probe_cache
    
    current_time = time.time()
    cache_age = current_time - _probe_cache["last_check_time"]
    
    # Use cache if valid and not currently checking
    if not _probe_cache["checking"] and cache_age < PROBE_CACHE_TTL:
        logger.info(f"Using cached probe result (age: {cache_age:.2f}s): {_probe_cache['result']}")
        return _probe_cache["result"]
    
    # Wait for ongoing check to complete
    if _probe_cache["checking"]:
        logger.info("Probe check already in progress, waiting for result")
        wait_start = time.time()
        
        # Wait up to 30 seconds for the check to complete
        for i in range(30):
            time.sleep(1)
            if not _probe_cache["checking"]:
                wait_time = time.time() - wait_start
                logger.info(f"Got probe result after waiting {wait_time:.2f}s: {_probe_cache['result']}")
                return _probe_cache["result"]
        
        # Timeout if waiting too long
        logger.error("Timeout waiting for probe check to complete")
        return False
    
    # Mark as checking and run a new probe
    _probe_cache["checking"] = True
    _probe_cache["error"] = None
    
    try:
        # Execute probe
        start_time = time.time()
        result = local_probe()
        elapsed = time.time() - start_time
        
        # Update cache
        _probe_cache["result"] = result
        _probe_cache["last_check_time"] = time.time()
        _probe_cache["checking"] = False
        
        logger.info(f"Direct probe completed in {elapsed:.2f}s, result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Error in direct probe: {e}")
        _probe_cache["error"] = str(e)
        _probe_cache["checking"] = False
        _probe_cache["result"] = False
        return False