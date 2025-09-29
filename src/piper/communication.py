"""Cross-platform communication helper for pipe-like functionality."""

import json
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CrossPlatformCommunicator:
    """Cross-platform communication using temporary files with blocking behavior."""
    
    def __init__(self, input_name: str = "g2p_in", output_name: str = "g2p_out"):
        """
        Initialize the communicator.
        
        :param input_name: Name for the input communication channel
        :param output_name: Name for the output communication channel
        """
        self.input_name = input_name
        self.output_name = output_name
        
        # Create temporary directory for communication files
        self.temp_dir = Path(tempfile.gettempdir())
        self.input_path = self.temp_dir / input_name
        self.output_path = self.temp_dir / output_name
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Create the communication files
        self._create_communication_files()
    
    def _create_communication_files(self):
        """Create the communication files."""
        try:
            # Create empty files if they don't exist
            if not self.input_path.exists():
                self.input_path.touch()
            if not self.output_path.exists():
                self.output_path.touch()
        except Exception as e:
            raise RuntimeError(f"Failed to create communication files: {e}")
    
    def write_data(self, data: Dict[str, Any]) -> None:
        """
        Write data to the input channel.
        
        :param data: Dictionary to write
        """
        with self._lock:
            try:
                with open(self.input_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                raise RuntimeError(f"Failed to write data: {e}")
    
    def read_data(self) -> Dict[str, Any]:
        """
        Read data from the output channel with blocking behavior.
        This will block indefinitely until data is available.
        
        :return: Dictionary with data
        """
        while True:
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            # Small delay to prevent busy waiting
            time.sleep(0.01)
    
    def clear_output(self) -> None:
        """Clear the output file."""
        with self._lock:
            try:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    f.write('')
            except Exception as e:
                raise RuntimeError(f"Failed to clear output: {e}")
    
    def cleanup(self) -> None:
        """Clean up communication files."""
        try:
            if self.input_path.exists():
                self.input_path.unlink()
            if self.output_path.exists():
                self.output_path.unlink()
        except Exception as e:
            # Don't raise exception during cleanup
            pass


class CrossPlatformServer:
    """Server-side communicator for processing requests."""
    
    def __init__(self, input_name: str = "g2p_in", output_name: str = "g2p_out"):
        """
        Initialize the server communicator.
        
        :param input_name: Name for the input communication channel
        :param output_name: Name for the output communication channel
        """
        self.input_name = input_name
        self.output_name = output_name
        
        # Create temporary directory for communication files
        self.temp_dir = Path(tempfile.gettempdir())
        self.input_path = self.temp_dir / input_name
        self.output_path = self.temp_dir / output_name
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Create the communication files
        self._create_communication_files()
    
    def _create_communication_files(self):
        """Create the communication files."""
        try:
            # Create empty files if they don't exist
            if not self.input_path.exists():
                self.input_path.touch()
            if not self.output_path.exists():
                self.output_path.touch()
        except Exception as e:
            raise RuntimeError(f"Failed to create communication files: {e}")
    
    def wait_for_data(self) -> Dict[str, Any]:
        """
        Wait for data from the input channel.
        This will block indefinitely until data is available.
        
        :return: Dictionary with data
        """
        while True:
            try:
                with open(self.input_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        # Clear the input file after reading
                        with open(self.input_path, 'w', encoding='utf-8') as f:
                            f.write('')
                        return data
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            # Small delay to prevent busy waiting
            time.sleep(0.01)
    
    def send_response(self, data: Dict[str, Any]) -> None:
        """
        Send response data to the output channel.
        
        :param data: Dictionary to send
        """
        with self._lock:
            try:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                raise RuntimeError(f"Failed to send response: {e}")
    
    def cleanup(self) -> None:
        """Clean up communication files."""
        try:
            if self.input_path.exists():
                self.input_path.unlink()
            if self.output_path.exists():
                self.output_path.unlink()
        except Exception as e:
            # Don't raise exception during cleanup
            pass
