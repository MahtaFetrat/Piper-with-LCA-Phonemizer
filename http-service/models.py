from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum

class OutputFormat(str, Enum):
    FILE = "file"
    STREAM = "stream"

class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text to convert to speech",
        min_length=1,
        max_length=5000,
        example="سلام، به سرویس تبدیل متن به گفتار ما خوش آمدید!"
    )
    voice_key: str = Field(
        ...,
        description="Voice identifier from the available voices list",
        example="fa_IR-mana-medium"
    )
    speaker_id: Optional[int] = Field(
        0,
        description="Speaker ID for multi-speaker voices (0-based index)",
        ge=0,
        example=0
    )
    speed: Optional[float] = Field(
        1.0,
        description="Speech speed multiplier (0.1 = very slow, 3.0 = very fast)",
        ge=0.1,
        le=3.0,
        example=1.0
    )
    noise_scale: Optional[float] = Field(
        0.667,
        description="Speech variability/naturalness (0.0 = monotone, 1.0 = very expressive)",
        ge=0.0,
        le=1.0,
        example=0.667
    )
    noise_scale_w: Optional[float] = Field(
        0.8,
        description="Timing variability (0.0 = rigid timing, 1.0 = natural timing)",
        ge=0.0,
        le=1.0,
        example=0.8
    )
    output_format: OutputFormat = Field(
        OutputFormat.FILE,
        description="Output format: 'file' returns complete audio file, 'stream' returns chunked audio"
    )
    sentence_silence: Optional[float] = Field(
        0.0,
        description="Seconds of silence after each sentence (streaming mode only)",
        ge=0.0,
        le=5.0,
        example=0.0
    )


class TTSResponse(BaseModel):
    success: bool = Field(description="Operation success status")
    message: str = Field(description="Response message")


class VoiceInfo(BaseModel):
    name: str = Field(description="Display name of the voice")
    language: Dict = Field(description="Language information")
    quality: str = Field(description="Voice quality (low, medium, high)")
    num_speakers: int = Field(description="Number of available speakers")
    speaker_names: List[str] = Field(description="Names of available speakers")


class VoicesResponse(BaseModel):
    success: bool = Field(description="Operation success status")
    voices: Dict[str, VoiceInfo] = Field(
        description="Available voices dictionary")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Error message")
