import os
import re
import tempfile
import wave
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask
from models import TTSRequest, OutputFormat, TTSResponse, VoicesResponse
from core import (
    get_voices_config, load_voice_model,
    prepare_synthesis_config,
    synthesize_stream_audio, add_wav_header
)

router = APIRouter(prefix="/api/tts")


def cleanup_temp_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except OSError:
        pass


@router.get(
    "/voices",
    response_model=VoicesResponse,
    tags=["Voices"],
    summary="Get available voices",
    description="Returns a list of all available voices."
)
async def get_voices():
    voices_config = get_voices_config()
    return {
        "success": True,
        "voices": {
            voice_key: {
                "name": voice_info["name"],
                "language": voice_info["language"],
                "quality": voice_info["quality"],
                "num_speakers": voice_info["num_speakers"],
                "speaker_names": list(voice_info["speaker_id_map"].keys()) if voice_info.get("speaker_id_map") else []
            }
            for voice_key, voice_info in voices_config.items()
        }
    }


@router.post(
    "/synthesize",
    response_class=FileResponse,
    tags=["Speech Synthesis"],
    summary="Convert text to speech",
    responses={
        200: {
            "description": "Audio file generated successfully",
            "content": {"audio/wav": {}},
            "headers": {
                "X-Audio-Duration": {"description": "Duration of the audio in seconds"},
                "X-Voice-Key": {"description": "Voice used for synthesis"},
            },
        }
    }
)
async def synthesize_speech(request: TTSRequest):
    try:
        voices_config = get_voices_config()

        if request.voice_key not in voices_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice '{request.voice_key}' not found. Available voices: {list(voices_config.keys())}"
            )

        model = load_voice_model(request.voice_key)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load voice model: {request.voice_key}"
            )

        voice_info = voices_config[request.voice_key]
        num_speakers = voice_info["num_speakers"]

        if request.speaker_id >= num_speakers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Speaker ID {request.speaker_id} not available. Voice has {num_speakers} speakers (0-{num_speakers-1})"
            )

        synthesis_config = prepare_synthesis_config(
            request.speaker_id, num_speakers, request.speed,
            request.noise_scale, request.noise_scale_w
        )

        if request.output_format == OutputFormat.STREAM:
            return await _synthesize_streaming(model, request, synthesis_config)
        else:
            return await _synthesize_file(model, request, synthesis_config)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(e)}"
        )


async def _synthesize_file(model, request: TTSRequest, synthesis_config):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        with wave.open(temp_file_path, 'wb') as wav_file:
            processed_text = re.sub(r'\n', ' ', request.text)
            processed_text = re.sub(r'[?.:;!!؟]|\.{3}', '،', processed_text)

            model.synthesize_wav(
                text=processed_text,
                wav_file=wav_file,
                syn_config=synthesis_config
            )

        with wave.open(temp_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate

        return FileResponse(
            temp_file_path,
            media_type="audio/wav",
            filename=f"tts_output_{request.voice_key}.wav",
            headers={
                "X-Audio-Duration": str(duration),
                "X-Voice-Key": request.voice_key,
                "X-Speaker-ID": str(request.speaker_id),
                "X-Speed": str(request.speed),
                "X-Output-Format": "file"
            },
            background=BackgroundTask(cleanup_temp_file, temp_file_path)
        )

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise e


async def _synthesize_streaming(model, request: TTSRequest, synthesis_config):
    def audio_stream_generator():
        try:
            wav_header = add_wav_header(model.config.sample_rate)
            yield wav_header

            for audio_chunk in synthesize_stream_audio(model, request.text, synthesis_config):
                yield audio_chunk

                if request.sentence_silence and request.sentence_silence > 0:
                    silence_samples = int(
                        model.config.sample_rate * request.sentence_silence)
                    # 16-bit audio = 2 bytes per sample
                    silence_bytes = b'\x00\x00' * silence_samples
                    yield silence_bytes

        except Exception as e:
            print(f"❌ Streaming error: {e}")
            raise

    return StreamingResponse(
        audio_stream_generator(),
        media_type="audio/wav",
        headers={
            "X-Voice-Key": request.voice_key,
            "X-Speaker-ID": str(request.speaker_id),
            "X-Speed": str(request.speed),
            "X-Output-Format": "stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get(
    "/health",
    response_class=PlainTextResponse,
    tags=["Health & Management"],
    summary="Health check",
    description="Simple health check that returns a 'healthy' string."
)
async def health_check():
    return PlainTextResponse("healthy")
