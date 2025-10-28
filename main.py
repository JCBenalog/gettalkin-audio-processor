import os
import json
import csv
import io
import re
import uuid
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import logging
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import base64

# Google Cloud Speech-to-Text and Storage imports
try:
    from google.cloud import speech_v1p1beta1 as speech
    from google.cloud import storage
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    logging.warning("Google Cloud libraries not available - transcript generation will be disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# SECURE CONFIGURATION - Load from environment variables
def load_config():
    """Load configuration from environment variables with validation"""
    config = {
        'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY'),
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY'),
        'USER_ID': os.getenv('USER_ID'),
        'NOTIFICATION_EMAIL': os.getenv('NOTIFICATION_EMAIL'),
        # Google Cloud credentials (optional for transcript generation)
        'GOOGLE_CLOUD_PROJECT_ID': os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
        'GOOGLE_APPLICATION_CREDENTIALS_JSON': os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'),
        'GOOGLE_CLOUD_STORAGE_BUCKET': os.getenv('GOOGLE_CLOUD_STORAGE_BUCKET', 'gettalkin-temp-audio')
    }
    
    # Validate required environment variables (Google Cloud is optional)
    required_vars = ['ELEVENLABS_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY', 'USER_ID', 'NOTIFICATION_EMAIL']
    missing_vars = [key for key in required_vars if not config.get(key)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Warn if Google Cloud credentials are missing
    if not config.get('GOOGLE_CLOUD_PROJECT_ID') or not config.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
        logger.warning("Google Cloud credentials not set - word-level timestamps will be disabled")
    else:
        logger.info("Google Cloud credentials loaded - word-level timestamp generation enabled")
        logger.info(f"Using GCS bucket: {config.get('GOOGLE_CLOUD_STORAGE_BUCKET')}")
    
    logger.info("All required environment variables loaded successfully")
    return config

# Load configuration
try:
    config = load_config()
    ELEVENLABS_API_KEY = config['ELEVENLABS_API_KEY']
    SUPABASE_URL = config['SUPABASE_URL']
    SUPABASE_KEY = config['SUPABASE_KEY']
    USER_ID = config['USER_ID']
    NOTIFICATION_EMAIL = config['NOTIFICATION_EMAIL']
    GOOGLE_CLOUD_PROJECT_ID = config.get('GOOGLE_CLOUD_PROJECT_ID')
    GOOGLE_APPLICATION_CREDENTIALS_JSON = config.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    GOOGLE_CLOUD_STORAGE_BUCKET = config.get('GOOGLE_CLOUD_STORAGE_BUCKET')
    
    logger.info("Secure configuration loaded")
    logger.info(f"ElevenLabs Key: {ELEVENLABS_API_KEY[:6]}...")
    logger.info(f"Supabase URL: {SUPABASE_URL}")
    
except ValueError as e:
    logger.error(f"Configuration Error: {e}")
    logger.error("Make sure all environment variables are set in Railway dashboard")
    raise

# Voice Configuration
VOICE_CONFIG = {
    "NARRATOR": {
        "voice_id": "L0Dsvb3SLTyegXwtm47J",
        "stability": 0.30,
        "similarity_boost": 0.50,
        "style": 0.15
    },
    "Balasz": {
        "voice_id": "3T7ttZm72GpMThZ8XPZP", 
        "stability": 0.60,
        "similarity_boost": 0.70,
        "style": 0.15
    },
    "Aggie": {
        "voice_id": "xjlfQQ3ynqiEyRpArrT8",
        "stability": 0.60,
        "similarity_boost": 0.70,
        "style": 0.15
    }
}

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Supabase: {e}")
    raise

# Initialize Google Cloud clients (if credentials available)
google_speech_client = None
google_storage_client = None

if GOOGLE_CLOUD_AVAILABLE and GOOGLE_CLOUD_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS_JSON:
    try:
        credentials_dict = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        google_speech_client = speech.SpeechClient(credentials=credentials)
        google_storage_client = storage.Client(credentials=credentials, project=GOOGLE_CLOUD_PROJECT_ID)
        logger.info("Google Cloud clients initialized successfully for word-level timestamps")
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud clients: {e}")
        google_speech_client = None
        google_storage_client = None

class AudioProcessor:
    def __init__(self):
        self.pronunciation_cache = {}
        self.daily_usage = 0
        self.max_daily_lessons = int(os.getenv('DAILY_LESSON_LIMIT', '20'))
        self.temp_dir = tempfile.mkdtemp()
        logger.info("AudioProcessor initialized")
        logger.info(f"Daily lesson limit set to: {self.max_daily_lessons}")
        
    def load_pronunciation_dictionary(self) -> Dict[str, str]:
        """Load pronunciation corrections from Supabase"""
        try:
            response = supabase.table('pronunciation_fixes').select('*').execute()
            return {row['hungarian_word']: row['phonetic_spelling'] for row in response.data}
        except Exception as e:
            logger.warning(f"Could not load pronunciation dictionary: {e}")
            return {"szoba": "soba"}

    def apply_pronunciation_fixes(self, text: str) -> str:
        """Apply pronunciation corrections to Hungarian text"""
        if not self.pronunciation_cache:
            self.pronunciation_cache = self.load_pronunciation_dictionary()
        
        corrected_text = text
        for original, corrected in self.pronunciation_cache.items():
            corrected_text = corrected_text.replace(original, corrected)
        return corrected_text

    def detect_pause_context(self, narrator_text: str) -> str:
        """Detect if this NARRATOR line should trigger a pause"""
        if ':' in narrator_text.strip()[-3:]:
            return "explicit_pause"
        return "dialogue"

    def calculate_pedagogical_pause(self, following_text: str, context: str) -> float:
        """Calculate pause duration based on text complexity and pedagogical context"""
        if context == "dialogue":
            return 1.0
            
        if context == "explicit_pause":
            word_count = len(following_text.split())
            
            if word_count == 1:
                return 3.0
            elif word_count <= 3:
                return 4.5
            elif word_count <= 6:
                return 6.0
            else:
                return 8.0
                
        return 1.0

    def generate_ssml(self, speaker: str, text: str) -> Tuple[str, Optional[str]]:
        """Generate SSML markup - minimal for narrator, raw for Hungarian speakers
        
        Returns:
            Tuple of (ssml_text, language_hint)
            language_hint is 'hu' for Hungarian speakers, None for English narrator
        """
        if speaker in ["Balasz", "Aggie"]:
            text = self.apply_pronunciation_fixes(text)
            return text, "hu"  # Hungarian language hint
        
        if speaker == "NARRATOR":
            ssml = '<speak>'
            
            if text.strip().endswith('?'):
                text = f'<prosody pitch="medium">{text}</prosody>'
            
            ssml += f'<prosody rate="0.95">{text}</prosody>'
            ssml += '</speak>'
            return ssml, None  # English (default)
        
        return text, None

    def standardize_script_formatting(self, file_content: str) -> str:
        """Clean only formatting issues that break parsing, preserve text integrity"""
        
        # Normalize quote variants (curly to straight)
        quote_variants = ['"', '"', '"', '„', '‟']
        for variant in quote_variants:
            file_content = file_content.replace(variant, '"')
        
        # Remove invisible formatting characters
        file_content = file_content.replace('\ufeff', '')  # BOM
        file_content = file_content.replace('\u200b', '')  # Zero-width space
        file_content = file_content.replace('\xa0', ' ')   # Non-breaking space
        
        # Normalize line endings
        file_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
        
        return file_content

    def parse_lesson_script(self, file_content: str) -> List[Dict]:
        """Parse lesson CSV content with manual parsing to handle nested quotes and commas"""
        lines = []
        
        try:
            # Diagnostic logging
            logger.info(f"RAW INPUT (first 300 chars): {repr(file_content[:300])}")
            
            # Standardize formatting before parsing
            file_content = self.standardize_script_formatting(file_content)
            
            # Split into lines and process manually
            content_lines = file_content.strip().split('\n')
            
            # Skip header line if present
            start_index = 0
            if content_lines[0].lower().startswith('speaker'):
                start_index = 1
            
            for i in range(start_index, len(content_lines)):
                line = content_lines[i].strip()
                if not line:
                    continue
                
                # Manual parsing: find first comma that's not inside quotes
                in_quotes = False
                comma_index = -1
                
                for j, char in enumerate(line):
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        comma_index = j
                        break
                
                if comma_index == -1:
                    logger.warning(f"Skipping malformed line (no comma): {line[:50]}")
                    continue
                
                speaker_part = line[:comma_index].strip()
                text_part = line[comma_index + 1:].strip()
                
                # Remove quotes from text_part if present
                if text_part.startswith('"') and text_part.endswith('"'):
                    text_part = text_part[1:-1]
                
                if speaker_part and text_part:
                    lines.append({
                        "speaker": speaker_part,
                        "text": text_part
                    })
                else:
                    logger.warning(f"Skipping line with empty speaker or text: {line[:50]}")
            
            logger.info(f"Successfully parsed {len(lines)} lines from script")
            return lines
            
        except Exception as e:
            logger.error(f"Failed to parse lesson script: {e}")
            raise Exception(f"Script parsing error: {str(e)}")

    def generate_audio_segment(self, speaker: str, text: str, language_hint: Optional[str] = None) -> bytes:
        """Generate audio for a single text segment using ElevenLabs API
        
        Args:
            speaker: Speaker name (NARRATOR, Balasz, Aggie)
            text: Text to convert to speech
            language_hint: 'hu' for Hungarian, None for English
        """
        try:
            voice_config = VOICE_CONFIG.get(speaker)
            if not voice_config:
                raise Exception(f"Unknown speaker: {speaker}")
            
            # Select model based on language
            if language_hint == "hu":
                model_id = "eleven_multilingual_v2"  # Better for Hungarian
                logger.info(f"Using multilingual model for Hungarian speaker: {speaker}")
            else:
                model_id = "eleven_turbo_v2_5"  # Faster for English narrator
                logger.info(f"Using turbo model for English narrator")
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            
            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": voice_config['stability'],
                    "similarity_boost": voice_config['similarity_boost'],
                    "style": voice_config.get('style', 0),
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate audio for '{text[:30]}...': {e}")
            raise

    def create_silence(self, duration_seconds: float) -> AudioSegment:
        """Generate silence of specified duration"""
        return AudioSegment.silent(duration=int(duration_seconds * 1000))

    def process_lesson_with_timing(
        self, 
        lesson_number: int, 
        title: str, 
        description: str, 
        csv_content: str
    ) -> Dict:
        """Process lesson with line-by-line timing tracking AND word-level timing extraction
        
        NEW: Now extracts word-level timestamps from audio using Google Cloud Speech-to-Text
        during audio generation, eliminating need for separate transcript generation endpoint.
        
        Returns dict with:
            - audio_data: Combined MP3 bytes
            - line_timings: List of {speaker, text, start_time, end_time}
            - word_timings: List of {word_text, start_time, end_time, speaker}  # NEW
            - duration_seconds: Total audio duration
            - lesson_metadata: Dict with lesson info
        """
        try:
            logger.info(f"Processing lesson {lesson_number}: {title}")
            
            # Parse script
            script_lines = self.parse_lesson_script(csv_content)
            if not script_lines:
                raise Exception("No valid script lines found")
            
            logger.info(f"Parsed {len(script_lines)} script lines")
            
            # Generate audio segments with timing
            audio_segments = []
            line_timings = []
            current_time = 0.0
            
            logger.info("Generating audio segments with pedagogical pauses...")
            
            for i, line in enumerate(script_lines):
                speaker = line['speaker']
                text = line['text']
                
                # Determine pause duration
                pause_context = "dialogue"
                pause_duration = 1.0
                
                if speaker == "NARRATOR":
                    pause_context = self.detect_pause_context(text)
                    
                    if pause_context == "explicit_pause" and i + 1 < len(script_lines):
                        next_text = script_lines[i + 1]['text']
                        pause_duration = self.calculate_pedagogical_pause(next_text, pause_context)
                
                # Generate SSML with language hint
                ssml_text, language_hint = self.generate_ssml(speaker, text)
                
                # Generate audio
                audio_bytes = self.generate_audio_segment(speaker, ssml_text, language_hint)
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                
                # Record timing for this line
                line_start = current_time
                line_end = current_time + (len(audio_segment) / 1000.0)
                
                line_timings.append({
                    "speaker": speaker,
                    "text": text,
                    "start_time": line_start,
                    "end_time": line_end
                })
                
                logger.info(f"Line {i+1}/{len(script_lines)}: {speaker} | {line_start:.2f}s-{line_end:.2f}s | Pause: {pause_duration}s ({pause_context})")
                
                # Add audio and pause
                audio_segments.append(audio_segment)
                current_time = line_end
                
                if i < len(script_lines) - 1:
                    pause = self.create_silence(pause_duration)
                    audio_segments.append(pause)
                    current_time += pause_duration
            
            # Combine all segments
            logger.info("Combining audio segments...")
            combined_audio = sum(audio_segments[1:], audio_segments[0])
            
            # Export to MP3
            logger.info("Exporting to MP3...")
            output_buffer = io.BytesIO()
            combined_audio.export(
                output_buffer,
                format="mp3",
                bitrate="128k",
                parameters=["-q:a", "2"]
            )
            audio_data = output_buffer.getvalue()
            
            duration_seconds = len(combined_audio) / 1000.0
            logger.info(f"Total audio duration: {duration_seconds:.2f} seconds")
            logger.info(f"Audio size: {len(audio_data) / 1024:.1f} KB")
            
            # NEW: Generate word-level timestamps using Google Cloud Speech-to-Text
            word_timings = []
            if google_speech_client and google_storage_client:
                try:
                    logger.info("Generating word-level timestamps using Google Cloud Speech-to-Text...")
                    word_timings = self.extract_word_timings_from_audio(
                        audio_data, 
                        line_timings
                    )
                    logger.info(f"Successfully extracted {len(word_timings)} word-level timestamps")
                except Exception as e:
                    logger.warning(f"Failed to generate word-level timestamps: {e}")
                    logger.warning("Continuing without word-level timestamps")
            else:
                logger.info("Google Cloud not configured - skipping word-level timestamp generation")
            
            return {
                'audio_data': audio_data,
                'line_timings': line_timings,
                'word_timings': word_timings,  # NEW
                'duration_seconds': duration_seconds,
                'lesson_metadata': {
                    'lesson_number': lesson_number,
                    'title': title,
                    'description': description,
                    'level': 'beginner',
                    'line_count': len(line_timings),
                    'word_count': len(word_timings)  # NEW
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process lesson: {e}")
            raise

    def extract_word_timings_from_audio(self, audio_data: bytes, line_timings: List[Dict]) -> List[Dict]:
        """Extract word-level timestamps from audio using Google Cloud Speech-to-Text
        
        Args:
            audio_data: MP3 audio bytes
            line_timings: List of line timings for speaker attribution
            
        Returns:
            List of {word_text, start_time, end_time, speaker}
        """
        try:
            # Upload audio to GCS temporarily
            logger.info("Uploading audio to Google Cloud Storage...")
            bucket = google_storage_client.bucket(GOOGLE_CLOUD_STORAGE_BUCKET)
            temp_filename = f"temp_audio_{uuid.uuid4()}.mp3"
            blob = bucket.blob(temp_filename)
            blob.upload_from_string(audio_data, content_type='audio/mpeg')
            
            gcs_uri = f"gs://{GOOGLE_CLOUD_STORAGE_BUCKET}/{temp_filename}"
            logger.info(f"Audio uploaded to: {gcs_uri}")
            
            # Configure Speech-to-Text
            audio = speech.RecognitionAudio(uri=gcs_uri)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                sample_rate_hertz=48000,
                language_code="hu-HU",
                alternative_language_codes=["en-US"],
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model="latest_long",
                diarization_config=speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,
                    max_speaker_count=3,
                )
            )
            
            logger.info("Processing audio with Google Cloud Speech-to-Text...")
            logger.info("This may take 3-5 minutes...")
            
            # Process audio
            operation = google_speech_client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=600)
            
            # Clean up temporary file
            blob.delete()
            logger.info("Temporary audio deleted from GCS")
            
            # Extract word timings with speaker attribution
            word_timings = []
            word_index = 0
            
            for result in response.results:
                alternative = result.alternatives[0]
                
                for word_info in alternative.words:
                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()
                    
                    # Fix identical timestamps
                    if start_time == end_time:
                        end_time = start_time + 0.01
                    
                    # Attribute speaker based on line timings
                    speaker = self.get_speaker_for_word(start_time, line_timings)
                    
                    word_timings.append({
                        "word_index": word_index,
                        "word_text": word_info.word,
                        "start_time": start_time,
                        "end_time": end_time,
                        "speaker": speaker
                    })
                    word_index += 1
            
            logger.info(f"Extracted {len(word_timings)} word-level timestamps")
            return word_timings
            
        except Exception as e:
            logger.error(f"Failed to extract word timings: {e}")
            # Don't fail the entire lesson processing if word timing extraction fails
            return []

    def get_speaker_for_word(self, word_time: float, line_timings: List[Dict]) -> str:
        """Determine which speaker is speaking at a given time based on line timings"""
        for line in line_timings:
            if line['start_time'] <= word_time <= line['end_time']:
                return line['speaker']
        return "Unknown"

    def upload_to_supabase_storage(self, file_path: str, file_data: bytes, content_type: str = "audio/mpeg") -> str:
        """Upload file to Supabase Storage"""
        try:
            logger.info(f"Uploading to Supabase Storage: {file_path}")
            
            response = supabase.storage.from_('lesson-audio').upload(
                file_path,
                file_data,
                file_options={"content-type": content_type}
            )
            
            public_url = supabase.storage.from_('lesson-audio').get_public_url(file_path)
            logger.info(f"File uploaded successfully: {public_url}")
            
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload to Supabase Storage: {e}")
            raise

    def save_lesson_to_database(
        self, 
        lesson_metadata: Dict, 
        audio_url: str,
        line_timings: List[Dict],
        word_timings: List[Dict]  # NEW parameter
    ) -> str:
        """Save lesson metadata and timing information to database
        
        NEW: Now also saves word-level timings to dialogue_word_timings table
        """
        try:
            logger.info("Saving lesson to database...")
            
            # Create lesson record
            lesson_data = {
                'id': str(uuid.uuid4()),
                'lesson_number': lesson_metadata['lesson_number'],
                'title': lesson_metadata['title'],
                'description': lesson_metadata['description'],
                'level': lesson_metadata['level'],
                'audio_url': audio_url,
                'is_published': False,
                'created_by': USER_ID,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Insert lesson
            result = supabase.table('lessons').insert(lesson_data).execute()
            lesson_id = result.data[0]['id']
            
            logger.info(f"Lesson saved with ID: {lesson_id}")
            
            # Save line timings to lesson_transcript table
            logger.info(f"Saving {len(line_timings)} line timings to lesson_transcript table...")
            
            line_records = []
            for line_index, line_timing in enumerate(line_timings):
                line_records.append({
                    'id': str(uuid.uuid4()),
                    'lesson_id': lesson_id,
                    'line_index': line_index,
                    'speaker': line_timing['speaker'],
                    'text': line_timing['text'],
                    'start_time': line_timing['start_time'],
                    'end_time': line_timing['end_time'],
                    'created_at': datetime.now().isoformat()
                })
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(line_records), batch_size):
                batch = line_records[i:i + batch_size]
                supabase.table('lesson_transcript').insert(batch).execute()
                logger.info(f"  ✓ Batch {i//batch_size + 1} saved")
            
            logger.info(f"✓ All {len(line_timings)} line timings saved to lesson_transcript")
            
            # NEW: Save word timings to dialogue_word_timings table
            if word_timings:
                logger.info(f"Saving {len(word_timings)} word timings to dialogue_word_timings table...")
                
                word_records = []
                for word_index, word_timing in enumerate(word_timings):
                    word_records.append({
                        'id': str(uuid.uuid4()),
                        'lesson_id': lesson_id,
                        'word_index': word_index,
                        'word_text': word_timing['word_text'],
                        'start_time': word_timing['start_time'],
                        'end_time': word_timing['end_time'],
                        'speaker': word_timing['speaker'],
                        'created_at': datetime.now().isoformat()
                    })
                
                # Insert in batches
                for i in range(0, len(word_records), batch_size):
                    batch = word_records[i:i + batch_size]
                    supabase.table('dialogue_word_timings').insert(batch).execute()
                    logger.info(f"  ✓ Word batch {i//batch_size + 1} saved")
                
                logger.info(f"✓ All {len(word_timings)} word timings saved to dialogue_word_timings")
            else:
                logger.info("⚠ No word timings to save (Google Cloud may not be configured)")
            
            return lesson_id
            
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            raise

    # ========================================
    # VOCABULARY PROCESSING METHODS (RESTORED)
    # ========================================

    def parse_vocabulary_csv(self, file_content: str) -> List[Dict]:
        """Parse vocabulary CSV file
        
        Expected format:
        hungarian_word,english_translation,difficulty
        asztal,table,beginner
        víz,water,beginner
        """
        try:
            vocabulary = []
            
            # Standardize formatting
            file_content = self.standardize_script_formatting(file_content)
            
            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(file_content))
            
            for row in csv_reader:
                if not row.get('hungarian_word') or not row.get('english_translation'):
                    logger.warning(f"Skipping incomplete vocabulary row: {row}")
                    continue
                
                vocabulary.append({
                    'hungarian_word': row['hungarian_word'].strip(),
                    'english_translation': row['english_translation'].strip(),
                    'difficulty': row.get('difficulty', 'beginner').strip()
                })
            
            logger.info(f"Parsed {len(vocabulary)} vocabulary words")
            return vocabulary
            
        except Exception as e:
            logger.error(f"Failed to parse vocabulary CSV: {e}")
            raise

    def generate_vocabulary_audio(self, hungarian_word: str, word_index: int = 0) -> bytes:
        """Generate audio for a single vocabulary word using Hungarian voice
        
        Args:
            hungarian_word: The Hungarian word to generate audio for
            word_index: Index of word in list (used to alternate voices)
        """
        try:
            # Alternate between Aggie (even) and Balasz (odd)
            speaker = "Aggie" if word_index % 2 == 0 else "Balasz"
            voice_config = VOICE_CONFIG[speaker]
            
            logger.info(f"Using {speaker}'s voice for word #{word_index + 1}: {hungarian_word}")
            
            # Apply pronunciation fixes
            corrected_word = self.apply_pronunciation_fixes(hungarian_word)
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            
            data = {
                "text": corrected_word,
                "model_id": "eleven_turbo_v2_5",  # Faster turbo model
                "language_code": "hu",  # Hungarian language code for proper pronunciation
                "voice_settings": {
                    "stability": voice_config['stability'],
                    "similarity_boost": voice_config['similarity_boost'],
                    "style": voice_config.get('style', 0),
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.status_code}")
            
            logger.info(f"Generated audio for vocabulary word: {hungarian_word} ({speaker})")
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate vocabulary audio for '{hungarian_word}': {e}")
            raise

    def create_vocabulary_filename(self, hungarian_word: str) -> str:
        """Create a clean filename from Hungarian word"""
        # Remove special characters and normalize
        clean_word = unicodedata.normalize('NFKD', hungarian_word)
        clean_word = re.sub(r'[^\w\s-]', '', clean_word).strip().lower()
        clean_word = re.sub(r'[-\s]+', '_', clean_word)
        
        return f"{clean_word}.mp3"

    def get_lesson_id_by_number(self, lesson_number: int) -> Optional[str]:
        """Retrieve lesson ID from database by lesson number"""
        try:
            response = supabase.table('lessons').select('id').eq('lesson_number', lesson_number).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]['id']
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve lesson ID for lesson {lesson_number}: {e}")
            return None

    def update_vocabulary_database(self, lesson_id: str, vocabulary_data: List[Dict]) -> None:
        """Save vocabulary words to database"""
        try:
            logger.info(f"Saving {len(vocabulary_data)} vocabulary words to database...")
            
            vocab_records = []
            for vocab in vocabulary_data:
                vocab_records.append({
                    'id': str(uuid.uuid4()),
                    'lesson_id': lesson_id,
                    'word_hu': vocab['word_hu'],
                    'word_en': vocab['word_en'],
                    'difficulty': vocab['difficulty'],
                    'audio_file_path': vocab['audio_file_path'],
                    'created_at': datetime.now().isoformat()
                })
            
            # Insert all vocabulary words
            supabase.table('lesson_vocabulary').insert(vocab_records).execute()
            
            logger.info(f"Successfully saved {len(vocab_records)} vocabulary words")
            
        except Exception as e:
            logger.error(f"Failed to update vocabulary database: {e}")
            raise

    # ========================================
    # END VOCABULARY METHODS
    # ========================================

    def cleanup_temp_files(self):
        """Clean up temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

processor = AudioProcessor()

@app.route('/webhook/google-drive', methods=['POST'])
def webhook_google_drive():
    """Webhook endpoint for Google Apps Script to trigger lesson or vocabulary processing"""
    try:
        logger.info("="*80)
        logger.info("WEBHOOK CALLED FROM GOOGLE APPS SCRIPT")
        logger.info("="*80)
        
        # Get request data from Google Apps Script
        data = request.get_json()
        file_name = data.get('fileName')
        file_content = data.get('fileContent')
        file_type = data.get('fileType', 'lesson')  # Default to 'lesson' for backwards compatibility
        timestamp = data.get('timestamp')
        
        logger.info(f"Received file: {file_name}")
        logger.info(f"File type: {file_type}")
        logger.info(f"Content length: {len(file_content)} characters")
        logger.info(f"Timestamp: {timestamp}")
        
        # Validation
        if not file_name or not file_content:
            return jsonify({
                "status": "error",
                "message": "Missing fileName or fileContent"
            }), 400
        
        # Route to appropriate processor based on file_type
        if file_type == 'vocabulary':
            return process_vocabulary_file(file_name, file_content)
        else:  # Default to lesson processing
            return process_lesson_file(file_name, file_content)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"WEBHOOK ERROR: {e}")
        logger.error("="*80)
        import traceback
        logger.error(traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "success": False,
            "message": str(e)
        }), 500

def process_lesson_file(file_name: str, file_content: str) -> tuple:
    """Process lesson file from webhook"""
    try:
        # Extract lesson metadata from filename
        lesson_number = None
        title = "Untitled Lesson"
        
        # Try to parse lesson number and title from filename
        lesson_match = re.search(r'lesson\s*(\d+)', file_name, re.IGNORECASE)
        if lesson_match:
            lesson_number = int(lesson_match.group(1))
            logger.info(f"Extracted lesson number: {lesson_number}")
        else:
            logger.warning(f"Could not extract lesson number from filename: {file_name}")
            return jsonify({
                "status": "error",
                "message": f"Could not extract lesson number from filename: {file_name}. Expected format: 'Lesson 4 - Title.txt'"
            }), 400
        
        # Extract title (everything between lesson number and file extension)
        title_match = re.search(r'lesson\s*\d+\s*[-–]\s*(.+?)(?:\s*-\s*FINAL)?\.(?:txt|csv|docx)$', file_name, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            logger.info(f"Extracted title: {title}")
        else:
            # Fallback: use filename without extension
            title = file_name.rsplit('.', 1)[0].replace(f'Lesson {lesson_number}', '').strip(' -–')
            logger.info(f"Using fallback title: {title}")
        
        # Rate limiting
        if processor.daily_usage >= processor.max_daily_lessons:
            return jsonify({
                "status": "error",
                "message": f"Daily lesson limit reached ({processor.max_daily_lessons})"
            }), 429
        
        logger.info(f"Processing lesson {lesson_number}: {title}")
        
        # Process lesson with timing (NOW INCLUDES WORD-LEVEL TIMESTAMPS)
        result = processor.process_lesson_with_timing(
            lesson_number=lesson_number,
            title=title,
            description=f"Processed from {file_name}",
            csv_content=file_content
        )
        
        # Upload audio
        audio_filename = f"lesson_{lesson_number}_{title.replace(' ', '_')}.mp3"
        audio_url = processor.upload_to_supabase_storage(
            audio_filename,
            result['audio_data']
        )
        
        # Save to database with line timings AND word timings (MODIFIED)
        lesson_id = processor.save_lesson_to_database(
            result['lesson_metadata'],
            audio_url,
            result['line_timings'],
            result['word_timings']  # NEW: Pass word timings
        )
        
        # Increment usage counter
        processor.daily_usage += 1
        
        logger.info("="*80)
        logger.info(f"WEBHOOK SUCCESS: Lesson {lesson_number} processed")
        logger.info(f"Lesson ID: {lesson_id}")
        logger.info("="*80)
        
        return jsonify({
            "status": "success",
            "success": True,
            "message": f"Lesson {lesson_number} processed successfully",
            "lesson_id": lesson_id,
            "lesson_number": lesson_number,
            "title": title,
            "audio_url": audio_url,
            "duration_seconds": result['duration_seconds'],
            "line_count": len(result['line_timings']),
            "word_count": len(result['word_timings']),
            "line_timings_saved": True,
            "word_timings_saved": len(result['word_timings']) > 0
        })
        
    except Exception as e:
        logger.error(f"Failed to process lesson file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "success": False,
            "message": str(e)
        }), 500

def process_vocabulary_file(file_name: str, file_content: str) -> tuple:
    """Process vocabulary CSV file from webhook
    
    Expected filename format: "Lesson 4 - Vocabulary.csv"
    CSV format: hungarian_word,english_translation,difficulty
    """
    try:
        logger.info(f"Processing vocabulary file: {file_name}")
        
        # Parse vocabulary CSV
        vocabulary_data = processor.parse_vocabulary_csv(file_content)
        if not vocabulary_data:
            raise Exception("No vocabulary data found in file")
        
        # Extract lesson number from filename
        lesson_match = re.search(r'lesson\s*(\d+)', file_name, re.IGNORECASE)
        if not lesson_match:
            raise Exception(f"Could not extract lesson number from vocabulary filename: {file_name}")
        
        lesson_number = int(lesson_match.group(1))
        logger.info(f"Vocabulary for lesson number: {lesson_number}")
        
        # Get lesson ID from database
        lesson_id = processor.get_lesson_id_by_number(lesson_number)
        if not lesson_id:
            raise Exception(f"No lesson found in database for lesson number {lesson_number}. Please upload the lesson script first.")
        
        logger.info(f"Found lesson ID: {lesson_id}")
        
        # Process each vocabulary word
        processed_vocabulary = []
        for word_index, vocab_item in enumerate(vocabulary_data):
            hungarian_word = vocab_item['hungarian_word']
            
            try:
                # Generate audio for vocabulary word (alternates between Aggie and Balasz)
                audio_data = processor.generate_vocabulary_audio(hungarian_word, word_index)
                
                # Validate audio
                if len(audio_data) < 500:
                    logger.warning(f"Audio validation failed for word: {hungarian_word}")
                    continue
                
                # Create filename
                audio_filename = processor.create_vocabulary_filename(hungarian_word)
                
                # Upload to Supabase Storage
                vocab_uuid = str(uuid.uuid4())
                file_path = f"vocabulary/lesson_{lesson_number}/{vocab_uuid}_{audio_filename}"
                
                processor.upload_to_supabase_storage(file_path, audio_data, "audio/mpeg")
                
                processed_vocabulary.append({
                    "word_hu": vocab_item['hungarian_word'],
                    "word_en": vocab_item['english_translation'],
                    "difficulty": vocab_item['difficulty'],
                    "audio_file_path": file_path
                })
                
                logger.info(f"Processed vocabulary word: {hungarian_word}")
                
            except Exception as e:
                logger.error(f"Failed to process vocabulary word '{hungarian_word}': {e}")
                continue
        
        if not processed_vocabulary:
            raise Exception("No vocabulary words were successfully processed")
        
        # Save to database
        processor.update_vocabulary_database(lesson_id, processed_vocabulary)
        
        logger.info("="*80)
        logger.info(f"VOCABULARY SUCCESS: {len(processed_vocabulary)} words for Lesson {lesson_number}")
        logger.info("="*80)
        
        return jsonify({
            "status": "success",
            "success": True,
            "message": f"Successfully processed {len(processed_vocabulary)} vocabulary words for Lesson {lesson_number}",
            "lesson_id": lesson_id,
            "lesson_number": lesson_number,
            "vocabulary_count": len(processed_vocabulary),
            "words_processed": [v['word_hu'] for v in processed_vocabulary]
        })
        
    except Exception as e:
        logger.error(f"Failed to process vocabulary file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "success": False,
            "message": str(e)
        }), 500

@app.route('/process-lesson', methods=['POST'])
def process_lesson_endpoint():
    """Process a new lesson from CSV script with line timing tracking"""
    try:
        # Rate limiting
        if processor.daily_usage >= processor.max_daily_lessons:
            return jsonify({
                "status": "error",
                "message": f"Daily lesson limit reached ({processor.max_daily_lessons})"
            }), 429
        
        # Get request data
        data = request.get_json()
        lesson_number = data.get('lesson_number')
        title = data.get('title')
        description = data.get('description', '')
        csv_content = data.get('csv_content')
        
        # Validation
        if not all([lesson_number, title, csv_content]):
            return jsonify({
                "status": "error",
                "message": "Missing required fields: lesson_number, title, csv_content"
            }), 400
        
        logger.info(f"Processing lesson {lesson_number}: {title}")
        
        # Process lesson with timing (NOW INCLUDES WORD-LEVEL TIMESTAMPS)
        result = processor.process_lesson_with_timing(
            lesson_number=lesson_number,
            title=title,
            description=description,
            csv_content=csv_content
        )
        
        # Upload audio
        audio_filename = f"lesson_{lesson_number}_{title.replace(' ', '_')}.mp3"
        audio_url = processor.upload_to_supabase_storage(
            audio_filename,
            result['audio_data']
        )
        
        # Save to database with line timings AND word timings (MODIFIED)
        lesson_id = processor.save_lesson_to_database(
            result['lesson_metadata'],
            audio_url,
            result['line_timings'],
            result['word_timings']  # NEW: Pass word timings
        )
        
        # Increment usage counter
        processor.daily_usage += 1
        
        return jsonify({
            "status": "success",
            "message": f"Lesson {lesson_number} processed successfully with word-level timing",
            "lesson_id": lesson_id,
            "audio_url": audio_url,
            "duration_seconds": result['duration_seconds'],
            "line_count": len(result['line_timings']),
            "word_count": len(result['word_timings']),
            "line_timings_saved": True,
            "word_timings_saved": len(result['word_timings']) > 0
        })
        
    except Exception as e:
        logger.error(f"Error processing lesson: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with configuration status"""
    try:
        config_status = {
            "elevenlabs_key_loaded": bool(ELEVENLABS_API_KEY),
            "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
            "user_id_set": bool(USER_ID),
            "daily_usage": processor.daily_usage,
            "max_daily_lessons": processor.max_daily_lessons,
            "google_cloud_configured": bool(GOOGLE_CLOUD_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS_JSON),
            "google_cloud_storage_bucket": GOOGLE_CLOUD_STORAGE_BUCKET,
            "word_level_timestamps_enabled": bool(google_speech_client and google_storage_client),
            "vocabulary_processing_enabled": True  # NEW: Confirm vocabulary support
        }
        
        return jsonify({
            "status": "healthy", 
            "config": config_status,
            "message": "GetTalkin Audio Processor with lesson and vocabulary processing"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/pronunciation', methods=['POST'])
def add_pronunciation_fix():
    """Add new pronunciation correction"""
    try:
        data = request.get_json()
        hungarian_word = data.get('hungarian_word')
        phonetic_spelling = data.get('phonetic_spelling')
        
        if not hungarian_word or not phonetic_spelling:
            return jsonify({"error": "Missing required fields"}), 400
        
        pronunciation_data = {
            "hungarian_word": hungarian_word,
            "phonetic_spelling": phonetic_spelling,
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.table('pronunciation_fixes').insert(pronunciation_data).execute()
        
        processor.pronunciation_cache = {}
        
        return jsonify({"status": "success", "message": "Pronunciation fix added"})
        
    except Exception as e:
        logger.error(f"Failed to add pronunciation fix: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting GetTalkin Audio Processor")
        logger.info("Features: Lesson processing + Vocabulary processing + Word-level timestamps")
        logger.info("All secrets loaded from environment variables")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    finally:
        processor.cleanup_temp_files()
