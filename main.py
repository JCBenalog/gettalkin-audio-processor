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
        Returns: (processed_text, language_code)"""
        if speaker in ["Balasz", "Aggie"]:
            # Hungarian speakers: use Hungarian language code, apply pronunciation fixes
            text = self.apply_pronunciation_fixes(text)
            return text, "hu"
        
        if speaker == "NARRATOR":
            # Narrator: no language code (auto-detection for mixed English/Hungarian)
            ssml = '<speak>'
            
            if text.strip().endswith('?'):
                text = f'<prosody pitch="medium">{text}</prosody>'
            
            ssml += f'<prosody rate="0.95">{text}</prosody>'
            ssml += '</speak>'
            return ssml, None
        
        return text, None

    def generate_audio_segment(self, speaker: str, text: str, use_ssml: bool = True) -> Tuple[bytes, float]:
        """Generate audio for a single text segment using ElevenLabs API
        Returns: (audio_bytes, duration_seconds)"""
        voice_config = VOICE_CONFIG.get(speaker, VOICE_CONFIG["NARRATOR"])
        
        processed_text, language_code = self.generate_ssml(speaker, text) if use_ssml else (text, None)
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        payload = {
            "text": processed_text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": voice_config["stability"],
                "similarity_boost": voice_config["similarity_boost"],
                "style": voice_config.get("style", 0.15)
            }
        }
        
        if language_code:
            payload["language_code"] = language_code
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            audio_bytes = response.content
            
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            duration_seconds = len(audio_segment) / 1000.0
            
            logger.info(f"Generated {speaker} audio: {len(audio_bytes)} bytes, {duration_seconds:.2f}s")
            return audio_bytes, duration_seconds
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ElevenLabs API error: {e}")
            raise

    def get_word_timings_from_audio(self, audio_bytes: bytes, speaker: str, original_text: str) -> List[Dict]:
        """
        NEW METHOD: Get word-level timestamps from audio using Google Speech-to-Text
        Returns list of {word_text, start_time, end_time} relative to this audio segment
        """
        if not google_speech_client or not google_storage_client:
            logger.debug("Google Cloud not available - skipping word-level timestamps for this line")
            return []
        
        try:
            # Upload audio to temporary GCS location
            bucket = google_storage_client.bucket(GOOGLE_CLOUD_STORAGE_BUCKET)
            temp_filename = f"temp_line_{uuid.uuid4().hex}.mp3"
            blob = bucket.blob(temp_filename)
            blob.upload_from_string(audio_bytes, content_type='audio/mpeg')
            
            gcs_uri = f"gs://{GOOGLE_CLOUD_STORAGE_BUCKET}/{temp_filename}"
            
            # Configure Speech-to-Text
            audio = speech.RecognitionAudio(uri=gcs_uri)
            
            # Determine language for this speaker
            language_code = "hu-HU" if speaker in ["Balasz", "Aggie"] else "en-US"
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                sample_rate_hertz=48000,
                language_code=language_code,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model="latest_short"  # Use short model for individual lines
            )
            
            # Recognize speech (synchronous for short audio)
            response = google_speech_client.recognize(config=config, audio=audio)
            
            # Clean up temporary file
            blob.delete()
            
            # Extract word timings
            word_timings = []
            for result in response.results:
                alternative = result.alternatives[0]
                for word_info in alternative.words:
                    start_time = word_info.start_time.total_seconds()
                    end_time = word_info.end_time.total_seconds()
                    
                    # Fix identical timestamps (Google bug)
                    if start_time == end_time:
                        end_time = start_time + 0.01
                    
                    word_timings.append({
                        "word_text": word_info.word,
                        "start_time": start_time,
                        "end_time": end_time
                    })
            
            logger.info(f"  ✓ Extracted {len(word_timings)} word timings for line")
            return word_timings
            
        except Exception as e:
            logger.warning(f"Failed to get word timings for line: {e}")
            return []

    def create_silence(self, duration_seconds: float) -> bytes:
        """Generate silence audio segment"""
        silence = AudioSegment.silent(duration=int(duration_seconds * 1000))
        buffer = io.BytesIO()
        silence.export(buffer, format="mp3", bitrate="128k")
        return buffer.getvalue()

    def standardize_script_formatting(self, csv_content: str) -> str:
        """Standardize script formatting to prevent parsing errors"""
        csv_content = csv_content.replace('"', '"').replace('"', '"')
        csv_content = csv_content.replace(''', "'").replace(''', "'")
        csv_content = ''.join(char for char in csv_content if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
        return csv_content

    def parse_lesson_script(self, csv_content: str) -> List[Dict]:
        """Parse lesson script CSV with robust error handling"""
        csv_content = self.standardize_script_formatting(csv_content)
        
        lines_data = []
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        for row_num, row in enumerate(csv_reader, start=2):
            try:
                speaker = row.get('speaker', '').strip()
                line_text = row.get('line', '').strip()
                
                if not speaker or not line_text:
                    logger.warning(f"Row {row_num}: Missing speaker or line text")
                    continue
                
                lines_data.append({
                    'speaker': speaker,
                    'line': line_text,
                    'row_num': row_num
                })
                
            except Exception as e:
                logger.error(f"Error parsing row {row_num}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(lines_data)} dialogue lines")
        return lines_data

    def process_lesson_with_timing(self, lesson_number: int, title: str, description: str, csv_content: str) -> Dict:
        """
        MODIFIED METHOD: Process lesson and track BOTH line timing AND word timing
        Returns: Dict with audio_data, lesson_metadata, line_timings, and word_timings
        """
        
        logger.info("="*80)
        logger.info(f"PROCESSING LESSON {lesson_number}: {title}")
        logger.info("WITH WORD-LEVEL TIMESTAMP GENERATION" if google_speech_client else "WITHOUT WORD-LEVEL TIMESTAMPS (Google Cloud not configured)")
        logger.info("="*80)
        
        # Parse script
        lines_data = self.parse_lesson_script(csv_content)
        if not lines_data:
            raise ValueError("No valid dialogue lines found in script")
        
        logger.info(f"Processing {len(lines_data)} dialogue lines with timing tracking...")
        
        # Initialize timing tracking
        cumulative_time = 0.0
        line_timings = []
        all_word_timings = []  # NEW: Store word timings across all lines
        audio_segments = []
        
        # Process each line and track timing
        for i, line_data in enumerate(lines_data):
            speaker = line_data['speaker']
            line_text = line_data['line']
            
            # Calculate start time (before generating audio)
            line_start_time = cumulative_time
            
            logger.info(f"\n[Line {i+1}/{len(lines_data)}] {speaker}: {line_text[:50]}...")
            
            # Generate audio segment
            audio_bytes, duration = self.generate_audio_segment(speaker, line_text)
            audio_segments.append(audio_bytes)
            
            # NEW: Get word-level timestamps for this line (if Google Cloud available)
            word_timings = self.get_word_timings_from_audio(audio_bytes, speaker, line_text)
            
            # NEW: Adjust word timestamps by cumulative offset and add speaker/line info
            for word_timing in word_timings:
                all_word_timings.append({
                    "word_text": word_timing["word_text"],
                    "start_time": cumulative_time + word_timing["start_time"],
                    "end_time": cumulative_time + word_timing["end_time"],
                    "speaker": speaker,
                    "line_index": i + 1
                })
            
            # Update cumulative time with actual audio duration
            cumulative_time += duration
            line_end_time = cumulative_time
            
            # Calculate pause (if needed)
            pause_duration = 1.0  # Default
            if speaker == "NARRATOR":
                context = self.detect_pause_context(line_text)
                if context == "explicit_pause" and i + 1 < len(lines_data):
                    following_text = lines_data[i + 1]['line']
                    pause_duration = self.calculate_pedagogical_pause(following_text, context)
                    logger.info(f"  └─ Pedagogical pause: {pause_duration}s (following text: {len(following_text.split())} words)")
            
            # Add pause
            if pause_duration > 0:
                silence = self.create_silence(pause_duration)
                audio_segments.append(silence)
                cumulative_time += pause_duration
            
            # Save line timing
            line_timings.append({
                'speaker': speaker,
                'line_text': line_text,
                'start_time': line_start_time,
                'end_time': line_end_time
            })
            
            logger.info(f"  ✓ Audio: {duration:.2f}s | Start: {line_start_time:.2f}s | End: {line_end_time:.2f}s | Pause: {pause_duration:.2f}s")
        
        # Combine all audio segments
        logger.info("\n" + "="*80)
        logger.info("COMBINING AUDIO SEGMENTS...")
        logger.info("="*80)
        
        combined_audio = AudioSegment.empty()
        for audio_bytes in audio_segments:
            segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            combined_audio += segment
        
        # Export final audio
        buffer = io.BytesIO()
        combined_audio.export(buffer, format="mp3", bitrate="128k")
        final_audio_data = buffer.getvalue()
        
        final_duration = len(combined_audio) / 1000.0
        
        logger.info("="*80)
        logger.info(f"LESSON COMPLETE: {len(lines_data)} lines processed")
        logger.info(f"Total duration: {final_duration:.2f}s ({final_duration/60:.1f} minutes)")
        logger.info(f"Line timings tracked: {len(line_timings)} entries")
        logger.info(f"Word timings tracked: {len(all_word_timings)} words")  # NEW
        logger.info("="*80)
        
        return {
            'audio_data': final_audio_data,
            'duration_seconds': final_duration,
            'line_timings': line_timings,
            'word_timings': all_word_timings,  # NEW: Return word timings
            'lesson_metadata': {
                'lesson_number': lesson_number,
                'title': title,
                'description': description,
                'line_count': len(lines_data),
                'duration_seconds': final_duration
            }
        }

    def upload_to_supabase_storage(self, file_path: str, audio_data: bytes, content_type: str = "audio/mpeg") -> str:
        """Upload audio file to Supabase Storage"""
        try:
            bucket_name = "lesson-audio"
            
            response = supabase.storage.from_(bucket_name).upload(
                file_path,
                audio_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                }
            )
            
            public_url = supabase.storage.from_(bucket_name).get_public_url(file_path)
            logger.info(f"Audio uploaded successfully: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload to Supabase Storage: {e}")
            raise

    def save_lesson_to_database(self, lesson_data: Dict, audio_url: str, line_timings: List[Dict], word_timings: List[Dict]) -> str:
        """
        MODIFIED METHOD: Save lesson metadata, line timings, AND word timings to Supabase
        Returns: lesson_id
        """
        try:
            # Create lesson record
            lesson_record = {
                "id": str(uuid.uuid4()),
                "lesson_number": lesson_data['lesson_number'],
                "title": lesson_data['title'],
                "description": lesson_data['description'],
                "audio_url": audio_url,
                "duration": int(lesson_data['duration_seconds']),
                "level": "beginner",
                "created_by": USER_ID,
                "is_published": False,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info("Saving lesson to database...")
            lesson_response = supabase.table('lessons').insert(lesson_record).execute()
            lesson_id = lesson_response.data[0]['id']
            logger.info(f"✓ Lesson saved with ID: {lesson_id}")
            
            # Save line timings to lesson_transcript table
            logger.info(f"Saving {len(line_timings)} line timings to lesson_transcript table...")
            
            transcript_records = []
            for i, timing in enumerate(line_timings):
                transcript_records.append({
                    'id': str(uuid.uuid4()),
                    'lesson_id': lesson_id,
                    'speaker': timing['speaker'],
                    'text_content': timing['line_text'],
                    'line_number': i + 1,
                    'start_time': timing['start_time'],
                    'end_time': timing['end_time'],
                    'created_at': datetime.now().isoformat()
                })
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(transcript_records), batch_size):
                batch = transcript_records[i:i + batch_size]
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
    """Webhook endpoint for Google Apps Script to trigger lesson processing"""
    try:
        logger.info("="*80)
        logger.info("WEBHOOK CALLED FROM GOOGLE APPS SCRIPT")
        logger.info("="*80)
        
        # Get request data from Google Apps Script
        data = request.get_json()
        file_name = data.get('fileName')
        file_content = data.get('fileContent')
        file_type = data.get('fileType', 'lesson')
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
        
        # Extract lesson metadata from filename
        # Expected format: "Lesson 4 - in the restaurant - FINAL.txt"
        lesson_number = None
        title = "Untitled Lesson"
        
        # Try to parse lesson number and title from filename
        import re
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
            "word_count": len(result['word_timings']),  # NEW
            "line_timings_saved": True,
            "word_timings_saved": len(result['word_timings']) > 0  # NEW
        })
        
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
            "word_count": len(result['word_timings']),  # NEW
            "line_timings_saved": True,
            "word_timings_saved": len(result['word_timings']) > 0  # NEW
        })
        
    except Exception as e:
        logger.error(f"Error processing lesson: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def map_google_speakers_to_lesson_speakers(lesson_id: str, word_timings: List[Dict]) -> List[Dict]:
    """Map Google's speaker tags to actual lesson speakers using lesson_transcript timing
    
    Strategy:
    1. Get lesson_transcript data with speaker labels and time ranges
    2. For each word, find which lesson_transcript segment it falls into
    3. Assign that segment's speaker to the word
    
    Args:
        lesson_id: UUID of the lesson
        word_timings: List of word timing dicts with speaker_tag from Google
        
    Returns:
        Updated word_timings list with 'speaker' field added
    """
    try:
        # Fetch lesson transcript with speaker info
        transcript_response = supabase.table('lesson_transcript').select('*').eq('lesson_id', lesson_id).order('start_time').execute()
        transcript_segments = transcript_response.data
        
        if not transcript_segments:
            logger.warning("No transcript segments found for mapping - using Google speaker tags")
            # Fallback: Use Google's speaker tags
            for word in word_timings:
                word['speaker'] = f"Speaker {word.get('speaker_tag', 'Unknown')}"
            return word_timings
        
        logger.info(f"Mapping {len(word_timings)} words to {len(transcript_segments)} transcript segments")
        
        # Map each word to a transcript segment based on timing overlap
        for word in word_timings:
            word_start = word['start_time']
            word_end = word['end_time']
            word_mid = (word_start + word_end) / 2  # Use midpoint for matching
            
            # Find best matching segment
            best_match = None
            best_overlap = 0
            
            for segment in transcript_segments:
                seg_start = float(segment['start_time'])
                seg_end = float(segment['end_time'])
                
                # Check if word midpoint falls within segment
                if seg_start <= word_mid <= seg_end:
                    best_match = segment
                    break
                
                # Calculate overlap as fallback
                overlap_start = max(word_start, seg_start)
                overlap_end = min(word_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = segment
            
            # Assign speaker from best matching segment
            if best_match:
                word['speaker'] = best_match['speaker']
            else:
                word['speaker'] = f"Speaker {word.get('speaker_tag', 'Unknown')}"
                logger.debug(f"No match for word '{word['word_text']}' at {word_start:.2f}s")
        
        # Log speaker distribution
        speaker_counts = {}
        for word in word_timings:
            speaker = word['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        logger.info("Speaker mapping complete:")
        for speaker, count in speaker_counts.items():
            logger.info(f"  {speaker}: {count} words")
        
        return word_timings
        
    except Exception as e:
        logger.error(f"Error mapping speakers: {e}")
        # Fallback: Use Google speaker tags
        for word in word_timings:
            word['speaker'] = f"Speaker {word.get('speaker_tag', 'Unknown')}"
        return word_timings

@app.route('/generate-transcript', methods=['POST'])
def generate_transcript():
    """
    LEGACY ENDPOINT: Generate word-level transcript from FULL lesson audio using Google Speech-to-Text
    
    NOTE: This endpoint is now DEPRECATED in favor of inline word-level timestamp generation
    during audio processing (which provides better speaker labeling and doesn't create runon paragraphs).
    
    This endpoint is kept for backwards compatibility and can still be used to regenerate
    transcripts for lessons that were processed before word-level timestamps were added.
    """
    
    # Check if Google Cloud is configured
    if not GOOGLE_CLOUD_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "Google Cloud libraries not installed. Install with: pip install google-cloud-speech google-cloud-storage"
        }), 500
    
    if not GOOGLE_CLOUD_PROJECT_ID or not GOOGLE_APPLICATION_CREDENTIALS_JSON:
        return jsonify({
            "status": "error",
            "message": "Google Cloud credentials not configured. Set GOOGLE_CLOUD_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS_JSON environment variables."
        }), 500
    
    try:
        # Get lesson_id from request
        data = request.get_json()
        lesson_id = data.get('lesson_id')
        
        if not lesson_id:
            return jsonify({"status": "error", "message": "lesson_id is required"}), 400
        
        # Fetch lesson data
        logger.info(f"Fetching lesson data for: {lesson_id}")
        lesson_response = supabase.table('lessons').select('*').eq('id', lesson_id).execute()
        
        if not lesson_response.data:
            return jsonify({"status": "error", "message": "Lesson not found"}), 404
        
        lesson = lesson_response.data[0]
        lesson_title = lesson.get('title', 'Unknown')
        audio_file_path = lesson.get('audio_file_path')
        
        if not audio_file_path:
            return jsonify({"status": "error", "message": "Lesson has no audio file"}), 400
        
        logger.info(f"Processing lesson: {lesson_title}")
        logger.info(f"Audio URL: {audio_file_path}")
        
        # Download audio from Supabase Storage
        logger.info("Downloading audio from Supabase Storage...")
        audio_response = requests.get(audio_file_path)
        audio_response.raise_for_status()
        audio_content = audio_response.content
        logger.info(f"Audio downloaded: {len(audio_content)} bytes")
        
        # Initialize Google Cloud credentials
        logger.info("Initializing Google Cloud credentials...")
        credentials_json = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_json)
        
        # Upload audio to Google Cloud Storage (required for long audio files)
        logger.info("Uploading audio to Google Cloud Storage...")
        storage_client = storage.Client(credentials=credentials, project=GOOGLE_CLOUD_PROJECT_ID)
        bucket = storage_client.bucket(GOOGLE_CLOUD_STORAGE_BUCKET)
        
        # Create unique filename for temporary storage
        temp_filename = f"temp_audio_{lesson_id}_{datetime.now().timestamp()}.mp3"
        blob = bucket.blob(temp_filename)
        
        logger.info(f"Uploading audio to GCS bucket: {GOOGLE_CLOUD_STORAGE_BUCKET}/{temp_filename}")
        blob.upload_from_string(audio_content, content_type='audio/mpeg')
        
        # Construct GCS URI
        gcs_uri = f"gs://{GOOGLE_CLOUD_STORAGE_BUCKET}/{temp_filename}"
        logger.info(f"Audio uploaded to: {gcs_uri}")
        
        # Initialize Speech-to-Text client
        speech_client = speech.SpeechClient(credentials=credentials)
        
        # Prepare audio for Google API using GCS URI
        audio = speech.RecognitionAudio(uri=gcs_uri)
        
        # Configure recognition with speaker diarization
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=48000,
            language_code="hu-HU",  # Primary language
            alternative_language_codes=["en-US"],  # For narrator's English
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model="latest_long",
            # Enable speaker diarization (this is the key addition)
            diarization_config=speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=3,
            )
        )
        
        logger.info("Sending audio to Google Cloud Speech-to-Text API...")
        logger.info("Configuration: Multi-language + Speaker diarization enabled")
        logger.info("This may take 3-5 minutes for a typical lesson...")
        
        # Use long_running_recognize for files longer than 1 minute
        operation = speech_client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=600)
        
        # Delete temporary audio file from GCS
        logger.info("Deleting temporary audio from GCS...")
        blob.delete()
        logger.info("Temporary audio deleted")
        
        # Extract word-level timestamps with speaker tags and language codes
        word_timings = []
        word_index = 0
        
        for result in response.results:
            alternative = result.alternatives[0]
            
            # Extract detected language for this result
            detected_language = getattr(result, 'language_code', 'unknown')
            
            for word_info in alternative.words:
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                
                # FIX: Identical timestamps (Google bug for very short words)
                if start_time == end_time:
                    end_time = start_time + 0.01
                    logger.debug(f"Fixed identical timestamp for word: {word_info.word}")
                
                # Extract speaker tag from Google (if diarization enabled)
                speaker_tag = getattr(word_info, 'speaker_tag', None)
                
                word_timings.append({
                    "word_index": word_index,
                    "word_text": word_info.word,
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker_tag": speaker_tag,
                    "language_code": detected_language
                })
                word_index += 1
        
        logger.info(f"Successfully extracted {len(word_timings)} words from Google API")
        
        # Map Google speakers to lesson speakers using lesson_transcript
        logger.info("Mapping Google speaker tags to lesson speakers...")
        word_timings = map_google_speakers_to_lesson_speakers(lesson_id, word_timings)
        
        # Save to database with speaker information
        logger.info("Saving transcript with speaker labels to dialogue_word_timings table...")
        
        # Delete existing transcript for this lesson (if re-generating)
        try:
            supabase.table('dialogue_word_timings').delete().eq('lesson_id', lesson_id).execute()
            logger.info("Cleared existing transcript data")
        except Exception as e:
            logger.warning(f"No existing transcript to clear: {e}")
        
        # Prepare records for batch insert
        db_records = []
        for word_timing in word_timings:
            db_records.append({
                'id': str(uuid.uuid4()),
                'lesson_id': lesson_id,
                'word_index': word_timing['word_index'],
                'word_text': word_timing['word_text'],
                'start_time': word_timing['start_time'],
                'end_time': word_timing['end_time'],
                'speaker': word_timing.get('speaker', 'Unknown'),
                'created_at': datetime.now().isoformat()
            })
        
        # Insert in batches (Supabase has limits on batch size)
        batch_size = 100
        for i in range(0, len(db_records), batch_size):
            batch = db_records[i:i + batch_size]
            supabase.table('dialogue_word_timings').insert(batch).execute()
            logger.info(f"Inserted batch {i//batch_size + 1}/{(len(db_records) + batch_size - 1)//batch_size}")
        
        logger.info(f"Successfully saved {len(word_timings)} words with speaker labels to database")
        
        return jsonify({
            "status": "success",
            "message": f"Generated and saved transcript with {len(word_timings)} words",
            "word_count": len(word_timings),
            "lesson_id": lesson_id,
            "lesson_title": lesson_title,
            "saved_to_database": True,
            "note": "This endpoint is deprecated. New lessons automatically generate word-level timestamps during audio processing."
        })
        
    except Exception as e:
        logger.error(f"Failed to generate transcript: {e}")
        
        # Attempt to clean up temporary file if it exists
        try:
            if 'blob' in locals() and blob.exists():
                blob.delete()
                logger.info("Cleaned up temporary audio file after error")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
        
        return jsonify({"status": "error", "message": str(e)}), 500

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
            "word_level_timestamps_enabled": bool(google_speech_client and google_storage_client)
        }
        
        return jsonify({
            "status": "healthy", 
            "config": config_status,
            "message": "GetTalkin Audio Processor with inline word-level timestamp generation"
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
        logger.info("Starting GetTalkin Audio Processor with inline word-level timestamp generation")
        logger.info("All secrets loaded from environment variables")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    finally:
        processor.cleanup_temp_files()
