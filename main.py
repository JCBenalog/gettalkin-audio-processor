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
        logger.warning("Google Cloud credentials not set - transcript generation will be disabled")
    else:
        logger.info("Google Cloud credentials loaded - transcript generation enabled")
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

    def standardize_script_formatting(self, file_content: str) -> str:
        """Standardize quote characters and remove invisible formatting"""
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
                
                if not line or line.startswith('#'):
                    continue
                
                try:
                    if ',' in line:
                        comma_pos = line.index(',')
                        speaker = line[:comma_pos].strip().strip('"')
                        text = line[comma_pos+1:].strip().strip('"')
                    else:
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            speaker, text = parts
                            speaker = speaker.strip('"')
                            text = text.strip('"')
                        else:
                            logger.warning(f"Line {i}: Could not parse '{line}'")
                            continue
                    
                    if speaker and text:
                        lines.append({
                            'speaker': speaker,
                            'text': text
                        })
                        logger.info(f"Line {i}: {speaker}: {text[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"Line {i}: Parse error - {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(lines)} lines")
            return lines
            
        except Exception as e:
            logger.error(f"Failed to parse lesson script: {e}")
            return []

    def parse_vocabulary_csv(self, file_content: str) -> List[Dict]:
        """Parse vocabulary CSV content"""
        try:
            file_content = self.standardize_script_formatting(file_content)
            
            vocabulary_data = []
            csv_reader = csv.DictReader(io.StringIO(file_content))
            
            for row in csv_reader:
                vocabulary_data.append({
                    'hungarian_word': row.get('word_hu', row.get('hungarian_word', '')),
                    'english_translation': row.get('word_en', row.get('english_translation', '')),
                    'difficulty': row.get('difficulty', 'beginner')
                })
            
            logger.info(f"Successfully parsed {len(vocabulary_data)} vocabulary words")
            return vocabulary_data
            
        except Exception as e:
            logger.error(f"Failed to parse vocabulary CSV: {e}")
            return []

    def create_vocabulary_filename(self, hungarian_word: str) -> str:
        """Create safe filename from Hungarian word"""
        safe_word = re.sub(r'[^\w\s-]', '', hungarian_word.lower())
        safe_word = re.sub(r'[-\s]+', '-', safe_word)
        return f"{safe_word}.wav"

    def generate_audio_segment(self, speaker: str, text: str, voice_config: dict) -> bytes:
        """Generate audio for a single text segment using ElevenLabs"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}"
        
        processed_text, language_code = self.generate_ssml(speaker, text)
        
        payload = {
            "text": processed_text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": voice_config['stability'],
                "similarity_boost": voice_config['similarity_boost'],
                "style": voice_config.get('style', 0.0),
                "use_speaker_boost": True
            }
        }
        
        if language_code:
            payload["language_code"] = language_code
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
        
        return response.content

    def generate_lesson_audio(self, script_data: List[Dict]) -> Tuple[bytes, List[Dict]]:
        """Generate complete lesson audio with pedagogical pauses and return timing data
        Returns: (audio_data, line_timings)"""
        try:
            segments = []
            line_timings = []
            cumulative_time = 0.0
            
            for i, line in enumerate(script_data):
                speaker = line['speaker']
                text = line['text']
                
                if speaker not in VOICE_CONFIG:
                    logger.warning(f"Unknown speaker: {speaker}, defaulting to NARRATOR")
                    speaker = "NARRATOR"
                
                voice_config = VOICE_CONFIG[speaker]
                
                # Track start time for this line
                line_start_time = cumulative_time
                
                logger.info(f"Generating audio segment {i+1}/{len(script_data)}: {speaker}")
                audio_data = self.generate_audio_segment(speaker, text, voice_config)
                segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                segments.append(segment)
                
                # Update cumulative time with segment duration
                cumulative_time += len(segment) / 1000.0  # Convert ms to seconds
                line_end_time = cumulative_time
                
                # Store line timing
                line_timings.append({
                    'line_number': i,
                    'speaker': speaker,
                    'text_content': text,
                    'start_time': line_start_time,
                    'end_time': line_end_time
                })
                
                logger.info(f"  Line timing: {line_start_time:.2f}s - {line_end_time:.2f}s")
                
                # Calculate pause
                pause_duration = 1.0
                
                if speaker == "NARRATOR":
                    pause_context = self.detect_pause_context(text)
                    
                    following_text = ""
                    if i + 1 < len(script_data):
                        try:
                            following_text = script_data[i + 1]['text']
                        except (KeyError, IndexError):
                            logger.warning(f"Could not parse next line after line {i}")
                            following_text = ""
                    
                    pause_duration = self.calculate_pedagogical_pause(following_text, pause_context)
                    
                    logger.info(f"PAUSE DEBUG ANALYSIS:")
                    logger.info(f"  Line {i}: '{text}'")
                    logger.info(f"  Context: {pause_context}")
                    logger.info(f"  Following text: '{following_text[:50]}...'")
                    logger.info(f"  Pause duration: {pause_duration}s")
                
                pause = AudioSegment.silent(duration=int(pause_duration * 1000))
                segments.append(pause)
                
                # Update cumulative time with pause
                cumulative_time += pause_duration
            
            combined = sum(segments)
            
            output_buffer = io.BytesIO()
            combined.export(output_buffer, format="mp3", bitrate="128k")
            
            audio_data = output_buffer.getvalue()
            logger.info(f"Generated {len(segments)//2} audio segments with pedagogical pauses")
            logger.info(f"Total audio size: {len(audio_data) / 1024:.2f} KB")
            logger.info(f"Total duration: {cumulative_time:.2f} seconds")
            
            return audio_data, line_timings
            
        except Exception as e:
            logger.error(f"Failed to generate lesson audio: {e}")
            raise

    def generate_vocabulary_audio(self, hungarian_word: str) -> bytes:
        """Generate audio for a single vocabulary word"""
        try:
            voice_config = VOICE_CONFIG["Balasz"]
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}"
            
            corrected_word = self.apply_pronunciation_fixes(hungarian_word)
            
            payload = {
                "text": corrected_word,
                "model_id": "eleven_turbo_v2_5",
                "language_code": "hu",
                "voice_settings": {
                    "stability": voice_config['stability'],
                    "similarity_boost": voice_config['similarity_boost'],
                    "style": voice_config.get('style', 0.0),
                    "use_speaker_boost": True
                }
            }
            
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
            
            mp3_audio = response.content
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_audio))
            
            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format="wav")
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate vocabulary audio for '{hungarian_word}': {e}")
            raise

    def validate_audio(self, audio_data: bytes, expected_min_size: int = 1000) -> bool:
        """Validate generated audio data"""
        if not audio_data:
            logger.error("Audio data is empty")
            return False
        
        if len(audio_data) < expected_min_size:
            logger.error(f"Audio size too small: {len(audio_data)} bytes (minimum: {expected_min_size})")
            return False
        
        return True

    def upload_to_supabase_storage(self, file_path: str, audio_data: bytes, content_type: str = "audio/mpeg") -> str:
        """Upload audio file to Supabase Storage"""
        try:
            bucket_name = "lessons"
            
            response = supabase.storage.from_(bucket_name).upload(
                file_path,
                audio_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                }
            )
            
            public_url = supabase.storage.from_(bucket_name).get_public_url(file_path)
            
            logger.info(f"Uploaded to Supabase Storage: {file_path}")
            logger.info(f"Public URL: {public_url}")
            
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload to Supabase Storage: {e}")
            raise

    def update_lesson_database(self, lesson_data: dict) -> str:
        """Update lesson information in Supabase database
        Returns: lesson_id"""
        try:
            response = supabase.table('lessons').insert(lesson_data).execute()
            lesson_id = response.data[0]['id']
            logger.info(f"Lesson data saved to database: {lesson_data['title']}")
            return lesson_id
            
        except Exception as e:
            logger.error(f"Failed to update lesson database: {e}")
            raise

    def save_lesson_transcript(self, lesson_id: str, line_timings: List[Dict]) -> None:
        """Save lesson transcript with speaker and timing information"""
        try:
            transcript_records = []
            for timing in line_timings:
                transcript_records.append({
                    'id': str(uuid.uuid4()),
                    'lesson_id': lesson_id,
                    'line_number': timing['line_number'],
                    'speaker': timing['speaker'],
                    'text_content': timing['text_content'],
                    'start_time': timing['start_time'],
                    'end_time': timing['end_time'],
                    'created_at': datetime.now().isoformat()
                })
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(transcript_records), batch_size):
                batch = transcript_records[i:i + batch_size]
                supabase.table('lesson_transcript').insert(batch).execute()
            
            logger.info(f"Saved {len(transcript_records)} transcript lines to database")
            
        except Exception as e:
            logger.error(f"Failed to save lesson transcript: {e}")
            raise

    def get_lesson_id_by_number(self, lesson_number: int) -> Optional[str]:
        """Get lesson ID from database by lesson number"""
        try:
            response = supabase.table('lessons').select('id').eq('lesson_number', lesson_number).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get lesson ID for lesson {lesson_number}: {e}")
            return None

    def update_vocabulary_database(self, lesson_id: str, vocabulary_data: List[Dict]) -> None:
        """Update vocabulary information in Supabase database"""
        try:
            for vocab_item in vocabulary_data:
                vocab_item['lesson_id'] = lesson_id
                vocab_item['created_at'] = datetime.now().isoformat()
            
            response = supabase.table('lesson_vocabulary').insert(vocabulary_data).execute()
            logger.info(f"Saved {len(vocabulary_data)} vocabulary items to database for lesson {lesson_id}")
            
        except Exception as e:
            logger.error(f"Failed to update vocabulary database: {e}")
            raise

    def send_notification_email(self, subject: str, message: str):
        """Send email notification for processing status"""
        logger.info(f"EMAIL NOTIFICATION: {subject} - {message}")

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

# Initialize processor
processor = AudioProcessor()

@app.route('/webhook/google-drive', methods=['POST'])
def handle_google_drive_webhook():
    """Handle webhook from Google Drive Apps Script"""
    try:
        data = request.get_json()
        file_name = data.get('fileName')
        file_content = data.get('fileContent')
        file_type = data.get('fileType', 'lesson')
        
        logger.info(f"Processing file: {file_name}, type: {file_type}")
        
        if processor.daily_usage >= processor.max_daily_lessons:
            raise Exception("Daily processing limit reached")
        
        if file_type == 'lesson':
            success = process_lesson_file(file_name, file_content)
        elif file_type == 'vocabulary':
            success = process_vocabulary_file(file_name, file_content)
        else:
            raise Exception(f"Unknown file type: {file_type}")
        
        if success:
            processor.daily_usage += 1
            return jsonify({"status": "success", "message": "File processed successfully"})
        else:
            return jsonify({"status": "error", "message": "Processing failed"}), 500
            
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        processor.send_notification_email("Processing Failed", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

def process_lesson_file(file_name: str, file_content: str) -> bool:
    """Process full lesson script file with enhanced audio processing and transcript saving"""
    try:
        script_data = processor.parse_lesson_script(file_content)
        if not script_data:
            raise Exception("No valid script data found in file")
        
        logger.info(f"Generating lesson audio with pedagogical pauses and line timing")
        
        audio_data, line_timings = processor.generate_lesson_audio(script_data)
        
        if not processor.validate_audio(audio_data, expected_min_size=10000):
            raise Exception("Generated audio failed validation")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{USER_ID}/{timestamp}-{file_name}.mp3"
        audio_url = processor.upload_to_supabase_storage(file_path, audio_data)
        
        lesson_match = re.search(r'Lesson (\d+)', file_name)
        lesson_number = int(lesson_match.group(1)) if lesson_match else 0
        
        lesson_data = {
            "lesson_number": lesson_number,
            "title": file_name.replace('.docx', '').replace('.txt', '').replace('.csv', ''),
            "description": f"Lesson {lesson_number}",
            "level": "beginner",
            "audio_url": audio_url,
            "is_published": False,
            "created_by": USER_ID,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save lesson and get lesson_id
        lesson_id = processor.update_lesson_database(lesson_data)
        
        # Save transcript with speaker and timing data
        processor.save_lesson_transcript(lesson_id, line_timings)
        
        logger.info(f"Successfully processed lesson: {file_name}")
        logger.info(f"Saved {len(line_timings)} transcript lines with timing data")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process lesson file {file_name}: {e}")
        processor.send_notification_email("Lesson Processing Failed", f"File: {file_name}\nError: {str(e)}")
        return False

def process_vocabulary_file(file_name: str, file_content: str) -> bool:
    """Process vocabulary CSV file"""
    try:
        vocabulary_data = processor.parse_vocabulary_csv(file_content)
        if not vocabulary_data:
            raise Exception("No vocabulary data found in file")
        
        lesson_match = re.search(r'Lesson (\d+)', file_name)
        if not lesson_match:
            raise Exception("Could not extract lesson number from vocabulary filename")
        
        lesson_number = int(lesson_match.group(1))
        
        lesson_id = processor.get_lesson_id_by_number(lesson_number)
        if not lesson_id:
            raise Exception(f"No lesson found in database for lesson number {lesson_number}")
        
        processed_vocabulary = []
        for vocab_item in vocabulary_data:
            hungarian_word = vocab_item['hungarian_word']
            
            logger.info(f"Generating audio for vocabulary word: {hungarian_word}")
            vocab_audio = processor.generate_vocabulary_audio(hungarian_word)
            
            if not processor.validate_audio(vocab_audio, expected_min_size=500):
                logger.warning(f"Skipping vocabulary word '{hungarian_word}' - audio validation failed")
                continue
            
            audio_filename = processor.create_vocabulary_filename(hungarian_word)
            vocab_audio_path = f"vocabulary/{audio_filename}"
            
            vocab_audio_url = processor.upload_to_supabase_storage(vocab_audio_path, vocab_audio, content_type="audio/wav")
            
            vocab_item['audio_file_path'] = vocab_audio_url
            processed_vocabulary.append(vocab_item)
        
        processor.update_vocabulary_database(lesson_id, processed_vocabulary)
        
        logger.info(f"Successfully processed vocabulary file: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process vocabulary file {file_name}: {e}")
        processor.send_notification_email("Vocabulary Processing Failed", f"File: {file_name}\nError: {str(e)}")
        return False

def map_google_speakers_to_lesson_speakers(lesson_id: str, google_word_timings: List[dict]) -> List[dict]:
    """
    Map Google's speaker tags to actual lesson speakers using lesson_transcript as reference.
    Falls back to language detection if no transcript available.
    """
    try:
        # Fetch lesson transcript with speaker labels and timing
        response = supabase.table('lesson_transcript')\
            .select('speaker, text_content, start_time, end_time')\
            .eq('lesson_id', lesson_id)\
            .order('line_number', desc=False)\
            .execute()
        
        if not response.data:
            logger.warning("No lesson_transcript data found - using language detection fallback")
            return map_speakers_by_language_detection(google_word_timings)
        
        lesson_lines = response.data
        logger.info(f"Found {len(lesson_lines)} lesson transcript lines for speaker mapping")
        
        # Build speaker mapping by analyzing time overlap
        speaker_mapping = {}  # {google_speaker_tag: lesson_speaker_name}
        
        for word in google_word_timings:
            word_start = word['start_time']
            word_google_speaker = word.get('speaker_tag')
            
            if not word_google_speaker:
                # No speaker tag from Google - try language detection
                detected_lang = word.get('language_code', 'hu-HU')
                word['speaker'] = 'Narrator' if 'en' in detected_lang.lower() else 'Speaker 1'
                continue
            
            # Already mapped this Google speaker?
            if word_google_speaker in speaker_mapping:
                word['speaker'] = speaker_mapping[word_google_speaker]
                continue
            
            # Find which lesson transcript line this word falls into
            for line in lesson_lines:
                line_start = line.get('start_time')
                line_end = line.get('end_time')
                
                if line_start is None or line_end is None:
                    continue
                
                # Word falls within this line's time range
                if line_start <= word_start <= line_end:
                    actual_speaker = line['speaker']
                    speaker_mapping[word_google_speaker] = actual_speaker
                    word['speaker'] = actual_speaker
                    logger.info(f"Mapped Google Speaker {word_google_speaker} -> {actual_speaker}")
                    break
            
            # If no match found, use generic label
            if 'speaker' not in word:
                word['speaker'] = f"Speaker {word_google_speaker}"
        
        logger.info(f"Speaker mapping complete: {speaker_mapping}")
        return google_word_timings
        
    except Exception as e:
        logger.error(f"Failed to map speakers: {e}")
        # Fallback: use language detection
        return map_speakers_by_language_detection(google_word_timings)

def map_speakers_by_language_detection(word_timings: List[dict]) -> List[dict]:
    """
    Fallback speaker mapping using language detection:
    - English segments = Narrator
    - Hungarian segments = Speaker 1, Speaker 2, etc. (based on speaker_tag)
    
    Admin can edit these labels before publishing.
    """
    logger.info("Using language detection for speaker mapping")
    
    for word in word_timings:
        detected_lang = word.get('language_code', 'hu-HU')
        speaker_tag = word.get('speaker_tag', 1)
        
        if 'en' in detected_lang.lower():
            word['speaker'] = 'Narrator'
        else:
            # Hungarian detected - use speaker tag
            word['speaker'] = f'Speaker {speaker_tag}'
    
    logger.info("Language-based speaker mapping complete")
    return word_timings

@app.route('/generate-transcript', methods=['POST'])
def generate_transcript():
    """Generate word-level transcript from lesson audio using Google Speech-to-Text
    with speaker mapping based on lesson_transcript reference data"""
    
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
        data = request.get_json()
        lesson_id = data.get('lesson_id')
        
        if not lesson_id:
            return jsonify({"status": "error", "message": "lesson_id is required"}), 400
        
        logger.info(f"Starting transcript generation for lesson: {lesson_id}")
        
        # Fetch lesson audio URL
        lesson_response = supabase.table('lessons').select('audio_url, title').eq('id', lesson_id).execute()
        
        if not lesson_response.data:
            return jsonify({"status": "error", "message": "Lesson not found"}), 404
        
        audio_url = lesson_response.data[0]['audio_url']
        lesson_title = lesson_response.data[0].get('title', 'Unknown')
        
        logger.info(f"Fetching audio from: {audio_url}")
        
        # Download audio file
        audio_response = requests.get(audio_url, timeout=120)
        if audio_response.status_code != 200:
            raise Exception(f"Failed to download audio: {audio_response.status_code}")
        
        audio_content = audio_response.content
        logger.info(f"Downloaded {len(audio_content) / (1024*1024):.2f} MB audio file")
        
        # Initialize Google Cloud clients with credentials
        credentials_dict = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        
        # Upload audio to Google Cloud Storage (temporary)
        storage_client = storage.Client(credentials=credentials, project=GOOGLE_CLOUD_PROJECT_ID)
        bucket = storage_client.bucket(GOOGLE_CLOUD_STORAGE_BUCKET)
        
        # Create unique filename for temporary storage
        temp_filename = f"temp-audio-{lesson_id}-{uuid.uuid4()}.mp3"
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
        
        # Configure recognition with automatic language detection and speaker diarization
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=48000,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model="latest_long",
            # Enable automatic language detection
            enable_automatic_language_detection=True,
            language_codes=["hu-HU", "en-US"],  # Hungarian primary, English secondary
            # Enable speaker diarization
            diarization_config=speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,  # At least 2 (narrator + 1 Hungarian speaker)
                max_speaker_count=3,  # Maximum 3 (narrator + Balász + Aggie)
            )
        )
        
        logger.info("Sending audio to Google Cloud Speech-to-Text API...")
        logger.info("Configuration: Automatic language detection + Speaker diarization enabled")
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
            "saved_to_database": True
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
            "transcript_generation_available": GOOGLE_CLOUD_AVAILABLE and bool(GOOGLE_CLOUD_PROJECT_ID)
        }
        
        return jsonify({
            "status": "healthy", 
            "config": config_status,
            "message": "GetTalkin Audio Processor with line timing tracking and improved speaker mapping"
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
        logger.info("Starting GetTalkin Audio Processor with line timing tracking and speaker mapping")
        logger.info("All secrets loaded from environment variables")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    finally:
        processor.cleanup_temp_files()
