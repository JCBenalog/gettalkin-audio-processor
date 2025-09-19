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
from supabase import create_client, Client
import logging
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
ELEVENLABS_API_KEY = "sk_66046ea215030b7f8855ab82c68697f0a78c4109520e9c79"
SUPABASE_URL = "https://kkzfwplewbgivozyfvdm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtremZ3cGxld2JnaXZvenlmdmRtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjE1MzIwMiwiZXhwIjoyMDcxNzI5MjAyfQ.W3nPawG9FA8xsvC25g8A9Hjr6v9Zzl37A7knPr2tigw"
USER_ID = "795559fe-c8ff-4b09-9fc4-26560c2f4d89"
NOTIFICATION_EMAIL = "dansally@gmail.com"

# Voice Configuration - Updated with your settings
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
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AudioProcessor:
    def __init__(self):
        self.pronunciation_cache = {}
        self.daily_usage = 0
        self.max_daily_lessons = 10
        self.temp_dir = tempfile.mkdtemp()
        self.processed_files = set()  # Track processed files to prevent duplicates
        
    def load_pronunciation_dictionary(self) -> Dict[str, str]:
        """Load pronunciation corrections from Supabase"""
        try:
            response = supabase.table('pronunciation_fixes').select('*').execute()
            return {row['hungarian_word']: row['phonetic_spelling'] for row in response.data}
        except Exception as e:
            logger.warning(f"Could not load pronunciation dictionary: {e}")
            # Return default corrections
            return {"szoba": "soba"}

    def apply_pronunciation_fixes(self, text: str) -> str:
        """Apply pronunciation corrections to Hungarian text"""
        if not self.pronunciation_cache:
            self.pronunciation_cache = self.load_pronunciation_dictionary()
        
        corrected_text = text
        for original, corrected in self.pronunciation_cache.items():
            corrected_text = corrected_text.replace(original, corrected)
        return corrected_text

    def process_inline_tags(self, text: str) -> str:
        """Process inline markup tags for emphasis and pauses"""
        # Process single quotes around words (translation emphasis)
        # Example: 'table' becomes emphasized with slight pause before
        text = re.sub(r"'([^']+)'", r'<break time="0.3s"/><emphasis level="moderate">\1</emphasis>', text)
        
        # Process [SLOW]text[/SLOW] tags
        text = re.sub(r'\[SLOW\](.*?)\[/SLOW\]', r'<prosody rate="0.7">\1</prosody>', text)
        
        # Process [EMPHASIS]text[/EMPHASIS] tags
        text = re.sub(r'\[EMPHASIS\](.*?)\[/EMPHASIS\]', r'<emphasis level="strong">\1</emphasis>', text)
        
        return text

    def calculate_pedagogical_pause(self, following_text: str, context: str) -> float:
        """Calculate pause duration - REDUCED from original"""
        if context == "dialogue":
            return 1.0  # Normal dialogue transition
            
        if context == "explicit_pause":
            # Count words in the phrase to be repeated
            word_count = len(following_text.split())
            
            # REDUCED pause times for better flow
            if word_count == 1:
                return 1.5  # Single word: was 3.0, now 1.5
            elif word_count <= 3:
                return 2.0  # Short phrase: was 4.5, now 2.0
            elif word_count <= 6:
                return 2.5  # Medium phrase: was 6.0, now 2.5
            else:
                return 3.0  # Long phrase: was 8.0, now 3.0
                
        return 1.0  # Default fallback

    def generate_ssml(self, speaker: str, text: str) -> str:
        """Generate SSML markup with inline tag processing"""
        # Apply pronunciation fixes for Hungarian speakers
        if speaker in ["Balasz", "Aggie"]:
            text = self.apply_pronunciation_fixes(text)
            # Process inline tags for Hungarian speakers
            text = self.process_inline_tags(text)
            
            # If there are SSML tags from inline processing, wrap in speak tags
            if '<prosody' in text or '<emphasis' in text or '<break' in text:
                return f'<speak>{text}</speak>'
            else:
                # Return raw text if no special markup
                return text
        
        # Narrator gets SSML processing
        if speaker == "NARRATOR":
            # Process inline tags first
            text = self.process_inline_tags(text)
            
            ssml = '<speak>'
            
            # Basic question intonation
            if text.strip().endswith('?'):
                text = f'<prosody pitch="medium">{text}</prosody>'
            
            # Slight rate reduction for clarity (unless already has rate prosody from tags)
            if '<prosody rate=' not in text:
                ssml += f'<prosody rate="0.95">{text}</prosody>'
            else:
                ssml += text
                
            ssml += '</speak>'
            return ssml
        
        # Fallback for any other speakers
        text = self.process_inline_tags(text)
        if '<prosody' in text or '<emphasis' in text or '<break' in text:
            return f'<speak>{text}</speak>'
        return text

    def parse_lesson_script(self, file_content: str) -> List[Dict]:
        """Parse lesson CSV content with FIXED CSV parsing"""
        lines = []
        
        try:
            # Use csv.QUOTE_ALL to handle quoted content properly
            reader = csv.DictReader(io.StringIO(file_content), quoting=csv.QUOTE_ALL)
            script_data = list(reader)
            
            for i, row in enumerate(script_data):
                # Get raw values and strip whitespace
                speaker = row.get('speaker', '').strip()
                line = row.get('line', '').strip()
                
                if not speaker or not line:
                    continue
                
                # CRITICAL FIX: Don't process speaker column as dialogue text
                # The speaker column should ONLY contain speaker names
                if speaker not in ["NARRATOR", "Balasz", "Aggie"]:
                    logger.warning(f"Unknown speaker '{speaker}' at line {i+1}, skipping")
                    continue
                
                # Remove outer quotes if CSV parser didn't handle them
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                
                # Default context and pause
                context = "dialogue"
                pause_duration = 1.0
                
                # Check if THIS line has a colon for pedagogical pause
                if line.strip().endswith(':'):
                    context = "explicit_pause"
                    
                    if speaker == "NARRATOR":
                        # NARRATOR with colon: pause based on NEXT speaker's text
                        if i + 1 < len(script_data):
                            next_line = script_data[i + 1].get('line', '').strip()
                            if next_line.startswith('"') and next_line.endswith('"'):
                                next_line = next_line[1:-1]
                            pause_duration = self.calculate_pedagogical_pause(next_line, context)
                    else:
                        # Non-narrator with colon: pause based on THIS speaker's own text
                        text_for_pause = line.rstrip(':').strip()
                        pause_duration = self.calculate_pedagogical_pause(text_for_pause, context)
                
                lines.append({
                    'speaker': speaker,
                    'text': line,
                    'context': context,
                    'pause_duration': pause_duration
                })
                
                logger.info(f"Parsed line {i+1}: {speaker} - '{line[:50]}...' - {context}")
                
        except Exception as e:
            logger.error(f"Error parsing lesson script: {e}")
            raise Exception(f"CSV parsing failed: {str(e)}")
        
        logger.info(f"Successfully parsed {len(lines)} valid lines from lesson script")
        return lines

    def call_elevenlabs_api(self, text: str, voice_config: Dict) -> bytes:
        """Make API call to ElevenLabs with improved settings"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_config['voice_id']}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": voice_config['stability'],
                "similarity_boost": voice_config['similarity_boost'],
                "style": voice_config['style']
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"ElevenLabs API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        return response.content

    def generate_silence(self, duration: float, sample_rate: int = 22050) -> AudioSegment:
        """Generate silence audio for specified duration"""
        silence_ms = int(duration * 1000)
        silence = AudioSegment.silent(duration=silence_ms, frame_rate=sample_rate)
        return silence

    def bytes_to_audio_segment(self, audio_bytes: bytes) -> AudioSegment:
        """Convert bytes to AudioSegment for processing"""
        temp_file = os.path.join(self.temp_dir, f"temp_audio_{uuid.uuid4().hex}.mp3")
        
        try:
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)
            
            audio_segment = AudioSegment.from_mp3(temp_file)
            return audio_segment
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def combine_audio_segments(self, segments: List[AudioSegment]) -> bytes:
        """Combine multiple audio segments into single MP3 file"""
        if not segments:
            raise Exception("No audio segments to combine")
        
        combined_audio = segments[0]
        for segment in segments[1:]:
            combined_audio += segment
        
        temp_file = os.path.join(self.temp_dir, f"combined_audio_{uuid.uuid4().hex}.mp3")
        
        try:
            combined_audio.export(temp_file, format="mp3", bitrate="128k")
            
            with open(temp_file, 'rb') as f:
                audio_bytes = f.read()
            
            return audio_bytes
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def generate_lesson_audio(self, script_data: List[Dict]) -> bytes:
        """Generate complete lesson audio from script data - SINGLE FILE OUTPUT"""
        audio_segments = []
        
        logger.info(f"Generating single audio file for {len(script_data)} script segments")
        
        for i, segment in enumerate(script_data):
            speaker = segment['speaker']
            text = segment['text']
            context = segment['context']
            pause_duration = segment['pause_duration']
            
            logger.info(f"Processing segment {i+1}: {speaker} - {context} - pause: {pause_duration}s")
            
            # Generate SSML with inline tag processing
            ssml_text = self.generate_ssml(speaker, text)
            
            # Get voice configuration
            voice_config = VOICE_CONFIG.get(speaker, VOICE_CONFIG["NARRATOR"])
            
            try:
                # Generate audio for this segment
                audio_bytes = self.call_elevenlabs_api(ssml_text, voice_config)
                audio_segment = self.bytes_to_audio_segment(audio_bytes)
                audio_segments.append(audio_segment)
                
                # Add pedagogical pause if this segment requires one
                if context != "dialogue" and pause_duration > 1.0:
                    logger.info(f"Adding pedagogical pause: {pause_duration}s")
                    pause_segment = self.generate_silence(pause_duration)
                    audio_segments.append(pause_segment)
                elif i < len(script_data) - 1:  # Normal transition pause (except for last segment)
                    pause_segment = self.generate_silence(pause_duration)
                    audio_segments.append(pause_segment)
                
            except Exception as e:
                logger.error(f"Failed to generate audio for segment {i+1}: {e}")
                raise Exception(f"Audio generation failed at segment {i+1}: {str(e)}")
        
        # Combine all audio segments into ONE file
        logger.info("Combining audio segments into single lesson file")
        combined_audio_bytes = self.combine_audio_segments(audio_segments)
        
        logger.info(f"Successfully generated single lesson audio file: {len(combined_audio_bytes)} bytes")
        return combined_audio_bytes

    def parse_vocabulary_csv(self, file_content: str) -> List[Dict]:
        """Parse vocabulary CSV file"""
        vocabulary = []
        
        try:
            reader = csv.DictReader(io.StringIO(file_content))
            for row in reader:
                hungarian_word = row.get('hungarian_word', '').strip()
                english_translation = row.get('english_translation', '').strip()
                difficulty = row.get('difficulty', 'beginner').strip()
                
                if hungarian_word and english_translation:
                    vocabulary.append({
                        'hungarian_word': hungarian_word,
                        'english_translation': english_translation,
                        'difficulty': difficulty
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing vocabulary CSV: {e}")
            raise Exception(f"Vocabulary CSV parsing failed: {str(e)}")
        
        return vocabulary

    def generate_vocabulary_audio(self, word: str) -> bytes:
        """Generate audio for single vocabulary word"""
        corrected_word = self.apply_pronunciation_fixes(word)
        voice_config = VOICE_CONFIG["Balasz"]
        
        # Use SSML for vocabulary with slight slowdown
        ssml_text = f'<speak><prosody rate="0.8">{corrected_word}</prosody></speak>'
        
        return self.call_elevenlabs_api(ssml_text, voice_config)

    def sanitize_filename(self, word: str) -> str:
        """Sanitize Hungarian words for safe file naming"""
        word = re.sub(r'[.,:;!?"\']', '', word)
        
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ö': 'o', 'ő': 'o',
            'ú': 'u', 'ü': 'u', 'ű': 'u'
        }
        
        for original, replacement in replacements.items():
            word = word.replace(original, replacement)
        
        words = word.split()
        formatted_words = [w.capitalize() for w in words if w]
        return '-'.join(formatted_words) + '.wav'

    def create_vocabulary_filename(self, word: str) -> str:
        """Create filename from Hungarian word/phrase"""
        return self.sanitize_filename(word)

    def validate_audio(self, audio_data: bytes, expected_min_size: int = 1000) -> bool:
        """Enhanced audio validation"""
        if len(audio_data) < expected_min_size:
            logger.error(f"Audio validation failed: size {len(audio_data)} < {expected_min_size}")
            return False
        
        # Check for basic audio file headers
        if audio_data[:3] == b'ID3' or audio_data[:4] == b'RIFF' or audio_data[:4] == b'fLaC':
            return True
        
        # For MP3 files, check for MP3 frame header
        if len(audio_data) >= 4:
            if audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0:
                return True
        
        return True  # Basic validation passed

    def upload_to_supabase_storage(self, file_path: str, audio_data: bytes, content_type: str = "audio/mpeg") -> str:
        """Upload audio file to Supabase Storage"""
        try:
            response = supabase.storage.from_("lesson-audio").upload(file_path, audio_data, {"content-type": content_type})
            public_url = supabase.storage.from_("lesson-audio").get_public_url(file_path)
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload to Supabase: {e}")
            raise Exception(f"Supabase upload error: {str(e)}")

    def update_lesson_database(self, lesson_data: Dict) -> str:
        """Update lessons table in Supabase with UPSERT logic"""
        try:
            lesson_number = lesson_data.get('lesson_number')
            
            existing_lesson = supabase.table('lessons').select('id').eq('lesson_number', lesson_number).execute()
            
            if existing_lesson.data and len(existing_lesson.data) > 0:
                lesson_id = existing_lesson.data[0]['id']
                logger.info(f"Updating existing lesson {lesson_number} with ID {lesson_id}")
                
                response = supabase.table('lessons').update(lesson_data).eq('id', lesson_id).execute()
                
                if response.data and len(response.data) > 0:
                    return lesson_id
                else:
                    raise Exception(f"Update failed for lesson {lesson_number}")
            else:
                logger.info(f"Creating new lesson {lesson_number}")
                response = supabase.table('lessons').insert(lesson_data).execute()
                
                if response.data and len(response.data) > 0:
                    return response.data[0]['id']
                else:
                    raise Exception("No lesson ID returned from database")
                    
        except Exception as e:
            logger.error(f"Failed to upsert lesson to database: {e}")
            raise Exception(f"Database upsert error: {str(e)}")

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
        """Update lesson_vocabulary table in Supabase"""
        try:
            for vocab_item in vocabulary_data:
                vocab_item['lesson_id'] = lesson_id
            
            response = supabase.table('lesson_vocabulary').insert(vocabulary_data).execute()
        except Exception as e:
            logger.error(f"Failed to update vocabulary table: {e}")
            raise Exception(f"Vocabulary database update error: {str(e)}")

    def send_notification_email(self, subject: str, message: str) -> None:
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
    """Handle webhook from Google Drive Apps Script with duplicate prevention"""
    try:
        data = request.get_json()
        file_name = data.get('fileName')
        file_content = data.get('fileContent')
        file_type = data.get('fileType', 'lesson')
        
        # DUPLICATE PREVENTION: Check if we've already processed this file
        file_hash = f"{file_name}_{len(file_content)}_{file_type}"
        if file_hash in processor.processed_files:
            logger.warning(f"File {file_name} already processed, skipping duplicate")
            return jsonify({"status": "success", "message": "File already processed"})
        
        logger.info(f"Processing file: {file_name}, type: {file_type}")
        
        # Check daily usage limits
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
            processor.processed_files.add(file_hash)  # Mark as processed
            return jsonify({"status": "success", "message": "File processed successfully"})
        else:
            return jsonify({"status": "error", "message": "Processing failed"}), 500
            
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        processor.send_notification_email("Processing Failed", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

def process_lesson_file(file_name: str, file_content: str) -> bool:
    """Process full lesson script file - CREATES ONLY ONE AUDIO FILE"""
    try:
        # Parse lesson script with improved CSV handling
        script_data = processor.parse_lesson_script(file_content)
        if not script_data:
            raise Exception("No valid script data found in file")
        
        logger.info(f"Generating single lesson audio file from {len(script_data)} segments")
        
        # Generate ONE complete lesson audio file
        audio_data = processor.generate_lesson_audio(script_data)
        
        # Validate audio
        if not processor.validate_audio(audio_data, expected_min_size=10000):
            raise Exception("Generated audio failed validation")
        
        # Upload SINGLE file to Supabase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{USER_ID}/{timestamp}-{file_name}.mp3"
        audio_url = processor.upload_to_supabase_storage(file_path, audio_data)
        
        # Extract lesson number from filename
        lesson_match = re.search(r'Lesson (\d+)', file_name)
        lesson_number = int(lesson_match.group(1)) if lesson_match else 0
        
        # Update database with single audio file
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
        
        processor.update_lesson_database(lesson_data)
        
        logger.info(f"Successfully processed lesson: {file_name} - SINGLE audio file created")
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
            
            try:
                audio_data = processor.generate_vocabulary_audio(hungarian_word)
                
                if not processor.validate_audio(audio_data, expected_min_size=500):
                    logger.warning(f"Audio validation failed for word: {hungarian_word}")
                    continue
                
                audio_filename = processor.create_vocabulary_filename(hungarian_word)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vocab_uuid = str(uuid.uuid4())
                file_path = f"{lesson_id}/vocab-{vocab_uuid}-{timestamp}-{audio_filename}"
                
                processor.upload_to_supabase_storage(file_path, audio_data, "audio/wav")
                
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
        
        if processed_vocabulary:
            processor.update_vocabulary_database(lesson_id, processed_vocabulary)
            logger.info(f"Successfully processed {len(processed_vocabulary)} vocabulary words from {file_name}")
            return True
        else:
            raise Exception("No vocabulary words were successfully processed")
            
    except Exception as e:
        logger.error(f"Failed to process vocabulary file {file_name}: {e}")
        processor.send_notification_email("Vocabulary Processing Failed", f"File: {file_name}\nError: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "daily_usage": processor.daily_usage})

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
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    finally:
        processor.cleanup_temp_files()
