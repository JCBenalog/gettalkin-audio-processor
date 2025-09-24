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

    def detect_pause_context(self, narrator_text: str) -> str:
        """Detect if this NARRATOR line should trigger a pause"""
        # Only colon trigger creates pauses
        if narrator_text.strip().endswith(':'):
            return "explicit_pause"
        
        # Everything else is normal dialogue
        return "dialogue"

    def calculate_pedagogical_pause(self, following_text: str, context: str) -> float:
        """Calculate pause duration based on text complexity and pedagogical context"""
        if context == "dialogue":
            return 1.0  # Normal dialogue transition
            
        if context == "explicit_pause":
            # Count words in the phrase to be repeated
            word_count = len(following_text.split())
            
            # Original working timing: repeat aloud + beat or two
            if word_count == 1:
                return 3.0  # Single word: repeat + pause
            elif word_count <= 3:
                return 4.5  # Short phrase
            elif word_count <= 6:
                return 6.0  # Medium phrase
            else:
                return 8.0  # Long phrase
                
        return 1.0  # Default fallback

    def generate_ssml(self, speaker: str, text: str) -> str:
        """Generate SSML markup - minimal for narrator, raw for Hungarian speakers"""
        # Apply pronunciation fixes for Hungarian speakers
        if speaker in ["Balasz", "Aggie"]:
            text = self.apply_pronunciation_fixes(text)
            # Return raw text for Hungarian speakers (they sounded better without SSML)
            return text
        
        # Narrator gets minimal SSML for natural delivery
        if speaker == "NARRATOR":
            ssml = '<speak>'
            
            # Basic question intonation
            if text.strip().endswith('?'):
                text = f'<prosody pitch="medium">{text}</prosody>'
            
            # Slight rate reduction for clarity
            ssml += f'<prosody rate="0.95">{text}</prosody>'
            ssml += '</speak>'
            return ssml
        
        # Fallback for any other speakers
        return text

    def parse_lesson_script(self, file_content: str) -> List[Dict]:
        """Parse lesson CSV content with manual parsing to handle nested quotes and commas"""
        lines = []
        
        try:
            # Split into lines and process manually
            content_lines = file_content.strip().split('\n')
            
            # Skip header line if present
            start_index = 0
            if content_lines[0].lower().startswith('speaker'):
                start_index = 1
            
            for i, line in enumerate(content_lines[start_index:], start=start_index):
                line = line.strip()
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
                    logger.warning(f"No valid comma separator found in line {i+1}: {line}")
                    continue
                
                speaker = line[:comma_index].strip()
                dialogue = line[comma_index + 1:].strip()
                
                # Remove BOM character if present
                if speaker.startswith('\ufeff'):
                    speaker = speaker[1:]
                
                # Remove outer quotes from dialogue if present
                if dialogue.startswith('"') and dialogue.endswith('"'):
                    dialogue = dialogue[1:-1]
                
                # Validate speaker and dialogue
                if not speaker or not dialogue:
                    logger.warning(f"Empty speaker or dialogue at line {i+1}: '{speaker}' - '{dialogue}'")
                    continue
                
                # Default context and pause
                context = "dialogue"
                pause_duration = 1.0
                
                # Check if dialogue ends with colon for pedagogical pause
                if dialogue.strip().endswith(':'):
                    context = "explicit_pause"
                    
                    if speaker == "NARRATOR":
                        # NARRATOR with colon: pause based on NEXT speaker's text
                        if i + 1 < len(content_lines):
                            next_line = content_lines[i + 1].strip()
                            next_comma = -1
                            next_in_quotes = False
                            
                            for k, char in enumerate(next_line):
                                if char == '"':
                                    next_in_quotes = not next_in_quotes
                                elif char == ',' and not next_in_quotes:
                                    next_comma = k
                                    break
                            
                            if next_comma != -1:
                                next_dialogue = next_line[next_comma + 1:].strip()
                                if next_dialogue.startswith('"') and next_dialogue.endswith('"'):
                                    next_dialogue = next_dialogue[1:-1]
                                pause_duration = self.calculate_pedagogical_pause(next_dialogue, context)
                    else:
                        # Non-narrator with colon: pause based on current text
                        text_for_pause = dialogue.rstrip(':').strip()
                        pause_duration = self.calculate_pedagogical_pause(text_for_pause, context)
                
                lines.append({
                    'speaker': speaker,
                    'text': dialogue,
                    'context': context,
                    'pause_duration': pause_duration
                })
                
                logger.info(f"Parsed line {len(lines)}: {speaker} - '{dialogue[:30]}...' - {context}")
                
        except Exception as e:
            logger.error(f"Error parsing lesson script: {e}")
            raise Exception(f"Script parsing failed: {str(e)}")
        
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
            "model_id": "eleven_turbo_v2_5",  # Using your proven model from manual sessions
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
        # Generate silence in milliseconds
        silence_ms = int(duration * 1000)
        silence = AudioSegment.silent(duration=silence_ms, frame_rate=sample_rate)
        return silence

    def bytes_to_audio_segment(self, audio_bytes: bytes) -> AudioSegment:
        """Convert bytes to AudioSegment for processing"""
        # Create temporary file to load the MP3 data
        temp_file = os.path.join(self.temp_dir, f"temp_audio_{uuid.uuid4().hex}.mp3")
        
        try:
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)
            
            audio_segment = AudioSegment.from_mp3(temp_file)
            return audio_segment
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def combine_audio_segments(self, segments: List[AudioSegment]) -> bytes:
        """Combine multiple audio segments into single MP3 file"""
        if not segments:
            raise Exception("No audio segments to combine")
        
        # Start with the first segment
        combined_audio = segments[0]
        
        # Add each subsequent segment
        for segment in segments[1:]:
            combined_audio += segment
        
        # Export to bytes
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
        """Generate complete lesson audio from script data with proper pauses"""
        audio_segments = []
        
        logger.info(f"Generating audio for {len(script_data)} script segments")
        
        for i, segment in enumerate(script_data):
            speaker = segment['speaker']
            text = segment['text']
            context = segment['context']
            pause_duration = segment['pause_duration']
            
            logger.info(f"Processing segment {i+1}: {speaker} - {context} - pause: {pause_duration}s")
            
            # Generate SSML
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
        
        # Combine all audio segments
        logger.info("Combining audio segments")
        combined_audio_bytes = self.combine_audio_segments(audio_segments)
        
        logger.info(f"Successfully generated lesson audio: {len(combined_audio_bytes)} bytes")
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
        # Apply pronunciation fixes
        corrected_word = self.apply_pronunciation_fixes(word)
        
        # Use Hungarian voice (Balasz)
        voice_config = VOICE_CONFIG["Balasz"]
        
        # Generate SSML for vocabulary word
        ssml_text = f'<speak><prosody rate="0.8">{corrected_word}</prosody></speak>'
        
        return self.call_elevenlabs_api(ssml_text, voice_config)

    def sanitize_filename(self, word: str) -> str:
        """Sanitize Hungarian words for safe file naming"""
        # Remove or replace problematic characters
        word = re.sub(r'[.,:;!?"\']', '', word)
        
        # Convert Hungarian characters to ASCII equivalents
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ö': 'o', 'ő': 'o',
            'ú': 'u', 'ü': 'u', 'ű': 'u'
        }
        
        for original, replacement in replacements.items():
            word = word.replace(original, replacement)
        
        # Handle spaces and create filename
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
        
        # Additional validation - try to load with pydub
        try:
            temp_file = os.path.join(self.temp_dir, f"validate_{uuid.uuid4().hex}.mp3")
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            audio_segment = AudioSegment.from_mp3(temp_file)
            os.remove(temp_file)
            
            # Check duration is reasonable (> 0.1 seconds)
            if len(audio_segment) < 100:  # milliseconds
                logger.error("Audio validation failed: duration too short")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed during pydub test: {e}")
            return False

    def upload_to_supabase_storage(self, file_path: str, audio_data: bytes, content_type: str = "audio/mpeg") -> str:
        """Upload audio file to Supabase Storage"""
        try:
            response = supabase.storage.from_("lesson-audio").upload(file_path, audio_data, {"content-type": content_type})
            
            # Get public URL
            public_url = supabase.storage.from_("lesson-audio").get_public_url(file_path)
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload to Supabase: {e}")
            raise Exception(f"Supabase upload error: {str(e)}")

    def update_lesson_database(self, lesson_data: Dict) -> str:
        """Update lessons table in Supabase with UPSERT logic"""
        
        # CRITICAL DEBUG - Add these lines at the very beginning
        logger.info(f"DEBUG - Raw lesson_data received: {json.dumps(lesson_data, indent=2)}")
        logger.info(f"DEBUG - USER_ID constant value: {USER_ID}")
        logger.info(f"DEBUG - created_by value specifically: {lesson_data.get('created_by')}")
        
        try:
            lesson_number = lesson_data.get('lesson_number')
            
            # Check if lesson already exists
            existing_lesson = supabase.table('lessons').select('id').eq('lesson_number', lesson_number).execute()
            
            if existing_lesson.data and len(existing_lesson.data) > 0:
                # Lesson exists - UPDATE it
                lesson_id = existing_lesson.data[0]['id']
                logger.info(f"Updating existing lesson {lesson_number} with ID {lesson_id}")
                
                response = supabase.table('lessons').update(lesson_data).eq('id', lesson_id).execute()
                
                if response.data and len(response.data) > 0:
                    return lesson_id
                else:
                    raise Exception(f"Update failed for lesson {lesson_number}")
            else:
                # Lesson doesn't exist - INSERT it
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
        # TODO: Implement actual email sending

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
            return jsonify({"status": "success", "message": "File processed successfully"})
        else:
            return jsonify({"status": "error", "message": "Processing failed"}), 500
            
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        processor.send_notification_email("Processing Failed", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

def process_lesson_file(file_name: str, file_content: str) -> bool:
    """Process full lesson script file with enhanced audio processing"""
    try:
        # Parse lesson script
        script_data = processor.parse_lesson_script(file_content)
        if not script_data:
            raise Exception("No valid script data found in file")
        
        logger.info(f"Generating lesson audio with pedagogical pauses")
        
        # Generate lesson audio with proper pauses
        audio_data = processor.generate_lesson_audio(script_data)
        
        # Validate audio
        if not processor.validate_audio(audio_data, expected_min_size=10000):
            raise Exception("Generated audio failed validation")
        
        # Upload to Supabase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{USER_ID}/{timestamp}-{file_name}.mp3"
        audio_url = processor.upload_to_supabase_storage(file_path, audio_data)
        
        # Extract lesson number from filename
        lesson_match = re.search(r'Lesson (\d+)', file_name)
        lesson_number = int(lesson_match.group(1)) if lesson_match else 0
        
        # Update database
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
        
        logger.info(f"Successfully processed lesson: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process lesson file {file_name}: {e}")
        processor.send_notification_email("Lesson Processing Failed", f"File: {file_name}\nError: {str(e)}")
        return False

def process_vocabulary_file(file_name: str, file_content: str) -> bool:
    """Process vocabulary CSV file"""
    try:
        # Parse vocabulary CSV
        vocabulary_data = processor.parse_vocabulary_csv(file_content)
        if not vocabulary_data:
            raise Exception("No vocabulary data found in file")
        
        # Extract lesson number from filename
        lesson_match = re.search(r'Lesson (\d+)', file_name)
        if not lesson_match:
            raise Exception("Could not extract lesson number from vocabulary filename")
        
        lesson_number = int(lesson_match.group(1))
        
        # Get the actual lesson ID from database
        lesson_id = processor.get_lesson_id_by_number(lesson_number)
        if not lesson_id:
            raise Exception(f"No lesson found in database for lesson number {lesson_number}")
        
        # Process each vocabulary word
        processed_vocabulary = []
        for vocab_item in vocabulary_data:
            hungarian_word = vocab_item['hungarian_word']
            
            try:
                # Generate audio for vocabulary word
                audio_data = processor.generate_vocabulary_audio(hungarian_word)
                
                # Validate audio
                if not processor.validate_audio(audio_data, expected_min_size=500):
                    logger.warning(f"Audio validation failed for word: {hungarian_word}")
                    continue
                
                # Create filename
                audio_filename = processor.create_vocabulary_filename(hungarian_word)
                
                # Upload to Supabase Storage
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vocab_uuid = str(uuid.uuid4())
                file_path = f"{lesson_id}/vocab-{vocab_uuid}-{timestamp}-{audio_filename}"
                
                processor.upload_to_supabase_storage(file_path, audio_data, "audio/wav")
                
                # Prepare vocabulary data for database
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
        
        # Update vocabulary database
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
        
        # Add to database
        pronunciation_data = {
            "hungarian_word": hungarian_word,
            "phonetic_spelling": phonetic_spelling,
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.table('pronunciation_fixes').insert(pronunciation_data).execute()
        
        # Clear cache to reload on next use
        processor.pronunciation_cache = {}
        
        return jsonify({"status": "success", "message": "Pronunciation fix added"})
        
    except Exception as e:
        logger.error(f"Failed to add pronunciation fix: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    finally:
        # Cleanup on shutdown
        processor.cleanup_temp_files()
