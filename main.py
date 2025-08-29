import os
import json
import csv
import io
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from flask import Flask, request, jsonify
from supabase import create_client, Client
import logging
from pathlib import Path

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
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AudioProcessor:
    def __init__(self):
        self.pronunciation_cache = {}
        self.daily_usage = 0
        self.max_daily_lessons = 10
        
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
    
    def calculate_pause_duration(self, text: str, context: str = "dialogue") -> float:
        """Calculate pedagogically appropriate pause duration"""
        word_count = len(text.split())
        
        if context == "your_turn":
            if word_count <= 2:
                return 8.0  # Simple recall
            elif word_count <= 5:
                return 12.0  # Contextual application
            else:
                return 15.0  # Complex construction
        elif context == "repeat":
            if word_count == 1:
                return 2.5
            elif word_count <= 3:
                return 4.0
            elif word_count <= 6:
                return 6.0
            else:
                return 8.0
        else:  # normal dialogue
            return 1.0
    
    def generate_ssml(self, speaker: str, text: str, context: str = "dialogue") -> str:
        """Generate SSML markup for enhanced speech synthesis"""
        
        # Apply pronunciation fixes for Hungarian speakers
        if speaker in ["Balasz", "Aggie"]:
            text = self.apply_pronunciation_fixes(text)
        
        ssml = '<speak>'
        
        # Add voice-specific prosody
        if speaker == "NARRATOR":
            # Add emphasis for quoted phrases
            text = re.sub(r'"([^"]*)"', r'<emphasis level="strong">"\1"</emphasis>', text)
            # Question intonation
            if text.strip().endswith('?'):
                text = f'<prosody pitch="high">{text}</prosody>'
            # Instructional clarity
            ssml += f'<prosody rate="0.9">{text}</prosody>'
        else:
            # Hungarian speakers - normalize volume and pace
            ssml += f'<prosody volume="medium" rate="1.0">{text}</prosody>'
        
        ssml += '</speak>'
        return ssml
    
    def parse_lesson_script(self, file_content: str) -> List[Dict]:
        """Parse lesson DOCX content (CSV format) into structured data"""
        lines = []
        reader = csv.DictReader(io.StringIO(file_content))
        
        for row in reader:
            speaker = row.get('speaker', '').strip()
            line = row.get('line', '').strip()
            
            if speaker and line:
                # Remove quotes around the line content
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                
                # Determine context for pause calculation
                context = "dialogue"
                if "listen and repeat" in line.lower():
                    context = "repeat"
                elif "your turn" in line.lower() or "now say" in line.lower():
                    context = "your_turn"
                
                lines.append({
                    'speaker': speaker,
                    'text': line,
                    'context': context
                })
        
        return lines
    
    def generate_lesson_audio(self, script_data: List[Dict]) -> bytes:
        """Generate complete lesson audio from script data"""
        audio_segments = []
        
        for i, segment in enumerate(script_data):
            speaker = segment['speaker']
            text = segment['text']
            context = segment['context']
            
            # Generate SSML
            ssml_text = self.generate_ssml(speaker, text, context)
            
            # Get voice configuration
            voice_config = VOICE_CONFIG.get(speaker, VOICE_CONFIG["NARRATOR"])
            
            # Call ElevenLabs API
            try:
                audio_bytes = self.call_elevenlabs_api(ssml_text, voice_config)
                audio_segments.append(audio_bytes)
                
                # Add pause after segment (except for last one)
                if i < len(script_data) - 1:
                    pause_duration = self.calculate_pause_duration(text, context)
                    pause_audio = self.generate_silence(pause_duration)
                    audio_segments.append(pause_audio)
                    
            except Exception as e:
                logger.error(f"Failed to generate audio for segment: {e}")
                raise
        
        # Combine all audio segments
        return self.combine_audio_segments(audio_segments)
    
    def call_elevenlabs_api(self, text: str, voice_config: Dict) -> bytes:
        """Make API call to ElevenLabs"""
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
    
    def generate_silence(self, duration: float) -> bytes:
        """Generate silence audio for specified duration"""
        # This is a simplified approach - in production, you'd generate actual silence
        # For now, we'll return empty bytes and handle pause timing in post-processing
        return b''
    
    def combine_audio_segments(self, segments: List[bytes]) -> bytes:
        """Combine multiple audio segments into single file"""
        # Simplified combination - in production, use pydub or similar
        combined = b''.join(segment for segment in segments if segment)
        return combined
    
    def parse_vocabulary_csv(self, file_content: str) -> List[Dict]:
        """Parse vocabulary CSV file"""
        vocabulary = []
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
    
    def create_vocabulary_filename(self, word: str) -> str:
        """Create filename from Hungarian word/phrase"""
        # Replace spaces with hyphens, capitalize each word
        words = word.split()
        formatted_words = [w.capitalize() for w in words]
        return '-'.join(formatted_words) + '.wav'
    
    def validate_audio(self, audio_data: bytes, expected_min_size: int = 1000) -> bool:
        """Basic audio validation"""
        if len(audio_data) < expected_min_size:
            return False
        
        # Check for basic audio file headers (simplified)
        if audio_data[:3] == b'ID3' or audio_data[:4] == b'RIFF':
            return True
            
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
    
    def update_lesson_database(self, lesson_data: Dict) -> None:
        """Update lessons table in Supabase"""
        try:
            response = supabase.table('lessons').insert(lesson_data).execute()
        except Exception as e:
            logger.error(f"Failed to update lessons table: {e}")
            raise Exception(f"Database update error: {str(e)}")
    
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
        # Simplified email notification - in production, integrate with SendGrid
        logger.info(f"EMAIL NOTIFICATION: {subject} - {message}")
        # TODO: Implement actual email sending

# Initialize processor
processor = AudioProcessor()

@app.route('/webhook/google-drive', methods=['POST'])
def handle_google_drive_webhook():
    """Handle webhook from Google Drive Apps Script"""
    try:
        data = request.get_json()
        file_name = data.get('fileName')
        file_content = data.get('fileContent')
        file_type = data.get('fileType', 'lesson')  # 'lesson' or 'vocabulary'
        
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
    """Process full lesson script file"""
    try:
        # Parse lesson script
        script_data = processor.parse_lesson_script(file_content)
        
        # Generate lesson audio
        audio_data = processor.generate_lesson_audio(script_data)
        
        # Validate audio
        if not processor.validate_audio(audio_data):
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
            "title": file_name.replace('.docx', ''),
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
        
        # Extract lesson number from filename
        lesson_match = re.search(r'Lesson (\d+)', file_name)
        if not lesson_match:
            raise Exception("Could not extract lesson number from vocabulary filename")
        
        lesson_number = lesson_match.group(1)
        lesson_id = str(uuid.uuid4())  # Generate lesson ID - in production, lookup existing lesson
        
        # Process each vocabulary word
        processed_vocabulary = []
        
        for vocab_item in vocabulary_data:
            hungarian_word = vocab_item['hungarian_word']
            
            try:
                # Generate audio for vocabulary word
                audio_data = processor.generate_vocabulary_audio(hungarian_word)
                
                # Validate audio
                if not processor.validate_audio(audio_data, expected_min_size=500):
                    raise Exception(f"Audio validation failed for word: {hungarian_word}")
                
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
                # Continue with other words rather than failing entire batch
                continue
        
        # Update vocabulary database
        if processed_vocabulary:
            processor.update_vocabulary_database(lesson_id, processed_vocabulary)
            logger.info(f"Successfully processed {len(processed_vocabulary)} vocabulary words from {file_name}")
        
        return len(processed_vocabulary) > 0
        
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
