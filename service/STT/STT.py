import cv2
import mediapipe as mp
import numpy as np
import os
import google.genai as genai
from google.genai import types
from PIL import Image
from deep_translator import GoogleTranslator
from gtts import gTTS
import json
import re
import io

class ASLTranslatorFinal:
    def __init__(self, gemini_api_key=None):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
           
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = 'gemini-2.0-flash-exp'
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        self.translator = GoogleTranslator(source='en', target='ar')
    
    def create_expert_prompt(self):
        return """You are a CERTIFIED ASL (American Sign Language) interpreter with 20+ years of experience.

🎯 YOUR TASK:
Analyze this video and translate the ASL signing into clear English text.

⚠️ CRITICAL ASL RULES YOU MUST FOLLOW:

1️⃣ ASL IS ABOUT MOVEMENT, NOT STATIC POSES:
   - A hand shape ALONE is not a sign
   - You must observe the MOTION: direction, speed, repetition
   - Example: Index finger pointing to chest + movement = "I/ME"
   - Example: Index finger pointing outward = "YOU"

2️⃣ COMMON SIGNS TO RECOGNIZE:

**Personal Pronouns:**
- I/ME: Index finger points to chest
- YOU: Index finger points forward
- WE: Index finger moves in arc from one side to other

**Learning/Education:**
- LEARN: Flat hand at forehead moves down to other palm
- STUDY: Similar to LEARN but with wiggling fingers
- KNOW: Fingers tap side of forehead
- UNDERSTAND: Index finger flicks up from forehead

**Communication:**
- SIGN: Both hands alternate circular motions (drawing in air)
- LANGUAGE: Two L-shapes (thumb+index) move apart from chin
- SPEAK/SAY: Index finger circles near mouth
- COMMUNICATE: C-shapes move back and forth

**Common Phrases:**
- "I am learning sign language":
  * I (point to chest)
  * LEARN (hand from forehead to palm)
  * SIGN (circular motions)
  * LANGUAGE (L-shapes from chin)

- "I speak sign language":
  * I (point to chest)
  * SPEAK (finger near mouth)
  * SIGN (circular motions)
  * LANGUAGE (L-shapes)

3️⃣ FACIAL EXPRESSIONS:
- Raised eyebrows + head forward = QUESTION
- Neutral face = STATEMENT
- Nodding = AFFIRMATION

4️⃣ SENTENCE STRUCTURE:
ASL often uses: TIME + TOPIC + COMMENT structure
Example: "TOMORROW I GO SCHOOL" = "I will go to school tomorrow"

5️⃣ FINGERSPELLING:
- Used for names, places, or words without a sign
- Each letter is spelled individually
- Usually faster than normal signing

🚫 COMMON MISTAKES TO AVOID:
- Don't confuse finger configurations with numbers
  * V-shape during movement = "VISIT" or "SEE", NOT "2"
  * Open hand = various meanings, NOT always "5"
- Don't translate word-by-word
  * ASL has its own grammar
- Don't miss compound signs
  * SIGN + LANGUAGE = one concept

📊 YOUR ANALYSIS PROCESS:

STEP 1: Watch the ENTIRE video first
STEP 2: Identify MOVEMENT PATTERNS (not just hand shapes)
STEP 3: Look for these elements:
   - Starting position
   - Movement direction and path
   - Ending position
   - Facial expression
   - Repetition or holds
STEP 4: Match patterns to known signs
STEP 5: Consider context and sentence flow
STEP 6: Form natural English sentence

📤 OUTPUT FORMAT (JSON):
{
  "signs_identified": [
    {
      "sign": "sign_name",
      "confidence": "high/medium/low",
      "movement": "brief description"
    }
  ],
  "english_translation": "Complete natural English sentence",
  "sentence_type": "statement/question/command",
  "confidence": "high/medium/low",
  "interpretation_notes": "Brief explanation of why you chose this translation"
}

✅ QUALITY CHECKLIST:
- Does the translation make sense as a complete sentence?
- Did you consider MOVEMENT, not just hand shapes?
- Did you check facial expressions?
- Is this grammatically correct English?

NOW ANALYZE THIS VIDEO CAREFULLY:"""
    
    def decode_frames_from_flutter(self, frames_data):
        decoded_frames = []
        for frame_bytes in frames_data:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                decoded_frames.append(frame)
        return decoded_frames
    
    def draw_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated = frame.copy()
        
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = annotated.shape
                for idx in [33, 133, 362, 263, 70, 300, 61, 291]:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 2, (255, 0, 0), -1)
        
        return annotated
    
    def analyze_with_gemini(self, frames):
        pil_images = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize((640, 480), Image.LANCZOS)
            pil_images.append(pil_img)
        
        prompt = self.create_expert_prompt()
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt] + pil_images,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.9,
                    max_output_tokens=500
                )
            )
            return self.parse_response(response.text.strip())
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return None
    
    def parse_response(self, text):
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        try:
            return json.loads(text.strip())
        except:
            match = re.search(r'"english_translation"\s*:\s*"([^"]+)"', text)
            if match:
                return {'english_translation': match.group(1), 'confidence': 'medium'}
            return None
    
    def translate_to_arabic(self, english_text):
        try:
            arabic = self.translator.translate(english_text)
            corrections = {
                'لغة الاشارة': 'لغة الإشارة',
                'اتعلم': 'أتعلم',
                'اتكلم': 'أتكلم',
            }
            for wrong, right in corrections.items():
                arabic = arabic.replace(wrong, right)
            return arabic
        except:
            return english_text
    
    def generate_audio_bytes(self, text):
        try:
            tts = gTTS(text=text, lang='ar', slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except:
            return None
    
    def generate_text_file(self, arabic_text):
        try:
            text_buffer = io.BytesIO()
            text_buffer.write(arabic_text.encode('utf-8'))
            text_buffer.seek(0)
            return text_buffer.read()
        except:
            return None
    
    def process_frames_from_flutter(self, frames_data):
        if not frames_data or len(frames_data) == 0:
            return {
                'success': False,
                'error': 'No frames provided',
                'audio_file': None,
                'text_file': None,
                'has_audio': False,
                'has_text': False
            }
        
        try:
            frames = self.decode_frames_from_flutter(frames_data)
            if not frames:
                return {
                    'success': False,
                    'error': 'Failed to decode frames',
                    'audio_file': None,
                    'text_file': None,
                    'has_audio': False,
                    'has_text': False
                }
            
            frames_with_landmarks = []
            for frame in frames:
                annotated = self.draw_landmarks(frame)
                if annotated is not None:
                    frames_with_landmarks.append(annotated)
            
            if not frames_with_landmarks:
                return {
                    'success': False,
                    'error': 'No valid frames with landmarks',
                    'audio_file': None,
                    'text_file': None,
                    'has_audio': False,
                    'has_text': False
                }
            
            result = self.analyze_with_gemini(frames_with_landmarks)
            if not result:
                return {
                    'success': False,
                    'error': 'Analysis failed',
                    'audio_file': None,
                    'text_file': None,
                    'has_audio': False,
                    'has_text': False
                }
            
            english = result.get('english_translation', 'N/A')
            arabic = self.translate_to_arabic(english)
            audio_bytes = self.generate_audio_bytes(arabic)
            text_bytes = self.generate_text_file(arabic)
            
            return {
                'success': True,
                'audio_file': audio_bytes,
                'text_file': text_bytes,
                'has_audio': audio_bytes is not None,
                'has_text': text_bytes is not None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'audio_file': None,
                'text_file': None,
                'has_audio': False,
                'has_text': False
            }