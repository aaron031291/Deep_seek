#!/usr/bin/env python3


from typing import Dict, Any, Optional, List, Set, Tuple, Union
from loguru import logger
import asyncio
import uuid
import json
import time
from datetime import datetime
import numpy as np
import threading
import queue
import os
import re
from pathlib import Path
import cv2
import pyaudio
import wave
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import whisper
from PIL import Image
import pytesseract
from pydantic import BaseModel, Field

from edgenative_umaas.utils.event_bus import EventBus
from edgenative_umaas.security.security_manager import SecurityManager

class ContextWindow(BaseModel):
    """Context window for maintaining conversation state and context."""
    session_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context_embeddings: Optional[List[float]] = None
    active_tasks: Dict[str, Any] = Field(default_factory=dict)
    screen_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    system_state: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message to the context window."""
        self.messages.append(message)
        # Limit the number of messages to prevent context explosion
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

class CognitiveCollaborationSystem:
    """
    Cognitive Collaboration System.
    
    A revolutionary yet attainable AI collaboration system that enables:
    1. Continuous contextual awareness through multimodal inputs
    2. Proactive assistance based on user behavior and system state
    3. Seamless voice-driven interaction with natural conversation
    4. Real-time screen understanding and intelligent automation
    5. Adaptive learning from user interactions and feedback
    """
    
    def __init__(self, event_bus: EventBus, security_manager: SecurityManager, config: Dict[str, Any]):
        """Initialize the Cognitive Collaboration System."""
        self.event_bus = event_bus
        self.security_manager = security_manager
        self.config = config
        
        # Core components
        self.active_sessions = {}  # session_id -> ContextWindow
        self.voice_streams = {}    # session_id -> audio stream
        self.screen_streams = {}   # session_id -> screen capture stream
        
        # AI models
        self.speech_recognizer = None  # Whisper model for speech recognition
        self.text_generator = None     # LLM for text generation
        self.vision_analyzer = None    # Vision model for screen understanding
        self.embedding_model = None    # Model for semantic embeddings
        
        # Processing queues
        self.voice_queue = queue.Queue()
        self.screen_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Knowledge base
        self.knowledge_vectors = []
        self.knowledge_texts = []
        
        # User behavior patterns
        self.user_patterns = {}  # user_id -> patterns
        
        # System capabilities registry
        self.capabilities = {
            "voice_commands": self._process_voice_command,
            "screen_analysis": self._analyze_screen,
            "task_automation": self._automate_task,
            "information_retrieval": self._retrieve_information,
            "code_generation": self._generate_code,
            "data_visualization": self._visualize_data,
            "system_monitoring": self._monitor_system,
            "predictive_assistance": self._provide_predictive_assistance
        }
        
        # Worker threads
        self.workers = []
    
    async def initialize(self) -> bool:
        """Initialize the Cognitive Collaboration System."""
        logger.info("Initializing Cognitive Collaboration System")
        
        try:
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Start worker threads
            self._start_workers()
            
            # Subscribe to events
            await self.event_bus.subscribe("user.joined", self._handle_user_joined)
            await self.event_bus.subscribe("user.left", self._handle_user_left)
            await self.event_bus.subscribe("user.message", self._handle_user_message)
            await self.event_bus.subscribe("system.state_changed", self._handle_system_state_changed)
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            logger.info("Cognitive Collaboration System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Collaboration System: {e}")
            return False
    
    async def _initialize_ai_models(self):
        """Initialize AI models using the most efficient approach for edge deployment."""
        # Use a background thread for model loading to avoid blocking
        def load_models():
            try:
                # 1. Initialize Whisper for speech recognition (small model for edge devices)
                self.speech_recognizer = whisper.load_model("base")
                
                # 2. Initialize vision model for screen understanding
                self.vision_analyzer = pipeline("image-classification", 
                                               model="microsoft/resnet-50")
                
                # 3. Initialize text embedding model
                self.embedding_model = pipeline("feature-extraction", 
                                               model="sentence-transformers/all-MiniLM-L6-v2")
                
                # 4. Initialize text generation model
                # Use a quantized model for edge efficiency
                model_name = "TheBloke/Llama-2-7B-Chat-GGML"
                self.text_generator = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto",
                    load_in_8bit=True  # 8-bit quantization for memory efficiency
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                logger.info("AI models loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading AI models: {e}")
                raise
        
        # Start model loading in background
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        
        # Wait for models to load with a timeout
        for _ in range(30):  # 30 second timeout
            if self.speech_recognizer and self.vision_analyzer and self.embedding_model and self.text_generator:
                return
            await asyncio.sleep(1)
        
        if not (self.speech_recognizer and self.vision_analyzer and self.embedding_model and self.text_generator):
            raise TimeoutError("Timed out waiting for AI models to load")
    
    def _start_workers(self):
        """Start worker threads for processing different input streams."""
        # 1. Voice processing worker
        voice_worker = threading.Thread(target=self._voice_processing_worker)
        voice_worker.daemon = True
        voice_worker.start()
        self.workers.append(voice_worker)
        
        # 2. Screen analysis worker
        screen_worker = threading.Thread(target=self._screen_processing_worker)
        screen_worker.daemon = True
        screen_worker.start()
        self.workers.append(screen_worker)
        
        # 3. Command execution worker
        command_worker = threading.Thread(target=self._command_processing_worker)
        command_worker.daemon = True
        command_worker.start()
        self.workers.append(command_worker)
        
        logger.info(f"Started {len(self.workers)} worker threads")
    
    async def _load_knowledge_base(self):
        """Load and index the knowledge base for quick retrieval."""
        knowledge_dir = Path(self.config.get("knowledge_dir", "./knowledge"))
        
        if not knowledge_dir.exists():
            logger.warning(f"Knowledge directory {knowledge_dir} does not exist. Creating it.")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Load all text files in the knowledge directory
        for file_path in knowledge_dir.glob("**/*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create embedding for the content
                embedding = self.embedding_model(content)
                embedding_vector = np.mean(embedding[0], axis=0)
                
                # Store the content and its embedding
                self.knowledge_texts.append(content)
                self.knowledge_vectors.append(embedding_vector)
                
            except Exception as e:
                logger.error(f"Error loading knowledge file {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge items")
    
    async def start_session(self, user_id: str) -> str:
        """Start a new collaboration session."""
        # Create a new session
        session_id = str(uuid.uuid4())
        
        # Initialize context window
        self.active_sessions[session_id] = ContextWindow(
            session_id=session_id,
            user_preferences=await self._load_user_preferences(user_id),
            system_state={"start_time": datetime.now().isoformat()}
        )
        
        # Start voice and screen streams
        await self._start_voice_stream(session_id)
        await self._start_screen_stream(session_id)
        
        logger.info(f"Started collaboration session {session_id} for user {user_id}")
        
        # Send welcome message
        await self.send_assistant_message(
            session_id,
            "I'm ready to collaborate with you. You can speak naturally or type commands. "
            "I'll observe your screen to provide contextual assistance when needed."
        )
        
        return session_id
    
    async def _start_voice_stream(self, session_id: str):
        """Start voice input stream for a session."""
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
            stream_callback=lambda in_data, frame_count, time_info, status_flags: 
                self._voice_callback(session_id, in_data, frame_count, time_info, status_flags)
        )
        
        # Store the stream
        self.voice_streams[session_id] = {
            "stream": stream,
            "pyaudio": p,
            "buffer": bytearray(),
            "last_process_time": time.time()
        }
        
        logger.info(f"Started voice stream for session {session_id}")
    
    def _voice_callback(self, session_id, in_data, frame_count, time_info, status_flags):
        """Callback for voice stream data."""
        if session_id not in self.voice_streams:
            return (in_data, pyaudio.paContinue)
        
        # Add data to buffer
        self.voice_streams[session_id]["buffer"].extend(in_data)
        
        # Process buffer if it's been at least 1 second since last processing
        current_time = time.time()
        if current_time - self.voice_streams[session_id]["last_process_time"] >= 1.0:
            # Copy buffer and clear it
            buffer_copy = self.voice_streams[session_id]["buffer"].copy()
            self.voice_streams[session_id]["buffer"] = bytearray()
            
            # Add to processing queue
            self.voice_queue.put((session_id, buffer_copy))
            
            # Update last process time
            self.voice_streams[session_id]["last_process_time"] = current_time
        
        return (in_data, pyaudio.paContinue)
    
    async def _start_screen_stream(self, session_id: str):
        """Start screen capture stream for a session."""
        # Initialize screen capture
        self.screen_streams[session_id] = {
            "active": True,
            "last_capture_time": 0,
            "capture_interval": 1.0,  # Capture every 1 second
            "last_image": None
        }
        
        # Start screen capture thread
        thread = threading.Thread(
            target=self._screen_capture_worker,
            args=(session_id,)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started screen capture for session {session_id}")
    
    def _screen_capture_worker(self, session_id: str):
        """Worker thread for screen capture."""
        try:
            import mss
            
            with mss.mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1]
                
                while session_id in self.screen_streams and self.screen_streams[session_id]["active"]:
                    current_time = time.time()
                    
                    # Capture screen at specified interval
                    if current_time - self.screen_streams[session_id]["last_capture_time"] >= self.screen_streams[session_id]["capture_interval"]:
                        # Capture screen
                        screenshot = sct.grab(monitor)
                        
                        # Convert to numpy array
                        img = np.array(screenshot)
                        
                        # Resize to reduce processing load
                        img = cv2.resize(img, (800, 600))
                        
                        # Store the image
                        self.screen_streams[session_id]["last_image"] = img
                        
                        # Add to processing queue
                        self.screen_queue.put((session_id, img.copy()))
                        
                        # Update last capture time
                        self.screen_streams[session_id]["last_capture_time"] = current_time
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in screen capture worker for session {session_id}: {e}")
    
    def _voice_processing_worker(self):
        """Worker thread for processing voice input."""
        while True:
            try:
                # Get item from queue
                session_id, audio_data = self.voice_queue.get(timeout=1.0)
                
                # Skip if speech recognizer is not initialized
                if not self.speech_recognizer:
                    self.voice_queue.task_done()
                    continue
                
                # Convert audio data to WAV format for Whisper
                with wave.open("temp_audio.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                
                # Transcribe audio
                result = self.speech_recognizer.transcribe("temp_audio.wav")
                transcription = result["text"].strip()
                
                # Skip if empty
                if not transcription:
                    self.voice_queue.task_done()
                    continue
                
                # Process the transcription
                logger.info(f"Voice input from session {session_id}: {transcription}")
                
                # Create a task to handle the voice input
                asyncio.run_coroutine_threadsafe(
                    self._handle_voice_input(session_id, transc
               # Create a task to handle the voice input
                asyncio.run_coroutine_threadsafe(
                    self._handle_voice_input(session_id, transcription),
                    asyncio.get_event_loop()
                )
                
                # Clean up
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in voice processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.voice_queue.task_done()
    
    def _screen_processing_worker(self):
        """Worker thread for processing screen captures."""
        while True:
            try:
                # Get item from queue
                session_id, image = self.screen_queue.get(timeout=1.0)
                
                # Skip if vision analyzer is not initialized
                if not self.vision_analyzer:
                    self.screen_queue.task_done()
                    continue
                
                # Process the screen capture
                screen_context = self._extract_screen_context(image)
                
                # Update context window with screen context
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].screen_context = screen_context
                
                # Check for significant changes that might require proactive assistance
                if self._should_provide_assistance(session_id, screen_context):
                    # Create a task to provide assistance
                    asyncio.run_coroutine_threadsafe(
                        self._provide_proactive_assistance(session_id, screen_context),
                        asyncio.get_event_loop()
                    )
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in screen processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.screen_queue.task_done()
    
    def _command_processing_worker(self):
        """Worker thread for processing commands."""
        while True:
            try:
                # Get item from queue
                session_id, command, args = self.command_queue.get(timeout=1.0)
                
                # Process the command
                logger.info(f"Processing command from session {session_id}: {command} {args}")
                
                # Execute the command
                result = None
                if command in self.capabilities:
                    try:
                        # Execute the command handler
                        result = self.capabilities[command](session_id, args)
                        
                        # Create a task to send the result
                        if result:
                            asyncio.run_coroutine_threadsafe(
                                self.send_assistant_message(session_id, result),
                                asyncio.get_event_loop()
                            )
                    
                    except Exception as e:
                        error_message = f"Error executing command {command}: {e}"
                        logger.error(error_message)
                        
                        # Send error message
                        asyncio.run_coroutine_threadsafe(
                            self.send_assistant_message(session_id, error_message),
                            asyncio.get_event_loop()
                        )
                else:
                    # Unknown command
                    unknown_command_message = f"Unknown command: {command}. Type 'help' for available commands."
                    
                    # Send unknown command message
                    asyncio.run_coroutine_threadsafe(
                        self.send_assistant_message(session_id, unknown_command_message),
                        asyncio.get_event_loop()
                    )
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in command processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.command_queue.task_done()
    
    def _extract_screen_context(self, image) -> Dict[str, Any]:
        """
        Extract context from screen capture using computer vision.
        
        This is a revolutionary feature that provides real-time understanding
        of what the user is seeing and doing on their screen.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "elements": [],
            "text": [],
            "recognized_apps": [],
            "active_window": None
        }
        
        try:
            # 1. Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Extract text using OCR
            text = pytesseract.image_to_string(Image.fromarray(rgb_image))
            if text:
                context["text"] = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 3. Detect UI elements using edge detection and contour analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours to identify UI elements
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify the element
                element_type = self._classify_ui_element(rgb_image[y:y+h, x:x+w])
                
                # Add to elements list
                context["elements"].append({
                    "id": f"element_{i}",
                    "type": element_type,
                    "bounds": {"x": x, "y": y, "width": w, "height": h}
                })
            
            # 4. Recognize applications using the vision analyzer
            # Convert image to PIL format for the vision model
            pil_image = Image.fromarray(rgb_image)
            
            # Get predictions
            predictions = self.vision_analyzer(pil_image)
            
            # Extract recognized applications
            for prediction in predictions:
                if prediction["score"] > 0.7:  # Confidence threshold
                    context["recognized_apps"].append({
                        "name": prediction["label"],
                        "confidence": prediction["score"]
                    })
            
            # 5. Determine active window based on UI analysis
            if context["elements"]:
                # Heuristic: The largest element near the top of the screen is likely the active window
                top_elements = sorted(context["elements"], key=lambda e: e["bounds"]["y"])[:5]
                largest_element = max(top_elements, key=lambda e: e["bounds"]["width"] * e["bounds"]["height"])
                context["active_window"] = largest_element["id"]
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting screen context: {e}")
            return context
    
    def _classify_ui_element(self, element_image) -> str:
        """Classify a UI element based on its appearance."""
        # This is a simplified version - in a real implementation, 
        # we would use a trained model for more accurate classification
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple heuristic classification
        if aspect_ratio > 5:
            return "menu_bar"
        elif aspect_ratio > 3:
            return "toolbar"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif w > 300 and h > 200:
            if avg_brightness > 200:
                return "dialog"
            else:
                return "window"
        elif w < 100 and h < 50:
            if avg_brightness > 200:
                return "button"
            else:
                return "icon"
        elif w > 200 and h < 50:
            return "text_field"
        else:
            return "panel"
    
    def _should_provide_assistance(self, session_id: str, screen_context: Dict[str, Any]) -> bool:
        """
        Determine if proactive assistance should be provided based on screen context.
        
        This is a revolutionary feature that allows the AI to offer help exactly
        when needed without explicit requests.
        """
        if session_id not in self.active_sessions:
            return False
        
        context_window = self.active_sessions[session_id]
        previous_context = context_window.screen_context
        
        # If no previous context, don't provide assistance yet
        if not previous_context:
            return False
        
        # Check for significant changes in active window
        if (previous_context.get("active_window") != screen_context.get("active_window") and
            screen_context.get("active_window") is not None):
            return True
        
        # Check for error messages in text
        error_patterns = ["error", "exception", "failed", "warning", "invalid"]
        for text in screen_context.get("text", []):
            if any(pattern in text.lower() for pattern in error_patterns):
                return True
        
        # Check for complex UI with many elements (user might need help navigating)
        if len(screen_context.get("elements", [])) > 15:
            # But only if this is a change from before
            if len(previous_context.get("elements", [])) < 10:
                return True
        
        # Check for recognized applications that might need assistance
        assistance_apps = ["terminal", "code editor", "database", "configuration"]
        for app in screen_context.get("recognized_apps", []):
            if any(assist_app in app["name"].lower() for assist_app in assistance_apps):
                # Only provide assistance if this is a newly detected app
                previous_apps = [a["name"] for a in previous_context.get("recognized_apps", [])]
                if app["name"] not in previous_apps:
                    return True
        
        return False
    
    async def _provide_proactive_assistance(self, session_id: str, screen_context: Dict[str, Any]):
        """
        Provide proactive assistance based on screen context.
        
        This is where the AI becomes truly collaborative by offering
        timely and relevant help without explicit requests.
        """
        if session_id not in self.active_sessions:
            return
        
        context_window = self.active_sessions[session_id]
        
        # Generate assistance message based on screen context
        assistance_message = await self._generate_assistance_message(session_id, screen_context)
        
        # Send the assistance message
        if assistance_message:
            await self.send_assistant_message(
                session_id,
                f"I noticed you might need help: {assistance_message}\n\nWould you like me to assist with this?"
            )
    
    async def _generate_assistance_message(self, session_id: str, screen_context: Dict[str, Any]) -> str:
        """Generate a contextual assistance message."""
        # Extract key information from screen context
        active_app = None
        if screen_context.get("recognized_apps"):
            active_app = screen_context["recognized_apps"][0]["name"]
        
        error_text = None
        for text in screen_context.get("text", []):
            error_patterns = ["error", "exception", "failed", "warning", "invalid"]
            if any(pattern in text.lower() for pattern in error_patterns):
                error_text = text
                break
        
        # Generate appropriate assistance
        if error_text:
            # Search knowledge base for similar errors
            similar_knowledge = await self._search_knowledge_base(error_text)
            if similar_knowledge:
                return f"I see an error: '{error_text}'. Based on my knowledge, this might be related to: {similar_knowledge}"
            else:
                return f"I notice you're encountering an error: '{error_text}'. Would you like me to help troubleshoot this?"
        
        elif active_app:
            if "terminal" in active_app.lower():
                return "I see you're working in the terminal. I can help with command suggestions or explain command outputs."
            
            elif "code" in active_app.lower() or "editor" in active_app.lower():
                return "I notice you're coding. I can help with code suggestions, debugging, or explaining concepts."
            
            elif "browser" in active_app.lower():
                return "I see you're browsing. I can help search for information or explain concepts on the current page."
            
            elif "database" in active_app.lower():
                return "I notice you're working with a database. I can help with query optimization or data modeling."
        
        # Default assistance based on UI complexity
        element_count = len(screen_context.get("elements", []))
        if element_count > 15:
            return "I notice you're working with a complex interface. I can help navigate or explain functionality."
        
        return None
    
    async def _search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for relevant information."""
        if not self.knowledge_vectors or not self.embedding_model:
            return None
        
        # Generate embedding for the query
        query_embedding = self.embedding_model(query)
        query_vector = np.mean(query_embedding[0], axis=0)
        
        # Calculate similarity with all knowledge items
        similarities = []
        for i, vector in enumerate(self.knowledge_vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most similar item if similarity is above threshold
        if similarities and similarities[0][1] > 0.7:
            index = similarities[0][0]
            # Return a snippet of the knowledge text
            text = self.knowledge_texts[index]
            # Extract a relevant snippet (first 200 characters)
            return text[:200] + "..." if len(text) > 200 else text
        
        return None
    
    async def _handle_voice_input(self, session_id: str, text: str):
        """Handle voice input from the user."""
        if session_id not in self.active_sessions:
            logger.warning(f"Received voice input for unknown session {session_id}")
            return
        
        logger.info(f"Processing voice input for session {session_id}: {text}")
        
        # Add to context window
        context_window = self.active_sessions[session_id]
        context_window.add_message({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat(),
            "type": "voice"
        })
        
        # Check if it's a command
        command_match = re.match(r"^([\w]+)(?:\s+(.+))?$", text.strip().lower())
        if command_match and command_match.group(1) in self.capabilities:
       # Check if it's a command
        command_match = re.match(r"^([\w]+)(?:\s+(.+))?$", text.strip().lower())
        if command_match and command_match.group(1) in self.capabilities:
            command = command_match.group(1)
            args = command_match.group(2) or ""
            
            # Add to command queue
            self.command_queue.put((session_id, command, args))
            
            return
        
        # Process as natural language
        await self._process_natural_language(session_id, text)
    
    async def _process_natural_language(self, session_id: str, text: str):
        """
        Process natural language input from the user.
        
        This is where the AI understands and responds to conversational input,
        making the interaction feel natural and human-like.
        """
        if session_id not in self.active_sessions:
            return
        
        context_window = self.active_sessions[session_id]
        
        # Prepare context for the language model
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps users with their tasks. You can see their screen and understand their voice commands. Be concise, helpful, and proactive."}
        ]
        
        # Add recent conversation history
        for msg in context_window.messages[-10:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # Add screen context if available
        if context_window.screen_context:
            screen_summary = self._summarize_screen_context(context_window.screen_context)
            messages.append({"role": "system", "content": f"Current screen context: {screen_summary}"})
        
        try:
            # Generate response using the language model
            response = await self._generate_response(messages)
            
            # Send the response
            await self.send_assistant_message(session_id, response)
            
            # Check for actionable insights in the response
            await self._extract_and_execute_actions(session_id, response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await self.send_assistant_message(
                session_id,
                "I'm sorry, I encountered an error while processing your request. Please try again."
            )
    
    def _summarize_screen_context(self, screen_context: Dict[str, Any]) -> str:
        """Summarize screen context for inclusion in LLM prompt."""
        summary_parts = []
        
        # Add active window
        if screen_context.get("active_window"):
            active_element = next(
                (e for e in screen_context.get("elements", []) 
                 if e.get("id") == screen_context["active_window"]),
                None
            )
            if active_element:
                summary_parts.append(f"Active window: {active_element.get('type', 'window')}")
        
        # Add recognized applications
        if screen_context.get("recognized_apps"):
            apps = [app["name"] for app in screen_context["recognized_apps"][:3]]
            summary_parts.append(f"Applications: {', '.join(apps)}")
        
        # Add UI element summary
        if screen_context.get("elements"):
            element_types = {}
            for element in screen_context["elements"]:
                element_type = element.get("type", "unknown")
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            element_summary = ", ".join(f"{count} {element_type}s" for element_type, count in element_types.items())
            summary_parts.append(f"UI elements: {element_summary}")
        
        # Add text summary (first few items)
        if screen_context.get("text"):
            text_items = screen_context["text"][:3]
            if text_items:
                text_summary = "; ".join(text_items)
                if len(text_summary) > 100:
                    text_summary = text_summary[:100] + "..."
                summary_parts.append(f"Visible text: {text_summary}")
        
        return " | ".join(summary_parts)
    
    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the language model."""
        if not self.text_generator or not self.tokenizer:
            return "I'm still initializing my language capabilities. Please try again in a moment."
        
        try:
            # Convert messages to a prompt
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.text_generator.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.text_generator.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            
            # Remove any assistant prefix
            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages to a prompt for the language model."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _extract_and_execute_actions(self, session_id: str, response: str):
        """
        Extract and execute actions from the assistant's response.
        
        This enables the AI to not just talk about actions but actually perform them,
        making it a true collaborative partner.
        """
        # Look for action patterns in the response
        action_patterns = [
            (r"I'll search for\s+(.+?)[\.\n]", "search"),
            (r"I'll analyze\s+(.+?)[\.\n]", "analyze_data"),
            (r"I'll create\s+(.+?)[\.\n]", "create"),
            (r"I'll generate\s+(.+?)[\.\n]", "generate_code"),
            (r"I'll show you\s+(.+?)[\.\n]", "visualize_data"),
            (r"I'll monitor\s+(.+?)[\.\n]", "monitor_system")
        ]
        
        for pattern, action in action_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                # Add to command queue
                self.command_queue.put((session_id, action, match))
                
                logger.info(f"Extracted action from response: {action} {match}")
    
    async def send_assistant_message(self, session_id: str, content: str):
        """Send a message from the assistant to the user."""
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot send message to unknown session {session_id}")
            return
        
        # Create message
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "text"
        }
        
        # Add to context window
        self.active_sessions[session_id].add_message(message)
        
        # Publish message event
        await self.event_bus.publish("assistant.message", {
            "session_id": session_id,
            "message": message
        })
        
        logger.info(f"Sent assistant message to session {session_id}: {content[:50]}...")
    
    async def end_session(self, session_id: str):
        """End a collaboration session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot end unknown session {session_id}")
            return False
        
        # Stop voice stream
        if session_id in self.voice_streams:
            try:
                self.voice_streams[session_id]["stream"].stop_stream()
                self.voice_streams[session_id]["stream"].close()
                self.voice_streams[session_id]["pyaudio"].terminate()
                del self.voice_streams[session_id]
            except Exception as e:
                logger.error(f"Error stopping voice stream: {e}")
        
        # Stop screen stream
        if session_id in self.screen_streams:
            try:
                self.screen_streams[session_id]["active"] = False
                del self.screen_streams[session_id]
            except Exception as e:
                logger.error(f"Error stopping screen stream: {e}")
        
        # Remove session
        del self.active_sessions[session_id]
        
        logger.info(f"Ended collaboration session {session_id}")
        return True
    
    # Command handlers
    def _process_voice_command(self, session_id: str, args: str) -> str:
        """Process a voice command."""
        return f"Processing voice command: {args}"
    
    def _analyze_screen(self, session_id: str, args: str) -> str:
        """Analyze the current screen."""
        if session_id not in self.screen_streams or not self.screen_streams[session_id].get("last_image") is not None:
            return "No screen capture available to analyze."
        
        # Get the last captured image
        image = self.screen_streams[session_id]["last_image"]
        
        # Extract context
        context = self._extract_screen_context(image)
        
        # Generate a human-readable analysis
        analysis = []
        
        if context.get("recognized_apps"):
            apps = [f"{app['name']} ({app['confidence']:.2f})" for app in context["recognized_apps"]]
            analysis.append(f"Recognized applications: {', '.join(apps)}")
        
        if context.get("elements"):
            element_types = {}
            for element in context["elements"]:
                element_type = element.get("type", "unknown")
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            element_summary = ", ".join(f"{count} {element_type}s" for element_type, count in element_types.items())
            analysis.append(f"UI elements: {element_summary}")
        
        if context.get("text"):
            text_count = len(context["text"])
            analysis.append(f"Detected {text_count} text elements")
            
            # Include a few examples
            if text_count > 0:
                examples = context["text"][:3]
                analysis.append(f"Text examples: {'; '.join(examples)}")
        
        return "\n".join(analysis)
    
    def _automate_task(self, session_id: str, args: str) -> str:
        """Automate a task based on screen context."""
        return f"Automating task: {args}"
    
    def _retrieve_information(self, session_id: str, args: str) -> str:
        """Retrieve information from knowledge base."""
        # Search knowledge base
        result = asyncio.run(self._search_knowledge_base(args))
        
        if result:
            return f"Found relevant information: {result}"
        else:
            return f"No relevant information found for: {args}"
    
    def _generate_code(self, session_id: str, args: str) -> str:
        """Generate code based on description."""
        return f"Generating code for: {args}"
    
    def _visualize_data(self, session_id: str, args: str) -> str:
        """Visualize data."""
        return f"Visualizing data: {args}"
    
    def _monitor_system(self, session_id: str, args: str) -> str:
        """Monitor system metrics."""
        return f"Monitoring system: {args}"
    
    def _provide_predictive_assistance(self, session_id: str, args: str) -> str:
        """Provide predictive assistance based on user patterns."""
        return f"Providing predictive assistance: {args}"
    
    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences."""
        # In a real implementation, this would load from a database
        return {
            "voice_enabled": True,
            "screen_capture_interval": 1.0,
            "proactive_assistance": True,
            "preferred_communication_style": "conversational"
        }
    
    async def _handle_user_joined(self, event_data: Dict[str, Any]):
        """Handle user joined event."""
        user_id = event_data.get("user_id")
        if not user_id:
            return
        
        # Start a session for the user
        session_id = await self.start_session(user_id)
        
        # Publish session started event
        await self.event_bus.publish("assistant.session_started", {
            "user_id": user_id,
            "session_id": session_id
        })
    
    async def _handle_user_left(self, event_data: Dict[str, Any]):
        """Handle user left event."""
        user_id = event_data.get("user_id")
        session_id = event_data.get("session_id")
        
        if not user_id or not session_id:
            return
        
        # End the session
        await self.end_session(session_id)
    
    async def _handle_user_message(self, event_data: Dict[str, Any]):
        """Handle user message event."""
        session_id = event_data.get("session_id")
        message = event_data.get("message")
        
        if not session_id or not message:
            return
        
        # Add to context window
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_message(message)
        
        # Process the message
        content = message.get("content", "")
        content_type = message.get("content_type", "text")
        
        if content_type == "text":
            await self._process_natural_language(session_id, content)
        elif content_type == "command":
            # Parse command
            parts = content.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Add to command queue
            self.command_queue.put((session_id, command, args))
    
    async def _handle_system_state_changed(self, event_data: Dict[str, Any]):
        """Handle system state changed event."""
        # Update system state for all active sessions
        for session_i
   async def _handle_system_state_changed(self, event_data: Dict[str, Any]):
        """Handle system state changed event."""
        # Update system state for all active sessions
        for session_id, context_window in self.active_sessions.items():
            # Update system state
            context_window.system_state.update(event_data)
            
            # Check if proactive notification is needed
            if self._should_notify_system_change(session_id, event_data):
                await self.send_assistant_message(
                    session_id,
                    f"System update: {self._format_system_change(event_data)}"
                )
    
    def _should_notify_system_change(self, session_id: str, event_data: Dict[str, Any]) -> bool:
        """Determine if a system change should trigger a notification."""
        # Check user preferences
        if session_id in self.active_sessions:
            user_preferences = self.active_sessions[session_id].user_preferences
            if not user_preferences.get("proactive_assistance", True):
                return False
        
        # Check importance of the change
        importance = event_data.get("importance", "low")
        if importance == "high":
            return True
        elif importance == "medium":
            # Only notify if the user isn't actively engaged
            if session_id in self.active_sessions:
                # Check if there was recent user activity (within last 5 minutes)
                recent_messages = [
                    msg for msg in self.active_sessions[session_id].messages[-10:]
                    if msg.get("role") == "user"
                ]
                
                if recent_messages:
                    last_message_time = datetime.fromisoformat(recent_messages[-1].get("timestamp", ""))
                    now = datetime.now()
                    time_diff = (now - last_message_time).total_seconds()
                    
                    # If user was active in the last 5 minutes, don't interrupt
                    if time_diff < 300:
                        return False
            
            return True
        
        # Low importance changes don't trigger notifications
        return False
    
    def _format_system_change(self, event_data: Dict[str, Any]) -> str:
        """Format a system change event for user notification."""
        event_type = event_data.get("type", "unknown")
        
        if event_type == "resource_warning":
            resource = event_data.get("resource", "unknown")
            level = event_data.get("level", "warning")
            details = event_data.get("details", "")
            return f"{level.upper()}: {resource} resource issue. {details}"
        
        elif event_type == "task_completed":
            task = event_data.get("task", "unknown")
            result = event_data.get("result", "completed")
            return f"Task '{task}' has been completed with result: {result}"
        
        elif event_type == "security_alert":
            alert = event_data.get("alert", "unknown")
            severity = event_data.get("severity", "medium")
            return f"{severity.upper()} security alert: {alert}"
        
        elif event_type == "update_available":
            component = event_data.get("component", "system")
            version = event_data.get("version", "unknown")
            return f"Update available for {component}: version {version}"
        
        else:
            # Generic formatting for unknown event types
            return ", ".join(f"{k}: {v}" for k, v in event_data.items() if k != "type")

class VoiceProcessor:
    """
    Voice processor for continuous speech recognition.
    
    This component enables the revolutionary natural voice interaction
    with the AI assistant.
    """
    
    def __init__(self, session_id: str, callback):
        """
        Initialize the voice processor.
        
        Args:
            session_id: Session ID
            callback: Callback function to call with transcribed text
        """
        self.session_id = session_id
        self.callback = callback
        self.active = False
        self.thread = None
        self.audio_queue = queue.Queue()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize Whisper model (will be loaded in the thread)
        self.model = None
    
    def start(self):
        """Start the voice processor."""
        if self.active:
            return
        
        self.active = True
        
        # Start processing thread
        self.thread = threading.Thread(target=self._processing_thread)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started voice processor for session {self.session_id}")
    
    def stop(self):
        """Stop the voice processor."""
        if not self.active:
            return
        
        self.active = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        # Clean up PyAudio
        self.p.terminate()
        self.p = None
        
        logger.info(f"Stopped voice processor for session {self.session_id}")
    
    def _processing_thread(self):
        """Processing thread for voice recognition."""
        try:
            # Load Whisper model
            self.model = whisper.load_model("base")
            
            # Open audio stream
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            stream.start_stream()
            
            # Process audio chunks
            while self.active:
                try:
                    # Get audio chunk from queue with timeout
                    audio_data = self.audio_queue.get(timeout=0.5)
                    
                    # Save to temporary file
                    with wave.open("temp_voice.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)
                    
                    # Transcribe
                    result = self.model.transcribe("temp_voice.wav")
                    text = result["text"].strip()
                    
                    # Call callback if text is not empty
                    if text:
                        self.callback(self.session_id, text)
                    
                    # Clean up
                    if os.path.exists("temp_voice.wav"):
                        os.remove("temp_voice.wav")
                    
                except queue.Empty:
                    # No audio data, continue
                    continue
                
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error in voice processing thread: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status_flags):
        """Callback for audio stream."""
        if self.active:
            # Add audio data to queue
            self.audio_queue.put(in_data)
        
        return (in_data, pyaudio.paContinue)

class ScreenAnalyzer:
    """
    Screen analyzer for understanding user's visual context.
    
    This component enables the revolutionary screen understanding
    capabilities of the AI assistant.
    """
    
    def __init__(self, vision_model, ocr_engine=None):
        """
        Initialize the screen analyzer.
        
        Args:
            vision_model: Vision model for image analysis
            ocr_engine: OCR engine for text extraction
        """
        self.vision_model = vision_model
        self.ocr_engine = ocr_engine or pytesseract
        
        # Initialize element classifiers
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize UI element classifiers."""
        # In a real implementation, this would load trained models
        # for UI element classification
        pass
    
    def analyze_screen(self, image) -> Dict[str, Any]:
        """
        Analyze a screen capture.
        
        Args:
            image: Screen capture image
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "elements": [],
            "text": [],
            "recognized_apps": [],
            "active_window": None
        }
        
        try:
            # 1. Convert to RGB for processing
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Extract text using OCR
            text = self.ocr_engine.image_to_string(Image.fromarray(rgb_image))
            if text:
                results["text"] = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 3. Detect UI elements
            elements = self._detect_ui_elements(rgb_image)
            results["elements"] = elements
            
            # 4. Recognize applications
            apps = self._recognize_applications(rgb_image)
            results["recognized_apps"] = apps
            
            # 5. Determine active window
            active_window = self._determine_active_window(elements)
            if active_window:
                results["active_window"] = active_window
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing screen: {e}")
            return results
    
    def _detect_ui_elements(self, image) -> List[Dict[str, Any]]:
        """Detect UI elements in the image."""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract element image
                element_image = image[y:y+h, x:x+w]
                
                # Classify element
                element_type = self._classify_element(element_image)
                
                # Add to elements list
                elements.append({
                    "id": f"element_{i}",
                    "type": element_type,
                    "bounds": {"x": x, "y": y, "width": w, "height": h}
                })
            
            return elements
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
            return elements
    
    def _classify_element(self, element_image) -> str:
        """Classify a UI element."""
        # This is a simplified version - in a real implementation, 
        # we would use a trained model for more accurate classification
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple heuristic classification
        if aspect_ratio > 5:
            return "menu_bar"
        elif aspect_ratio > 3:
            return "toolbar"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif w > 300 and h > 200:
            if avg_brightness > 200:
                return "dialog"
            else:
                return "window"
        elif w < 100 and h < 50:
            if avg_brightness > 200:
                return "button"
            else:
                return "icon"
        elif w > 200 and h < 50:
            return "text_field"
        else:
            return "panel"
    
    def _recognize_applications(self, image) -> List[Dict[str, Any]]:
        """Recognize applications in the image."""
        apps = []
        
        try:
            # Convert to PIL image for the vision model
            pil_image = Image.fromarray(image)
            
            # Get predictions from vision model
            predictions = self.vision_model(pil_image)
            
            # Process predictions
            for prediction in predictions:
                if prediction["score"] > 0.7:  # Confidence threshold
                    apps.append({
                        "name": prediction["label"],
                        "confidence": prediction["score"]
                    })
            
            return apps
            
        except Exception as e:
            logger.error(f"Error recognizing applications: {e}")
            return apps
    
    def _determine_active_window(self, elements: List[Dict[str, Any]]) -> Optional[str]:
        """Determine the active window from detected elements."""
        if not elements:
            return None
        
        # Heuristic: The largest element near the top of the screen is likely the active window
        window_elements = [e for e in elements if e["type"] in ["window", "dialog"]]
        
        if not window_elements:
            return None
        
        # Sort by y-coordinate (top to bottom)
        top_elements = sorted(window_elements, key=lambda e: e["bounds"]["y"])[:5]
        
        # Get the largest element
        largest_element = max(top_elements, key=lambda e: e["bounds"]["width"] * e["bounds"]["height"])
        
        return largest_element["id"]
class KnowledgeManager:
    """
    Knowledge manager for storing and retrieving information.
    
    This component enables the revolutionary knowledge capabilities
    of the AI assistant, allowing it to learn and adapt over time.
    """
    
    def __init__(self, embedding_model, storage_dir: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            embedding_model: Model for generating embeddings
            storage_dir: Directory for storing knowledge
        """
        self.embedding_model = embedding_model
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base
        self.knowledge_texts = []
        self.knowledge_vectors = []
        self.knowledge_metadata = []
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        # Load existing knowledge
        await self.load_knowledge()
    
    async def load_knowledge(self):
        """Load knowledge from storage."""
        # Clear existing knowledge
        self.knowledge_texts = []
        self.knowledge_vectors = []
        self.knowledge_metadata = []
        
        # Load knowledge index if it exists
        index_path = self.storage_dir / "knowledge_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                
                # Load knowledge items
                for item in index:
                    item_path = self.storage_dir / item["file"]
                    if item_path.exists():
                        with open(item_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        
                        # Load vector
                        vector_path = self.storage_dir / item["vector_file"]
                        if vector_path.exists():
                            vector = np.load(vector_path)
                            
                            # Add to knowledge base
                            self.knowledge_texts.append(text)
                            self.knowledge_vectors.append(vector)
                            self.knowledge_metadata.append(item["metadata"])
            
            except Exception as e:
                logger.error(f"Error loading knowledge index: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge items")
    
    async def add_knowledge(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add knowledge to the knowledge base.
        
        Args:
            text: Knowledge text
            metadata: Metadata for the knowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_model(text)
            vector = np.mean(embedding[0], axis=0)
            
            # Generate unique ID
            knowledge_id = str(uuid.uuid4())
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {}
            
            metadata["id"] = knowledge_id
            metadata["timestamp"] = datetime.now().isoformat()
            
            # Save text
            text_file = f"knowledge_{knowledge_id}.txt"
            with open(self.storage_dir / text_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Save vector
            vector_file = f"vector_{knowledge_id}.npy"
            np.save(self.storage_dir / vector_file, vector)
            
            # Add to knowledge base
            self.knowledge_texts.append(text)
            self.knowledge_vectors.append(vector)
            self.knowledge_metadata.append(metadata)
            
            # Update index
            await self._update_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        if not self.knowledge_vectors:
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model(query)
            query_vector = np.mean(query_embedding[0], axis=0)
            
            # Calculate similarity with all knowledge items
            similarities = []
            for i, vector in enumerate(self.knowledge_vectors):
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k results
            results = []
            for i, similarity in similarities[:top_k]:
                if similarity > 0.5:  # Similarity threshold
                    results.append({
                        "text": self.knowledge_texts[i],
                        "similarity": float(similarity),
                        "metadata": self.knowledge_metadata[i]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def _update_index(self):
        """Update the knowledge index."""
        try:
            # Create index
            index = []
            for i, metadata in enumerate(self.knowledge_metadata):
                knowledge_id = metadata.get("id", str(uuid.uuid4()))
                index.append({
                    "id": knowledge_id,
                    "file": f"knowledge_{knowledge_id}.txt",
                    "vector_file": f"vector_{knowledge_id}.npy",
                    "metadata": metadata
                })
            
            # Save index
            with open(self.storage_dir / "knowledge_index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating knowledge index: {e}")

class TaskAutomator:
    """
    Task automator for automating user tasks.
    
    This component enables the revolutionary automation capabilities
    of the AI assistant, allowing it to perform tasks on behalf of the user.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the task automator.
        
        Args:
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus
        
        # Automation capabilities
        self.capabilities = {
            "open_application": self._open_application,
            "close_application": self._close_application,
            "click": self._click,
            "type_text": self._type_text,
            "copy_text": self._copy_text,
            "paste_text": self._paste_text,
            "save_file": self._save_file,
            "open_file": self._open_file,
            "search": self._search,
            "navigate_to": self._navigate_to
        }
    
    async def execute_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task_type: Type of task to execute
            params: Task parameters
            
        Returns:
            Task result
        """
        if task_type not in self.capabilities:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
        
        try:
            # Execute the task
            result = await self.capabilities[task_type](params)
            
            # Publish task completed event
            await self.event_bus.publish("task.completed", {
                "type": "task_completed",
                "task": task_type,
                "params": params,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error executing task {task_type}: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _open_application(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open an application."""
        app_name = params.get("name")
        if not app_name:
            raise ValueError("Application name is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to open the application
        
        return {
            "status": "opened",
            "app_name": app_name
        }
    
    async def _close_application(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Close an application."""
        app_name = params.get("name")
        if not app_name:
            raise ValueError("Application name is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to close the application
        
        return {
            "status": "closed",
            "app_name": app_name
        }
    
    async def _click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Click at a position or on an element."""
        # Check if we have coordinates
        if "x" in params and "y" in params:
            x = params.get("x")
            y = params.get("y")
            
            # In a real implementation, this would use platform-specific APIs
            # to click at the specified coordinates
            
            return {
                "status": "clicked",
                "position": {"x": x, "y": y}
            }
        
        # Check if we have an element
        elif "element_id" in params:
            element_id = params.get("element_id")
            
            # In a real implementation, this would use platform-specific APIs
            # to click on the specified element
            
            return {
                "status": "clicked",
                "element_id": element_id
            }
        
        else:
            raise ValueError("Either coordinates (x, y) or element_id is required")
    
    async def _type_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Type text."""
        text = params.get("text")
        if not text:
            raise ValueError("Text is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to type the specified text
        
        return {
            "status": "typed",
            "text": text
        }
    
    async def _copy_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Copy text to clipboard."""
        text = params.get("text")
        if not text:
            raise ValueError("Text is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to copy the specified text to the clipboard
        
        return {
            "status": "copied",
            "text": text
        }
    
    async def _paste_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Paste text from clipboard."""
        # In a real implementation, this would use platform-specific APIs
        # to paste text from the clipboard
        
        return {
            "status": "pasted"
        }
    
    async def _save_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Save a file."""
        path = params.get("path")
        content = params.get("content")
        
        if not path:
            raise ValueError("File path is required")
        
        if content is None:
            raise ValueError("File content is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to save the file
        
        return {
            "status": "saved",
            "path": path
        }
    
    async def _open_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a file."""
        path = params.get("path")
        if not path:
            raise ValueError("File path is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to open the file
        
        return {
            "status": "opened",
            "path": path
        }
    
    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a search."""
        query = params.get("query")
        if not query:
            raise ValueError("Search query is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to perform the search
        
        return {
            "status": "searched",
            "query": query
        }
    
    async def _navigate_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a URL or location."""
        url = params.get("url")
        if not url:
            raise ValueError("URL is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to navigate to the URL
        
        return {
            "status": "navigated",
            "url": url
        }

class UserModelManager:
    """
    User model manager for tracking user preferences and behavior.
    
    This component enables the revolutionary personalization capabilities
    of the AI assistant, allowing it to adapt to each user's unique needs.
    """
    
    def __init__(self, storage_dir: str = "./user_models"):
        """
        Initialize the user model manager.
        
        Args:
            storage_dir: Directory for storing user models
        """
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # User models
        self.user_models = {}
    
    async def initialize(self):
        """Initialize the user model manager."""
        # Load existing user models
        await self.load_user_models()
    
    async def load_user_models(self):
        """Load user models from storage."""
        # Clear existing models
        self.user_models = {}
        
        # Load user models
        for file_path in self.storage_dir.glob("user_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    user_model = json.load(f)
                
                user_id = user_model.get("user_id")
                if user_id:
                    self.user_models[user_id] = user_model
            
            except Exception as e:
                logger.error(f"Error loading user model {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.user_models)} user models")
    
    async def get_user_model(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model
        """
        # Return existing model if available
        if user_id in self.user_models:
            return self.user_models[user_id]
        
        # Create new model
        user_model = {
            "user_id": user_id,
            "preferences": {
                "voice_enabled": True,
                "screen_capture_interval": 1.0,
                "proactive_assistance": True,
                "preferred_communication_style": "conversational"
            },
            "behavior_patterns": {},
            "interaction_history": [],
            "created_at": datetime.now().isoformat(),
            "
           "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save the model
        self.user_models[user_id] = user_model
        await self._save_user_model(user_id)
        
        return user_model
    
    async def update_user_preference(self, user_id: str, preference: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            preference: Preference name
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        # Get user model
        user_model = await self.get_user_model(user_id)
        
        # Update preference
        user_model["preferences"][preference] = value
        user_model["updated_at"] = datetime.now().isoformat()
        
        # Save the model
        return await self._save_user_model(user_id)
    
    async def record_interaction(self, user_id: str, interaction_type: str, details: Dict[str, Any]) -> bool:
        """
        Record a user interaction.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            details: Interaction details
            
        Returns:
            True if successful, False otherwise
        """
        # Get user model
        user_model = await self.get_user_model(user_id)
        
        # Add interaction to history
        interaction = {
            "type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        user_model["interaction_history"].append(interaction)
        
        # Limit history size
        if len(user_model["interaction_history"]) > 1000:
            user_model["interaction_history"] = user_model["interaction_history"][-1000:]
        
        # Update behavior patterns
        await self._update_behavior_patterns(user_model)
        
        # Save the model
        return await self._save_user_model(user_id)
    
    async def _update_behavior_patterns(self, user_model: Dict[str, Any]):
        """Update behavior patterns based on interaction history."""
        # Extract interaction history
        interactions = user_model["interaction_history"]
        
        # Skip if not enough interactions
        if len(interactions) < 10:
            return
        
        # Update patterns
        patterns = {}
        
        # 1. Preferred interaction times
        hour_counts = {}
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction["timestamp"])
            hour = timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours
        total_interactions = len(interactions)
        peak_hours = [hour for hour, count in hour_counts.items() if count > total_interactions * 0.1]
        patterns["preferred_hours"] = peak_hours
        
        # 2. Preferred interaction types
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction["type"]
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        # Find preferred types
        preferred_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        patterns["preferred_types"] = [t[0] for t in preferred_types]
        
        # 3. Response patterns
        response_times = []
        for i in range(1, len(interactions)):
            if interactions[i]["type"] == "assistant_message" and interactions[i-1]["type"] == "user_message":
                prev_time = datetime.fromisoformat(interactions[i-1]["timestamp"])
                curr_time = datetime.fromisoformat(interactions[i]["timestamp"])
                response_time = (curr_time - prev_time).total_seconds()
                response_times.append(response_time)
        
        if response_times:
            patterns["avg_response_time"] = sum(response_times) / len(response_times)
        
        # 4. Common queries
        queries = [
            interaction["details"].get("content", "")
            for interaction in interactions
            if interaction["type"] == "user_message"
        ]
        
        # Extract common words
        word_counts = {}
        for query in queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns["common_words"] = [w[0] for w in common_words]
        
        # Update user model
        user_model["behavior_patterns"] = patterns
    
    async def _save_user_model(self, user_id: str) -> bool:
        """Save a user model to storage."""
        if user_id not in self.user_models:
            return False
        
        try:
            # Get user model
            user_model = self.user_models[user_id]
            
            # Save to file
            file_path = self.storage_dir / f"user_{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_model, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user model for {user_id}: {e}")
            return False

class PredictiveEngine:
    """
    Predictive engine for anticipating user needs.
    
    This component enables the revolutionary predictive capabilities
    of the AI assistant, allowing it to anticipate user needs before
    they are explicitly expressed.
    """
    
    def __init__(self, user_model_manager: UserModelManager, knowledge_manager: KnowledgeManager):
        """
        Initialize the predictive engine.
        
        Args:
            user_model_manager: User model manager
            knowledge_manager: Knowledge manager
        """
        self.user_model_manager = user_model_manager
        self.knowledge_manager = knowledge_manager
    
    async def predict_user_needs(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict user needs based on context and user model.
        
        Args:
            user_id: User ID
            context: Current context
            
        Returns:
            List of predicted needs
        """
        # Get user model
        user_model = await self.user_model_manager.get_user_model(user_id)
        
        # Extract relevant information
        current_time = datetime.now()
        screen_context = context.get("screen_context", {})
        recent_messages = context.get("recent_messages", [])
        
        # Predictions
        predictions = []
        
        # 1. Time-based predictions
        time_predictions = await self._predict_time_based_needs(user_model, current_time)
        predictions.extend(time_predictions)
        
        # 2. Context-based predictions
        context_predictions = await self._predict_context_based_needs(user_model, screen_context)
        predictions.extend(context_predictions)
        
        # 3. Conversation-based predictions
        conversation_predictions = await self._predict_conversation_based_needs(user_model, recent_messages)
        predictions.extend(conversation_predictions)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predictions
    
    async def _predict_time_based_needs(self, user_model: Dict[str, Any], current_time: datetime) -> List[Dict[str, Any]]:
        """Predict needs based on time patterns."""
        predictions = []
        
        # Get behavior patterns
        patterns = user_model.get("behavior_patterns", {})
        preferred_hours = patterns.get("preferred_hours", [])
        
        # Check if current hour is a preferred hour
        current_hour = current_time.hour
        if current_hour in preferred_hours:
            # Predict based on common activities during this hour
            hour_interactions = [
                interaction for interaction in user_model.get("interaction_history", [])
                if datetime.fromisoformat(interaction["timestamp"]).hour == current_hour
            ]
            
            if hour_interactions:
                # Count interaction types
                type_counts = {}
                for interaction in hour_interactions:
                    interaction_type = interaction["type"]
                    type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
                
                # Find most common type
                most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                # Add prediction
                predictions.append({
                    "type": "time_based",
                    "need": f"typical_{most_common_type}",
                    "description": f"User typically performs {most_common_type} at this time",
                    "confidence": 0.7
                })
        
        return predictions
    
    async def _predict_context_based_needs(self, user_model: Dict[str, Any], screen_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict needs based on screen context."""
        predictions = []
        
        # Check for recognized applications
        recognized_apps = screen_context.get("recognized_apps", [])
        if recognized_apps:
            app_name = recognized_apps[0]["name"].lower()
            
            # Predict based on application
            if "code" in app_name or "editor" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "coding_assistance",
                    "description": "User might need help with coding",
                    "confidence": 0.8
                })
            
            elif "terminal" in app_name or "command" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "command_assistance",
                    "description": "User might need help with terminal commands",
                    "confidence": 0.8
                })
            
            elif "browser" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "information_search",
                    "description": "User might need help finding information",
                    "confidence": 0.7
                })
            
            elif "document" in app_name or "word" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "writing_assistance",
                    "description": "User might need help with writing",
                    "confidence": 0.7
                })
        
        # Check for error messages
        error_detected = False
        for text in screen_context.get("text", []):
            error_patterns = ["error", "exception", "failed", "warning", "invalid"]
            if any(pattern in text.lower() for pattern in error_patterns):
                error_detected = True
                break
        
        if error_detected:
            predictions.append({
                "type": "context_based",
                "need": "error_resolution",
                "description": "User might need help resolving an error",
                "confidence": 0.9
            })
        
        return predictions
    
    async def _predict_conversation_based_needs(self, user_model: Dict[str, Any], recent_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict needs based on recent conversation."""
        predictions = []
        
        # Skip if no recent messages
        if not recent_messages:
            return predictions
        
        # Extract user messages
        user_messages = [
            msg["content"] for msg in recent_messages
            if msg.get("role") == "user"
        ]
        
        # Skip if no user messages
        if not user_messages:
            return predictions
        
        # Analyze last user message
        last_message = user_messages[-1].lower()
        
        # Check for question patterns
        question_patterns = ["how", "what", "why", "when", "where", "can", "could", "would", "will", "?"]
        is_question = any(pattern in last_message for pattern in question_patterns)
        
        if is_question:
            # Search knowledge base for relevant information
            knowledge_results = await self.knowledge_manager.search_knowledge(last_message)
            
            if knowledge_results:
                predictions.append({
                    "type": "conversation_based",
                    "need": "knowledge_retrieval",
                    "description": "User is asking a question that might be answered from knowledge base",
                    "confidence": 0.8,
                    "knowledge": knowledge_results[0]
                })
        
        # Check for request patterns
        request_patterns = ["can you", "could you", "please", "help me", "i need", "show me"]
        is_request = any(pattern in last_message for pattern in request_patterns)
        
        if is_request:
            predictions.append({
                "type": "conversation_based",
                "need": "task_assistance",
                "description": "User is requesting assistance with a task",
                "confidence": 0.7
            })
        
        return predictions

class ContextWindow:
    """
    Context window for tracking conversation and context.
    
    This component enables the AI assistant to maintain context
    across interactions, providing a more coherent and personalized
    experience.
    """
    
    def __init__(self, user_id: str, session_id: str, max_messages: int = 100):
        """
        Initialize the context window.
        
        Args:
            user_id: User ID
            session_id: Session ID
            max_messages: Maximum number of messages to keep
        """
        self.user_id = user_id
        self.session_id = session_id
        self.max_messages = max_messages
        
        # Messages
        self.messages = []
        
        # Context
        self.screen_context = {}
        self.system_state = {}
        self.user_preferences = {}
    
    def add_message(self, message: Dict[str, Any]):
        """
        Add a message to the context window.
        
        Args:
            message: Message to add
        """
        # Add message
        self.messages.append(message)
        
        # Trim if necessary
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages.
        
        Args:
            count: Number of messages to get
            
        Returns:
            List of recent messages
        """
        return self.messages[-count:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.
        
        Returns:
            Context summary
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "recent_messages": self.get_recent_messages(5),
            "screen_context": self.screen_context,
            "system_state": self.system_state,
            "user_preferences": self.user_preferences
        }

# Main entry point
def create_cognitive_collaboration_system(config: Dict[str, Any] = None) -> CognitiveCollaborationSystem:
    """
    Create a cognitive collaboration system.
    
    Args:
        config: Configuration
   Create a cognitive collaboration system.
    
    Args:
        config: Configuration
        
    Returns:
        Cognitive collaboration system
    """
    # Default configuration
    default_config = {
        "model_path": "./models",
        "knowledge_dir": "./knowledge",
        "user_models_dir": "./user_models",
        "log_level": "INFO"
    }
    
    # Merge configurations
    if config is None:
        config = {}
    
    merged_config = {**default_config, **config}
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, merged_config["log_level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create event bus
    event_bus = EventBus()
    
    # Create system
    system = CognitiveCollaborationSystem(
        model_path=merged_config["model_path"],
        event_bus=event_bus
    )
    
    return system

# Example usage
if __name__ == "__main__":
    # Create system
    system = create_cognitive_collaboration_system()
    
    # Initialize system
    asyncio.run(system.initialize())
    
    # Start system
    asyncio.run(system.start())
    
    try:
        # Run forever
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        # Stop system
        asyncio.run(system.stop())
"""
Event bus for communication between components.

This module provides a simple event bus implementation for asynchronous
communication between components of the AI assistant.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class EventBus:
    """
    Event bus for communication between components.
    
    This class provides a simple publish-subscribe mechanism for
    asynchronous communication between components.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.event_history = {}
        self.max_history = 100
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        # Add to history
        if event_type not in self.event_history:
            self.event_history[event_type] = []
        
        self.event_history[event_type].append(event_data)
        
        # Trim history if necessary
        if len(self.event_history[event_type]) > self.max_history:
            self.event_history[event_type] = self.event_history[event_type][-self.max_history:]
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event subscriber for {event_type}: {e}")
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        """
        Subscribe to an event.
        
        Args:
            event_type: Type of event
            callback: Callback function
            
        Returns:
            Unsubscribe function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        
        # Return unsubscribe function
        def unsubscribe():
            if event_type in self.subscribers and callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
        
        return unsubscribe
    
    def get_recent_events(self, event_type: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events of a specific type.
        
        Args:
            event_type: Type of event
            count: Number of events to get
            
        Returns:
            List of recent events
        """
        if event_type not in self.event_history:
            return []
        
        return self.event_history[event_type][-count:]
"""
AI Assistant package for Edge-Native UMaaS.

This package provides the AI assistant capabilities for the Edge-Native
Universal Multimodal Assistant as a Service (UMaaS) platform.
"""

from .cognitive_collaboration_system import CognitiveCollaborationSystem, create_cognitive_collaboration_system
from .event_bus import EventBus

__all__ = [
    'CognitiveCollaborationSystem',
    'create_cognitive_collaboration_system',
    'EventBus'
]
#!/usr/bin/env python3
"""
Command-line interface for Edge-Native UMaaS.

This module provides a command-line interface for interacting with
the Edge-Native Universal Multimodal Assistant as a Service (UMaaS) platform.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any

from edgenative_umaas.ai_assistant import create_cognitive_collaboration_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class UMaaSCLI:
    """Command-line interface for UMaaS."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.system = None
        self.session_id = None
        self.user_id = os.environ.get("USER", "default_user")
    
    async def initialize(self, config_path: str = None):
        """
        Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Create system
        self.system = create_cognitive_collaboration_system(config)
        
        # Initialize system
        await self.system.initialize()
        
        # Start system
        await self.system.start()
        
        # Subscribe to assistant messages
        self.system.event_bus.subscribe("assistant.message", self._handle_assistant_message)
        
        # Start session
        self.session_id = await self.system.start_session(self.user_id)
        
        logger.info(f"Started session {self.session_id} for user {self.user_id}")
    
    async def _handle_assistant_message(self, event_data: Dict[str, Any]):
        """Handle assistant message event."""
        message = event_data.get("message", {})
        content = message.get("content", "")
        
        # Print message
        print(f"\nAssistant: {content}\n")
    
    async def process_command(self, command: str):
        """
        Process a command.
        
        Args:
            command: Command to process
        """
        if not self.system or not self.session_id:
            print("System not initialized")
            return
        
        if command.lower() in ["exit", "quit"]:
            # End session
            await self.system.end_session(self.session_id)
            
            # Stop system
            await self.system.stop()
            
            return False
        
        # Process command
        await self.system.process_user_input(self.session_id, command)
        
        return True
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("Edge-Native UMaaS CLI")
        print("Type 'exit' or 'quit' to exit")
        print()
        
        # Welcome message
        print("Assistant: Hello! I'm your AI assistant. How can I help you today?")
        
        # Main loop
        running = True
        while running:
            try:
                # Get user input
                user_input = input("\nYou: ")
                
                # Process command
                result = await self.process_command(user_input)
                if result is False:
                    running = False
            
            except KeyboardInterrupt:
                print("\nExiting...")
                
                # End session
                await self.system.end_session(self.session_id)
                
                # Stop system
                await self.system.stop()
                
                running = False
            
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(f"\nError: {e}")

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Edge-Native UMaaS CLI")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create CLI
    cli = UMaaSCLI()
    
    # Run
    try:
        # Initialize
        asyncio.run(cli.initialize(args.config))
        
        # Run in interactive mode
        asyncio.run(cli.interactive_mode())
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
from setuptools import setup, find_packages

setup(
    name="edgenative-umaas",
    version="0.1.0",
    description="Edge-Native Universal Multimodal Assistant as a Service",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "opencv-python",
        "pillow",
        "pytesseract",
        "pyaudio",
        "whisper",
    ],
    entry_points={
        "console_scripts": [
            "umaas-cli=edgenative_umaas.cli:main",
        ],
    },
    python_requires=">=3.8",
)
"""
Utility functions for the AI assistant.

This module provides utility functions used by various components
of the AI assistant.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False

def capture_screen() -> Optional[np.ndarray]:
    """
    Capture the screen.
    
    Returns:
        Screen image as numpy array, or None if failed
    """
    try:
        # This is a simplified implementation that works on most platforms
        # In a real implementation, we would use platform-specific APIs
        # for better performance and reliability
        
        # Try to use mss for fast screen capture
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # Primary monitor
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except ImportError:
            pass
        
        # Fall back to PIL and ImageGrab
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return np.array(img)
        except ImportError:
            pass
        
        logger.warning("No screen capture method available")
        return None
        
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return None

def save_image(image: np.ndarray, directory: str, prefix: str = "image") -> Optional[str]:
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array
        directory: Directory to save to
        prefix: Filename prefix
        
    Returns:
        Path to saved image, or None if failed
    """
    try:
        # Ensure directory exists
        if not ensure_directory(directory):
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
    """
    Crop an image.
    
    Args:
        image: Image as numpy array
        x: X coordinate
        y: Y coordinate
        width: Width
        height: Height
        
    Returns:
        Cropped image, or None if failed
    """
    try:
        # Check bounds
        img_height, img_width = image.shape[:2]
        
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            logger.warning(f"Crop region ({x}, {y}, {width}, {height}) out of bounds ({img_width}, {img_height})")
            return None
        
        # Crop image
        return image[y:y+height, x:x+width]
        
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return None

def resize_image(image: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """
    Resize an image.
    
    Args:
        image: Image as numpy array
        width: Width
        height: Height
        
    Returns:
        Resized image, or None if failed
    """
    try:
        return cv2.resize(image, (width, height))
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

def extract_text_from_image(image: np.ndarray, lang: str = "eng") -> List[str]:
    """
    Extract text from an image using OCR.
    
    Args:
        image: Image as numpy array
        lang: Language code
        
    Returns:
        List of extracted text lines
    """
    try:
        # Convert to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Extract text using pytesseract
        import pytesseract
        text = pytesseract.image_to_string(pil_image, lang=lang)
        
        # Split into lines and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return lines
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return []

def measure_execution_time(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper

def format_timestamp(timestamp: float) -> str:
    """
    Format a timestamp.
    
    Args:
        timestamp: Timestamp in seconds
        
    Returns:
        Formatted timestamp
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    try:
        # Split into words
        words = text.lower().split()
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "that", "this", "these", "those", "it", "they", 
                        "he", "she", "we", "you", "i", "me", "my", "your", "our", "their"}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    try:
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def parse_command(text: str) -> Tuple[str, str]:
    """
    Parse a command.
    
    Args:
        text: Command text
        
    Returns:
        Tuple of (command, arguments)
    """
    parts = text.strip().split(maxsplit=1)
    
    if len(parts) == 0:
        return "", ""
    
    if len(parts) == 1:
        return parts[0].lower(), ""
    
    return parts[0].lower(), parts[1]

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    
    hours = minutes / 60
    return f"{hours:.1f} hours"

def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.1f} GB"
"""
Model management for the AI assistant.

This module provides functionality for loading and managing AI models
used by the assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for loading and managing AI models.
    
    This class provides functionality for loading and managing AI models
    used by the assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Loaded models
        self.models = {}
        self.tokenizers = {}
    
    async def load_model(self, model_name: str, model_type: str = "causal_lm", device: str = None) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """
        Load a model.
        
        Args:
            model_name: Model name or path
            model_type: Model type (causal_lm or seq2seq_lm)
            device: Device to load model on (cpu, cuda, or None for auto)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if model is already loaded
        if model_name in self.models and model_name in self.tokenizers:
            return self.models[model_name], self.tokenizers[model_name]
        
        try:
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} on {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_name)
            elif model_type == "seq2seq_lm":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Move model to device
            model = model.to(device)
            
            # Save model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, None
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.models:
            return False
        
        try:
            # Remove model and tokenizer
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def generate_text(self, model_name: str, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using a model.
        
        Args:
            model_name: Model name
            prompt: Prompt text
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated text
        """
        if model_name not in self.models or model_name not in self.tokenizers:
            # Try to load the model
            model, tokenizer = await self.load_model(model_name)
            if model is None or tokenizer is None:
                return ""
        else:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate text
"""
Configuration management for the AI assistant.

This module provides functionality for loading and managing configuration
for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the AI assistant.
    
    This class provides functionality for loading and managing configuration
    for the AI assistant.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        
        # Default configuration
        self.default_config = {
            "model_path": "./models",
            "knowledge_dir": "./knowledge",
            "user_models_dir": "./user_models",
            "log_level": "INFO",
            "voice": {
                "enabled": True,
                "model": "base",
                "language": "en"
            },
            "screen_capture": {
                "enabled": True,
                "interval": 1.0
            },
            "models": {
                "assistant": "gpt2",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "vision": "facebook/detr-resnet-50"
            },
            "system": {
                "max_sessions": 10,
                "session_timeout": 3600,
                "max_messages": 100
            }
        }
    
    async def load_config(self) -> Dict[str, Any]:
        """
        Load configuration.
        
        Returns:
            Configuration dictionary
        """
        # Start with default configuration
        self.config = self.default_config.copy()
        
        # Load configuration from file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                
                # Merge configurations
                self._merge_config(self.config, file_config)
                
                logger.info(f"Loaded configuration from {self.config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")
        
        return self.config
    
    async def save_config(self) -> bool:
        """
        Save configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        # Split key into parts
        parts = key.split(".")
        
        # Navigate through configuration
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split key into parts
            parts = key.split(".")
            
            # Navigate through configuration
            config = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set value
            config[parts[-1]] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value for {key}: {e}")
            return False
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Merge two configuration dictionaries.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._merge_config(target[key], value)
            else:
                # Set value
                target[key] = value
   def _cleanup_sessions(self):
        """Clean up expired sessions."""
        # Find expired sessions
        expired_session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        
        # End expired sessions
        for session_id in expired_session_ids:
            self.end_session(session_id)
        
        if expired_session_ids:
            logger.info(f"Cleaned up {len(expired_session_ids)} expired sessions")
"""
Voice processing for the AI assistant.

This module provides functionality for speech recognition and synthesis
for the AI assistant.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for the AI assistant.
    
    This class provides functionality for speech recognition and synthesis
    for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the voice processor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Speech recognition model
        self.stt_model = None
        
        # Speech synthesis model
        self.tts_model = None
    
    async def initialize(self):
        """Initialize the voice processor."""
        try:
            # Initialize speech recognition
            await self._initialize_stt()
            
            # Initialize speech synthesis
            await self._initialize_tts()
            
        except Exception as e:
            logger.error(f"Error initializing voice processor: {e}")
    
    async def _initialize_stt(self):
        """Initialize speech recognition."""
        try:
            # Try to import whisper
            import whisper
            
            # Load model
            model_name = "base"
            logger.info(f"Loading Whisper model: {model_name}")
            self.stt_model = whisper.load_model(model_name)
            
            logger.info("Speech recognition initialized")
            
        except ImportError:
            logger.warning("Whisper not available, speech recognition disabled")
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    async def _initialize_tts(self):
        """Initialize speech synthesis."""
        try:
            # Try to import TTS
            from TTS.api import TTS
            
            # Load model
            logger.info("Loading TTS model")
            self.tts_model = TTS(gpu=torch.cuda.is_available())
            
            logger.info("Speech synthesis initialized")
            
        except ImportError:
            logger.warning("TTS not available, speech synthesis disabled")
        except Exception as e:
            logger.error(f"Error initializing speech synthesis: {e}")
    
    async def recognize_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            Recognition result
        """
        if self.stt_model is None:
            return {
                "success": False,
                "error": "Speech recognition not available"
            }
        
        try:
            # Recognize speech
            result = self.stt_model.transcribe(audio_data)
            
            return {
                "success": True,
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
            
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def synthesize_speech(self, text: str, language: str = "en") -> Tuple[Optional[np.ndarray], int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        if self.tts_model is None:
            return None, 0
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize speech
            self.tts_model.tts_to_file(
                text=text,
                file_path=temp_path,
                language=language
            )
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None, 0
    
    async def record_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], int]:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        try:
            # Try to import pyaudio
            import pyaudio
            import wave
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            # Record audio
            logger.info(f"Recording audio for {duration} seconds")
            frames = []
            for _ in range(0, int(sample_rate / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to file
            wf = wave.open(temp_path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except ImportError:
            logger.warning("PyAudio not available, audio recording disabled")
            return None, 0
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None, 0
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Play audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to import pyaudio
            import pyaudio
            import wave
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to file
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Open wave file
            wf = wave.open(temp_path, "rb")
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Play audio
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return True
            
        except ImportError:
            logger.warning("PyAudio not available, audio playback disabled")
            return False
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
"""
Vision processing for the AI assistant.

This module provides functionality for computer vision tasks
for the AI assistant.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import capture_screen, extract_text_from_image

# Configure logging
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Vision processor for the AI assistant.
    
    This class provides functionality for computer vision tasks
    for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the vision processor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Object detection model
        self.detection_model = None
        
        # Image classification model
        self.classification_model = None
    
    async def initialize(self):
        """Initialize the vision processor."""
        try:
            # Initialize object detection
            await self._initialize_detection()
            
            # Initialize image classification
            await self._initialize_classification()
            
        except Exception as e:
            logger.error(f"Error initializing vision processor: {e}")
    
    async def _initialize_detection(self):
        """Initialize object detection."""
        try:
            # Try to import transformers
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            # Load model
            model_name = "facebook/detr-resnet-50"
            logger.info(f"Loading object detection model: {model_name}")
            
            self.detection_processor = DetrImageProcessor.from_pretrained(model_name)
            self.detection_model = DetrForObjectDetection.from_pretrained(model_name)
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detection_model = self.detection_model.to(device)
            
            logger.info("Object detection initialized")
            
        except ImportError:
            logger.warning("Transformers not available, object detection disabled")
        except Exception as e:
            logger.error(f"Error initializing object detection: {e}")
    
    async def _initialize_classification(self):
        """Initialize image classification."""
        try:
            # Try to import transformers
            from transformers import ViTForImageClassification, ViTImageProcessor
            
            # Load model
            model_name = "google/vit-base-patch16-224"
            logger.info(f"Loading image classification model: {model_name}")
            
            self.classification_processor = ViTImageProcessor.from_pretrained(model_name)
            self.classification_model = ViTForImageClassification.from_pretrained(model_name)
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classification_model = self.classification_model.to(device)
            
            logger.info("Image classification initialized")
            
        except ImportError:
            logger.warning("Transformers not available, image classification disabled")
        except Exception as e:
            logger.error(f"Error initializing image classification: {e}")
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Analysis result
        """
        # Capture screen
        screen = capture_screen()
        if screen is None:
            return {
                "success": False,
                "error": "Failed to capture screen"
            }
        
        # Analyze screen
        result = await self.analyze_image(screen)
        
        # Add screen capture
        result["screen"] = screen
        
        return result
    
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Analysis result
        """
        result = {
            "success": True,
            "text": [],
            "objects": [],
            "classification": []
        }
        
        try:
            # Extract text
            text = extract_text_from_image(image)
            result["text"] = text
            
            # Detect objects
            if self.detection_model is not None:
                objects = await self._detect_objects(image)
                result["objects"] = objects
            
            # Classify image
            if self.classification_model is not None:
                classification = await self._classify_image(image)
                result["classification"] = classification
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected objects
        """
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare image for model
            inputs = self.detection_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.detection_model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.tensor([pil_image.size[::-1]])
            target_sizes = target_sizes.to(self.detection_model.device)
            results = self.detection_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )[0]
            
            # Extract results
            objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                objects.append({
                    "label": self.detection_model
                   "label": self.detection_model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": {
                        "x": box[0].item(),
                        "y": box[1].item(),
                        "width": box[2].item() - box[0].item(),
                        "height": box[3].item() - box[1].item()
                    }
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    async def _classify_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of classifications
        """
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare image for model
            inputs = self.classification_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.classification_model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Extract results
            classifications = []
            for prob, idx in zip(top5_prob, top5_indices):
                classifications.append({
                    "label": self.classification_model.config.id2label[idx.item()],
                    "score": prob.item()
                })
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return []
    
    async def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Extract results
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
            
            return face_list
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    async def recognize_app_windows(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize application windows in a screen image.
        
        Args:
            image: Screen image as numpy array
            
        Returns:
            List of recognized application windows
        """
        try:
            # This is a simplified implementation that uses basic image processing
            # In a real implementation, we would use platform-specific APIs
            # to get accurate window information
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            windows = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 100 or h < 100:
                    continue
                
                # Extract window region
                window_region = image[y:y+h, x:x+w]
                
                # Extract text from window title bar
                title_bar_region = window_region[:30, :]
                title_text = extract_text_from_image(title_bar_region)
                
                # Add window
                windows.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "title": title_text[0] if title_text else "Unknown"
                })
            
            return windows
            
        except Exception as e:
            logger.error(f"Error recognizing app windows: {e}")
            return []
    
    async def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected text regions
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            text_regions = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 10:
                    continue
                
                # Extract text region
                text_region = image[y:y+h, x:x+w]
                
                # Extract text
                text = extract_text_from_image(text_region)
                
                # Add text region
                if text:
                    text_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "text": text[0]
                    })
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    async def recognize_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize UI elements in a screen image.
        
        Args:
            image: Screen image as numpy array
            
        Returns:
            List of recognized UI elements
        """
        try:
            # This is a simplified implementation that uses basic image processing
            # In a real implementation, we would use more sophisticated techniques
            # or platform-specific APIs
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            ui_elements = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 10:
                    continue
                
                # Extract UI element region
                element_region = image[y:y+h, x:x+w]
                
                # Determine element type
                element_type = "unknown"
                
                # Check if it's a button
                if 20 <= w <= 200 and 20 <= h <= 50:
                    element_type = "button"
                
                # Check if it's a text field
                elif 100 <= w <= 400 and 20 <= h <= 40:
                    element_type = "text_field"
                
                # Check if it's a checkbox
                elif 10 <= w <= 30 and 10 <= h <= 30:
                    element_type = "checkbox"
                
                # Extract text
                text = extract_text_from_image(element_region)
                
                # Add UI element
                ui_elements.append({
                    "type": element_type,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "text": text[0] if text else ""
                })
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error recognizing UI elements: {e}")
            return []
"""
Knowledge management for the AI assistant.

This module provides functionality for managing knowledge
for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge manager for the AI assistant.
    
    This class provides functionality for managing knowledge
    for the AI assistant.
    """
    
    def __init__(self, knowledge_dir: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            knowledge_dir: Path to knowledge directory
        """
        self.knowledge_dir = Path(knowledge_dir)
        
        # Create knowledge directory if it doesn't exist
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base
        self.knowledge_base = {}
        
        # Embeddings
        self.embeddings = {}
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        try:
            # Load knowledge base
            await self.load_knowledge_base()
            
        except Exception as e:
            logger.error(f"Error initializing knowledge manager: {e}")
    
    async def load_knowledge_base(self) -> bool:
        """
        Load knowledge base from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear knowledge base
            self.knowledge_base = {}
            self.embeddings = {}
            
            # Load knowledge files
            for file_path in self.knowledge_dir.glob("*.json"):
                try:
                    # Load knowledge file
                    with open(file_path, "r", encoding="utf-8") as f:
                        knowledge = json.load(f)
                    
                    # Add to knowledge base
                    knowledge_id = file_path.stem
                    self.knowledge_base[knowledge_id] = knowledge
                    
                    # Load embedding if available
                    embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
                    if embedding_path.exists():
                        self.embeddings[knowledge_id] = np.load(embedding_path)
                    
                except Exception as e:
                    logger.error(f"Error loading knowledge file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    async def save_knowledge_base(self) -> bool:
        """
        Save knowledge base to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save knowledge files
            for knowledge_id, knowledge in self.knowledge_base.items():
                try:
                    # Save knowledge file
                    file_path = self.knowledge_dir / f"{knowledge_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(knowledge, f, indent=2)
                    
                    # Save embedding if available
                    if knowledge_id in self.embeddings:
                        embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
                        np.save(embedding_path, self.embeddings[knowledge_id])
                    
                except Exception as e:
                    logger.error(f"Error saving knowledge file {file_path}: {e}")
            
            logger.info(f"Saved {len(self.knowledge_base)} knowledge items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    async def add_knowledge(self, knowledge: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            embedding: Knowledge embedding
            
        Returns:
            Knowledge ID
        """
        try:
            # Generate knowledge ID
            import uuid
            knowledge_id = str(uuid.uuid4())
            
            # Add to knowledge base
            self.knowledge_base[knowledge_id] = knowledge
            
            # Add embedding if available
            if embedding is not None:
                self.embeddings[knowledge_id] = embedding
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        return self.knowledge_base.get(knowledge_id)
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            embedding: Updated knowledge embedding
            
        Returns:
            True if successful, False otherwise
        """
        if knowledge_id not in self.knowledge_base:
            return False
        
        try:
            # Update knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Update embedding if available
            if embedding is not None:
                self.embeddings[knowledge_id] = embedding
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge {knowledge_id}: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        
       Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        if knowledge_id not in self.knowledge_base:
            return False
        
        try:
            # Remove from knowledge base
            del self.knowledge_base[knowledge_id]
            
            # Remove embedding if available
            if knowledge_id in self.embeddings:
                del self.embeddings[knowledge_id]
            
            # Remove knowledge files
            file_path = self.knowledge_dir / f"{knowledge_id}.json"
            if file_path.exists():
                os.remove(file_path)
            
            embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
            if embedding_path.exists():
                os.remove(embedding_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge {knowledge_id}: {e}")
            return False
    
    async def search_knowledge(self, query: str, embedding: Optional[np.ndarray] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            embedding: Query embedding
            max_results: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        try:
            results = []
            
            # Search by embedding if available
            if embedding is not None and self.embeddings:
                # Calculate similarity scores
                scores = {}
                for knowledge_id, knowledge_embedding in self.embeddings.items():
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, knowledge_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(knowledge_embedding)
                    )
                    scores[knowledge_id] = similarity
                
                # Sort by similarity
                sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
                
                # Add top results
                for knowledge_id in sorted_ids[:max_results]:
                    knowledge = self.knowledge_base[knowledge_id]
                    results.append({
                        "id": knowledge_id,
                        "knowledge": knowledge,
                        "score": float(scores[knowledge_id])
                    })
            
            # Search by text if no results or no embedding
            if not results:
                # Convert query to lowercase
                query_lower = query.lower()
                
                # Search knowledge base
                for knowledge_id, knowledge in self.knowledge_base.items():
                    # Check if query matches knowledge
                    score = 0.0
                    
                    # Check title
                    if "title" in knowledge and query_lower in knowledge["title"].lower():
                        score += 0.8
                    
                    # Check content
                    if "content" in knowledge and query_lower in knowledge["content"].lower():
                        score += 0.5
                    
                    # Check tags
                    if "tags" in knowledge:
                        for tag in knowledge["tags"]:
                            if query_lower in tag.lower():
                                score += 0.3
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append({
                            "id": knowledge_id,
                            "knowledge": knowledge,
                            "score": score
                        })
                
                # Sort by score
                results = sorted(results, key=lambda r: r["score"], reverse=True)
                
                # Limit results
                results = results[:max_results]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge from the knowledge base.
        
        Returns:
            List of all knowledge items
        """
        return [
            {
                "id": knowledge_id,
                "knowledge": knowledge
            }
            for knowledge_id, knowledge in self.knowledge_base.items()
        ]
    
    async def import_knowledge(self, file_path: str) -> int:
        """
        Import knowledge from a file.
        
        Args:
            file_path: Path to knowledge file
            
        Returns:
            Number of imported knowledge items
        """
        try:
            # Check file extension
            if file_path.endswith(".json"):
                # Load JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check if it's a list or a single item
                if isinstance(data, list):
                    # Import multiple knowledge items
                    count = 0
                    for item in data:
                        await self.add_knowledge(item)
                        count += 1
                    
                    return count
                else:
                    # Import single knowledge item
                    await self.add_knowledge(data)
                    return 1
            
            elif file_path.endswith(".txt"):
                # Load text file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create knowledge item
                knowledge = {
                    "title": os.path.basename(file_path),
                    "content": content,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                }
                
                # Add knowledge
                await self.add_knowledge(knowledge)
                
                return 1
            
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return 0
            
        except Exception as e:
            logger.error(f"Error importing knowledge from {file_path}: {e}")
            return 0
    
    async def export_knowledge(self, file_path: str) -> bool:
        """
        Export knowledge to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all knowledge
            knowledge_items = [
                knowledge for knowledge_id, knowledge in self.knowledge_base.items()
            ]
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(knowledge_items, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge to {file_path}: {e}")
            return False
"""
User modeling for the AI assistant.

This module provides functionality for modeling user preferences
and behavior for the AI assistant.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class UserModel:
    """
    User model for the AI assistant.
    
    This class represents a model of a user's preferences and behavior.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the user model.
        
        Args:
            user_id: User ID
        """
        self.user_id = user_id
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        
        # User preferences
        self.preferences = {}
        
        # User interaction history
        self.interactions = []
        
        # User topics of interest
        self.topics = {}
    
    def update_preference(self, key: str, value: Any):
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.last_updated = datetime.now().isoformat()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        return self.preferences.get(key, default)
    
    def add_interaction(self, interaction_type: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a user interaction.
        
        Args:
            interaction_type: Type of interaction
            content: Interaction content
            metadata: Interaction metadata
        """
        # Create interaction
        interaction = {
            "type": interaction_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to interactions
        self.interactions.append(interaction)
        
        # Update last updated
        self.last_updated = datetime.now().isoformat()
        
        # Update topics of interest
        if interaction_type == "query":
            self._update_topics(content)
    
    def get_interactions(self, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        # Filter by type if specified
        if interaction_type:
            filtered = [i for i in self.interactions if i["type"] == interaction_type]
        else:
            filtered = self.interactions
        
        # Limit count if specified
        if count is not None:
            filtered = filtered[-count:]
        
        return filtered
    
    def get_top_topics(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top topics of interest.
        
        Args:
            count: Number of topics to get
            
        Returns:
            List of topics
        """
        # Sort topics by score
        sorted_topics = sorted(
            self.topics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to list of dictionaries
        return [
            {"topic": topic, "score": score}
            for topic, score in sorted_topics[:count]
        ]
    
    def _update_topics(self, content: str):
        """
        Update topics of interest based on content.
        
        Args:
            content: Content to analyze
        """
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Update topic scores
        for keyword in keywords:
            if keyword in self.topics:
                self.topics[keyword] += 1
            else:
                self.topics[keyword] = 1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Split into words
        words = text.lower().split()
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "that", "this", "these", "those", "it", "they", 
                        "he", "she", "we", "you", "i", "me", "my", "your", "our", "their"}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:10]]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the user model to a dictionary.
        
        Returns:
            User model dictionary
        """
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "preferences": self.preferences,
            "interactions": self.interactions,
            "topics": self.topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserModel':
        """
        Create a user model from a dictionary.
        
        Args:
            data: User model dictionary
            
        Returns:
            User model
        """
        user_model = cls(data["user_id"])
        user_model.created_at = data["created_at"]
        user_model.last_updated = data["last_updated"]
        user_model.preferences = data["preferences"]
        user_model.interactions = data["interactions"]
        user_model.topics = data["topics"]
        
        return user_model

class UserModelManager:
    """
    User model manager for the AI assistant.
    
    This class manages user models for the AI assistant.
    """
    
    def __init__(self, user_models_dir: str = "./user_models"):
        """
        Initialize the user model manager.
        
        Args:
            user_models_dir: Path to user models directory
        """
        self.user_models_dir = Path(user_models_dir)
        
        # Create user models directory if it doesn't exist
        self.user_models_dir.mkdir(parents=True, exist_ok=True)
        
        # User models
        self.user_models = {}
    
    async def initialize(self):
        """Initialize the user model manager."""
        try:
            # Load user models
            await self.load_user_models()
            
        except Exception as e:
            logger.error(f"Error initializing user model manager: {e}")
    
    async def load_user_models(self) -> bool:
        """
        Load user models from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear user models
            self.user_models = {}
            
            # Load user model files
            for file_path in self.user_models_dir.glob("*.json"):
                try:
                    # Load user model file
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Create user model
                    user_model = UserModel.from_dict(data)
                    
                    # Add to user models
                    self.user_models[user_model.user_id] = user_model
                    
                except Exception as e:
                    logger.error(f"Error loading user model file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.user_models)} user models")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading user models: {e}")
            return False
    
    async def save_user_model(self, user_id: str) -> bool:
        """
        Save a user model to disk.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.user_models:
            return False
        
        try:
            # Get user model
            user_model = self.user_models[user_id]
            
            # Save user model file
            file_path = self.user_models_dir / f"{user_i
           # Save user model file
            file_path = self.user_models_dir / f"{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_model.to_dict(), f, indent=2)
            
            logger.info(f"Saved user model for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user model for user {user_id}: {e}")
            return False
    
    async def save_all_user_models(self) -> bool:
        """
        Save all user models to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save each user model
            for user_id in self.user_models:
                await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving all user models: {e}")
            return False
    
    async def get_user_model(self, user_id: str) -> UserModel:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model
        """
        # Check if user model exists
        if user_id not in self.user_models:
            # Create new user model
            self.user_models[user_id] = UserModel(user_id)
        
        return self.user_models[user_id]
    
    async def delete_user_model(self, user_id: str) -> bool:
        """
        Delete a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.user_models:
            return False
        
        try:
            # Remove from user models
            del self.user_models[user_id]
            
            # Remove user model file
            file_path = self.user_models_dir / f"{user_id}.json"
            if file_path.exists():
                os.remove(file_path)
            
            logger.info(f"Deleted user model for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user model for user {user_id}: {e}")
            return False
    
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List of user IDs
        """
        return list(self.user_models.keys())
    
    async def add_user_interaction(self, user_id: str, interaction_type: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a user interaction.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            content: Interaction content
            metadata: Interaction metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Add interaction
            user_model.add_interaction(interaction_type, content, metadata)
            
            # Save user model
            await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding interaction for user {user_id}: {e}")
            return False
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Update preference
            user_model.update_preference(key, value)
            
            # Save user model
            await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference for user {user_id}: {e}")
            return False
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get preference
            return user_model.get_preference(key, default)
            
        except Exception as e:
            logger.error(f"Error getting preference for user {user_id}: {e}")
            return default
    
    async def get_user_interactions(self, user_id: str, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get interactions
            return user_model.get_interactions(interaction_type, count)
            
        except Exception as e:
            logger.error(f"Error getting interactions for user {user_id}: {e}")
            return []
    
    async def get_user_top_topics(self, user_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top topics of interest for a user.
        
        Args:
            user_id: User ID
            count: Number of topics to get
            
        Returns:
            List of topics
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get top topics
            return user_model.get_top_topics(count)
            
        except Exception as e:
            logger.error(f"Error getting top topics for user {user_id}: {e}")
            return []
"""
Utility functions for the AI assistant.

This module provides utility functions for the AI assistant.
"""

import logging
import os
import platform
import subprocess
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

# Configure logging
logger = logging.getLogger(__name__)

def capture_screen() -> Optional[np.ndarray]:
    """
    Capture the screen.
    
    Returns:
        Screen image as numpy array, or None if failed
    """
    try:
        # Try to import mss
        import mss
        
        # Capture screen
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]
            
            # Capture screenshot
            sct_img = sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(sct_img)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
    except ImportError:
        logger.warning("mss not available, trying alternative method")
        
        # Try alternative method based on platform
        system = platform.system()
        
        if system == "Windows":
            # Use PIL on Windows
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab()
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Error capturing screen on Windows: {e}")
                return None
                
        elif system == "Darwin":  # macOS
            try:
                # Use screencapture on macOS
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Capture screen to temporary file
                subprocess.run(["screencapture", "-x", temp_path], check=True)
                
                # Read image
                img = cv2.imread(temp_path)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                return img
                
            except Exception as e:
                logger.error(f"Error capturing screen on macOS: {e}")
                return None
                
        elif system == "Linux":
            try:
                # Use scrot on Linux
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Capture screen to temporary file
                subprocess.run(["scrot", temp_path], check=True)
                
                # Read image
                img = cv2.imread(temp_path)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                return img
                
            except Exception as e:
                logger.error(f"Error capturing screen on Linux: {e}")
                return None
        
        else:
            logger.error(f"Unsupported platform: {system}")
            return None
    
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return None

def extract_text_from_image(image: np.ndarray) -> List[str]:
    """
    Extract text from an image.
    
    Args:
        image: Image as numpy array
        
    Returns:
        List of extracted text
    """
    try:
        # Check if pytesseract is available
        if not pytesseract.get_tesseract_version():
            logger.warning("Tesseract not available, text extraction disabled")
            return []
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Split into lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        return lines
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return []

def execute_command(command: str) -> dict:
    """
    Execute a shell command.
    
    Args:
        command: Command to execute
        
    Returns:
        Dictionary with command result
    """
    try:
        # Execute command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Get output
        stdout, stderr = process.communicate()
        
        # Get return code
        return_code = process.returncode
        
        return {
            "success": return_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code
        }
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    try:
        # Get system information
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        # Get memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory_total"] = memory.total
            info["memory_available"] = memory.available
            info["memory_percent"] = memory.percent
            
            # Get disk information
            disk = psutil.disk_usage("/")
            info["disk_total"] = disk.total
            info["disk_free"] = disk.free
            info["disk_percent"] = disk.percent
            
            # Get CPU information
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
        except ImportError:
            logger.warning("psutil not available, limited system information")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system information: {e}")
        
        return {
            "error": str(e)
        }

def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"
"""
AI assistant main module.

This module provides the main AI assistant functionality.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import ConfigManager
from .knowledge import KnowledgeManager
from .models import ModelManager
from .session import SessionManager
from .user_model import UserModelManager
from .vision import VisionProcessor
from .voice import VoiceProcessor

# Configure logging
logger = logging.getLogger(__name__)

class AIAssistant:
    """
    AI assistant.
    
    This class provides the main AI assistant functionality.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the AI assistant.
        
        Args:
            config_path: Path to configuration file
        """
        # Configuration
        self.config_manager = ConfigManager(config_path)
        self.config = {}
        
        # Components
        self.model_manager = None
        self.session_manager = None
        self.knowledge_manager = None
        self.user_model_manager = None
        self.vision_processor = None
        self.voice_processor = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize the AI assistant."""
        try:
            # Load configuration
            self.config = await self.config_manager.load_config()
            
            # Configure logging
            log_level = self.config.get("log_level", "INFO")
            logging.basicConfig(
                level=getattr(logging, log_level),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
            # Initialize components
            await self._initialize_components()
            
            self.initialized = True
            logger.info("AI assistant initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI assistant: {e}")
    
    async def _initialize_components(self):
        """Initialize AI assistant components."""
        # Initialize model manager
        model_path
       # Initialize model manager
        model_path = self.config.get("model_path", "./models")
        self.model_manager = ModelManager(model_path)
        await self.model_manager.initialize()
        
        # Initialize session manager
        session_timeout = self.config.get("session_timeout", 300)
        self.session_manager = SessionManager(session_timeout)
        await self.session_manager.initialize()
        
        # Initialize knowledge manager
        knowledge_dir = self.config.get("knowledge_dir", "./knowledge")
        self.knowledge_manager = KnowledgeManager(knowledge_dir)
        await self.knowledge_manager.initialize()
        
        # Initialize user model manager
        user_models_dir = self.config.get("user_models_dir", "./user_models")
        self.user_model_manager = UserModelManager(user_models_dir)
        await self.user_model_manager.initialize()
        
        # Initialize vision processor
        self.vision_processor = VisionProcessor(model_path)
        await self.vision_processor.initialize()
        
        # Initialize voice processor
        self.voice_processor = VoiceProcessor(model_path)
        await self.voice_processor.initialize()
    
    async def process_query(self, query: str, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Start time
            start_time = time.time()
            
            # Get or create session
            if session_id is None:
                session_id = self.session_manager.create_session()
            else:
                # Update session
                self.session_manager.update_session(session_id)
            
            # Get session
            session = self.session_manager.get_session(session_id)
            if session is None:
                return {
                    "success": False,
                    "error": "Invalid session ID"
                }
            
            # Add query to session history
            session.add_message("user", query)
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "query",
                    query,
                    {"session_id": session_id}
                )
            
            # Process query
            response = await self._generate_response(query, session, user_id)
            
            # Add response to session history
            session.add_message("assistant", response["text"])
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "response",
                    response["text"],
                    {"session_id": session_id}
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return response
            return {
                "success": True,
                "session_id": session_id,
                "text": response["text"],
                "processing_time": processing_time,
                "actions": response.get("actions", []),
                "context": response.get("context", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_response(self, query: str, session, user_id: str = None) -> Dict[str, Any]:
        """
        Generate a response to a user query.
        
        Args:
            query: User query
            session: Session object
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        try:
            # Get conversation history
            history = session.get_messages()
            
            # Get user preferences if user ID is provided
            user_preferences = {}
            if user_id:
                user_model = await self.user_model_manager.get_user_model(user_id)
                user_preferences = user_model.preferences
            
            # Search knowledge base
            knowledge_results = await self.knowledge_manager.search_knowledge(query)
            
            # Prepare context
            context = {
                "knowledge": knowledge_results,
                "user_preferences": user_preferences,
                "session_data": session.data
            }
            
            # Generate response
            response = await self.model_manager.generate_response(
                query,
                history,
                context
            )
            
            # Extract actions
            actions = self._extract_actions(response)
            
            # Execute actions
            action_results = await self._execute_actions(actions, session, user_id)
            
            # Update response if needed
            if action_results:
                # Generate updated response
                updated_response = await self.model_manager.generate_response(
                    query,
                    history,
                    {
                        **context,
                        "action_results": action_results
                    }
                )
                
                response = updated_response
            
            return {
                "text": response,
                "actions": actions,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            return {
                "text": "I'm sorry, I encountered an error while processing your request."
            }
    
    def _extract_actions(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract actions from response.
        
        Args:
            response: Response text
            
        Returns:
            List of actions
        """
        # This is a simplified implementation
        # In a real implementation, we would use a more sophisticated approach
        # to extract actions from the response
        
        actions = []
        
        # Check for action markers
        if "[ACTION:" in response:
            # Extract action blocks
            action_blocks = response.split("[ACTION:")[1:]
            
            for block in action_blocks:
                # Extract action type and parameters
                action_end = block.find("]")
                if action_end != -1:
                    action_type = block[:action_end].strip()
                    
                    # Create action
                    action = {
                        "type": action_type,
                        "parameters": {}
                    }
                    
                    # Extract parameters
                    param_start = block.find("(")
                    param_end = block.find(")")
                    
                    if param_start != -1 and param_end != -1:
                        param_str = block[param_start+1:param_end]
                        
                        # Parse parameters
                        params = param_str.split(",")
                        for param in params:
                            if "=" in param:
                                key, value = param.split("=", 1)
                                action["parameters"][key.strip()] = value.strip()
                    
                    # Add action
                    actions.append(action)
        
        return actions
    
    async def _execute_actions(self, actions: List[Dict[str, Any]], session, user_id: str = None) -> Dict[str, Any]:
        """
        Execute actions.
        
        Args:
            actions: List of actions
            session: Session object
            user_id: User ID (optional)
            
        Returns:
            Action results
        """
        results = {}
        
        for action in actions:
            action_type = action["type"]
            parameters = action["parameters"]
            
            try:
                if action_type == "search_knowledge":
                    # Search knowledge base
                    query = parameters.get("query", "")
                    results["search_knowledge"] = await self.knowledge_manager.search_knowledge(query)
                
                elif action_type == "capture_screen":
                    # Capture and analyze screen
                    results["capture_screen"] = await self.vision_processor.capture_and_analyze_screen()
                
                elif action_type == "record_audio":
                    # Record audio
                    duration = float(parameters.get("duration", 5.0))
                    audio_data, sample_rate = await self.voice_processor.record_audio(duration)
                    
                    # Recognize speech
                    if audio_data is not None:
                        speech_result = await self.voice_processor.recognize_speech(audio_data, sample_rate)
                        results["record_audio"] = speech_result
                
                elif action_type == "synthesize_speech":
                    # Synthesize speech
                    text = parameters.get("text", "")
                    language = parameters.get("language", "en")
                    
                    audio_data, sample_rate = await self.voice_processor.synthesize_speech(text, language)
                    
                    # Play audio
                    if audio_data is not None:
                        await self.voice_processor.play_audio(audio_data, sample_rate)
                        
                        results["synthesize_speech"] = {
                            "success": True,
                            "text": text
                        }
                
                elif action_type == "update_user_preference":
                    # Update user preference
                    if user_id:
                        key = parameters.get("key", "")
                        value = parameters.get("value", "")
                        
                        if key:
                            success = await self.user_model_manager.update_user_preference(user_id, key, value)
                            results["update_user_preference"] = {
                                "success": success,
                                "key": key,
                                "value": value
                            }
                
                elif action_type == "store_session_data":
                    # Store session data
                    key = parameters.get("key", "")
                    value = parameters.get("value", "")
                    
                    if key:
                        session.data[key] = value
                        results["store_session_data"] = {
                            "success": True,
                            "key": key,
                            "value": value
                        }
                
                elif action_type == "execute_command":
                    # Execute command
                    command = parameters.get("command", "")
                    
                    if command:
                        from .utils import execute_command
                        result = execute_command(command)
                        results["execute_command"] = result
                
            except Exception as e:
                logger.error(f"Error executing action {action_type}: {e}")
                results[action_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def process_voice_query(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a voice query.
        
        Args:
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Record audio
            audio_data, sample_rate = await self.voice_processor.record_audio()
            
            if audio_data is None:
                return {
                    "success": False,
                    "error": "Failed to record audio"
                }
            
            # Recognize speech
            speech_result = await self.voice_processor.recognize_speech(audio_data, sample_rate)
            
            if not speech_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to recognize speech"
                }
            
            # Get recognized text
            query = speech_result["text"]
            
            # Process query
            response = await self.process_query(query, session_id, user_id)
            
            # Synthesize speech
            if response["success"]:
                audio_data, sample_rate = await self.voice_processor.synthesize_speech(response["text"])
                
                if audio_data is not None:
                    # Play audio
                    await self.voice_processor.play_audio(audio_data, sample_rate)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_visual_query(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a visual query.
        
        Args:
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Capture and analyze screen
            screen_result = await self.vision_processor.capture_and_analyze_screen()
            
            if not screen_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to capture and analyze screen"
                }
            
            # Get or create session
            if session_id is None:
                session_id = self.session_manager.create_session()
            else:
                # Update session
                self.session_manager.update_session(session_id)
            
            # Get session
            session = self.session_manager.get_session(session_id)
            if session is None:
                return {
                    "success": False,
                    "error": "Invalid session ID"
                }
            
            # Store screen analysis in session data
            session.data["screen_analysis"] = screen_result
            
            # Generate response
            response = await self._generate_response(
                "What do you see on my screen?",
                session,
                user_id
            )
            
            # Add response to session history
            session.add_message("assistant", response["text"])
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "visual_query",
                    "What do you see on my screen?",
                    {"session_id": session_id}
                )
                
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "response",
                    response["text"],
                    {"session_id": session_id}
                )
            
            return {
                "success": True,
                "session_id": session_id,
                "text": response["text"],
                "screen_analysis": screen_result,
                "actions": response.get("actions", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing visual query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return self.session_manager.end_session(session_id)
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information, or None if session doesn't exist
        """
       """
        if not self.initialized:
            return None
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        
        return {
            "session_id": session_id,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "message_count": len(session.messages),
            "data": session.data
        }
    
    async def get_session_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get session messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of messages, or None if session doesn't exist
        """
        if not self.initialized:
            return None
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        
        return session.get_messages()
    
    async def get_active_sessions(self) -> List[str]:
        """
        Get active session IDs.
        
        Returns:
            List of active session IDs
        """
        if not self.initialized:
            return []
        
        return self.session_manager.get_active_sessions()
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            
        Returns:
            Knowledge ID
        """
        if not self.initialized:
            return ""
        
        return await self.knowledge_manager.add_knowledge(knowledge)
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        if not self.initialized:
            return None
        
        return await self.knowledge_manager.get_knowledge(knowledge_id)
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.knowledge_manager.update_knowledge(knowledge_id, knowledge)
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.knowledge_manager.delete_knowledge(knowledge_id)
    
    async def search_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        if not self.initialized:
            return []
        
        return await self.knowledge_manager.search_knowledge(query, max_results=max_results)
    
    async def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model, or None if not found
        """
        if not self.initialized:
            return None
        
        user_model = await self.user_model_manager.get_user_model(user_id)
        return user_model.to_dict()
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.user_model_manager.update_user_preference(user_id, key, value)
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        if not self.initialized:
            return default
        
        return await self.user_model_manager.get_user_preference(user_id, key, default)
    
    async def get_user_interactions(self, user_id: str, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        if not self.initialized:
            return []
        
        return await self.user_model_manager.get_user_interactions(user_id, interaction_type, count)
    
    async def delete_user_model(self, user_id: str) -> bool:
        """
        Delete a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.user_model_manager.delete_user_model(user_id)
    
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List of user IDs
        """
        if not self.initialized:
            return []
        
        return await self.user_model_manager.get_all_user_ids()
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Screen analysis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.vision_processor.capture_and_analyze_screen()
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image analysis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.vision_processor.analyze_image(image_path)
    
    async def record_audio(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Record audio.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Audio recording result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        audio_data, sample_rate = await self.voice_processor.record_audio(duration)
        
        if audio_data is None:
            return {
                "success": False,
                "error": "Failed to record audio"
            }
        
        return {
            "success": True,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    
    async def recognize_speech(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Speech recognition result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.voice_processor.recognize_speech(audio_data, sample_rate)
    
    async def synthesize_speech(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Synthesize speech.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Speech synthesis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        audio_data, sample_rate = await self.voice_processor.synthesize_speech(text, language)
        
        if audio_data is None:
            return {
                "success": False,
                "error": "Failed to synthesize speech"
            }
        
        return {
            "success": True,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    
    async def play_audio(self, audio_data: bytes, sample_rate: int) -> bool:
        """
        Play audio.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.voice_processor.play_audio(audio_data, sample_rate)
    
    async def shutdown(self):
        """Shut down the AI assistant."""
        try:
            # Save user models
            if self.user_model_manager:
                await self.user_model_manager.save_all_user_models()
            
            # Save knowledge base
            if self.knowledge_manager:
                await self.knowledge_manager.save_knowledge_base()
            
            # Save configuration
            if self.config_manager:
                await self.config_manager.save_config(self.config)
            
            logger.info("AI assistant shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down AI assistant: {e}")
"""
AI assistant main entry point.

This module provides the main entry point for the AI assistant.
"""

import argparse
import asyncio
import logging
import sys

from .assistant import AIAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI assistant")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create AI assistant
    assistant = AIAssistant(args.config)
    
    try:
        # Initialize AI assistant
        await assistant.initialize()
        
        # Start interactive mode
        await interactive_mode(assistant)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error: {e}")
    
    finally:
        # Shut down AI assistant
        await assistant.shutdown()

async def interactive_mode(assistant: AIAssistant):
    """
    Interactive mode.
    
    Args:
        assistant: AI assistant
    """
    print("AI assistant interactive mode")
    print("Type 'exit' to exit")
    print("Type 'help' for help")
    print()
    
    session_id = None
    user_id = "interactive_user"
    
    while True:
        try:
            # Get user input
            user_input = input("> ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Check for help command
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Check for special commands
            if user_input.startswith("/"):
                await handle_command(assistant, user_input, session_id, user_id)
                continue
            
            # Process query
            response = await assistant.process_query(user_input, session_id, user_id)
            
            # Update session ID
            if response["success"]:
                session_id = response["session_id"]
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            logger.error(f"Error: {e}")

async def handle_command(assistant: AIAssistant, command: str, session_id: str, user_id: str):
    """
    Handle a special command.
    
    Args:
        assistant: AI assistant
        command: Command string
        session_id: Session ID
        user_id: User ID
    """
    # Parse command
    parts = command[1:].split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]
    
    try:
        if cmd == "voice":
            # Process voice query
            print("Listening...")
            response = await assistant.process_voice_query(session_id, user_id)
            
            if response["success"]:
                session_id = response["session_id"]
                print(f"You said: {response.get('query', '')}")
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        elif cmd == "visual":
            # Process visual query
            print("Capturing screen...")
            response = await assistant.process_visual_query(session_id, user_id)
            
            if response["success"]:
                session_id = response["session_id"]
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        elif cmd == "session":
            # Session commands
            if not args:
                # Get session info
                if session_id:
                    info = await assistant.get_session_info(session_id)
                    print(f"Session ID: {session_id}")
                    print(f"Created at: {info['created_at']}")
                    print(f"Last updated: {info['last_updated']}")
                    print(f"Message count: {info['message_count']}")
                else:
                    print("No active session")
            
            elif args[0] == "new":
                # Create new session
                session_id = assistant.session_manager.create_session()
                print(f"Created new session: {session_id}")
           elif args[0] == "list":
                # List active sessions
                sessions = await assistant.get_active_sessions()
                print(f"Active sessions ({len(sessions)}):")
                for s in sessions:
                    info = await assistant.get_session_info(s)
                    print(f"  {s} - {info['message_count']} messages, last updated: {info['last_updated']}")
            
            elif args[0] == "end":
                # End session
                if session_id:
                    success = await assistant.end_session(session_id)
                    if success:
                        print(f"Ended session: {session_id}")
                        session_id = None
                    else:
                        print(f"Failed to end session: {session_id}")
                else:
                    print("No active session")
            
            elif args[0] == "messages":
                # Get session messages
                if session_id:
                    messages = await assistant.get_session_messages(session_id)
                    print(f"Session messages ({len(messages)}):")
                    for msg in messages:
                        print(f"  {msg['role']}: {msg['content']}")
                else:
                    print("No active session")
            
            else:
                print(f"Unknown session command: {args[0]}")
        
        elif cmd == "knowledge":
            # Knowledge commands
            if not args:
                print("Usage: /knowledge [add|get|update|delete|search|list]")
            
            elif args[0] == "add":
                # Add knowledge
                if len(args) < 3:
                    print("Usage: /knowledge add <title> <content>")
                else:
                    title = args[1]
                    content = " ".join(args[2:])
                    
                    knowledge = {
                        "title": title,
                        "content": content,
                        "source": "interactive",
                        "tags": []
                    }
                    
                    knowledge_id = await assistant.add_knowledge(knowledge)
                    print(f"Added knowledge: {knowledge_id}")
            
            elif args[0] == "get":
                # Get knowledge
                if len(args) < 2:
                    print("Usage: /knowledge get <id>")
                else:
                    knowledge_id = args[1]
                    knowledge = await assistant.get_knowledge(knowledge_id)
                    
                    if knowledge:
                        print(f"Knowledge ID: {knowledge_id}")
                        print(f"Title: {knowledge['title']}")
                        print(f"Content: {knowledge['content']}")
                        print(f"Source: {knowledge.get('source', 'unknown')}")
                        print(f"Tags: {', '.join(knowledge.get('tags', []))}")
                    else:
                        print(f"Knowledge not found: {knowledge_id}")
            
            elif args[0] == "update":
                # Update knowledge
                if len(args) < 4:
                    print("Usage: /knowledge update <id> <title> <content>")
                else:
                    knowledge_id = args[1]
                    title = args[2]
                    content = " ".join(args[3:])
                    
                    # Get existing knowledge
                    knowledge = await assistant.get_knowledge(knowledge_id)
                    
                    if knowledge:
                        # Update knowledge
                        knowledge["title"] = title
                        knowledge["content"] = content
                        
                        success = await assistant.update_knowledge(knowledge_id, knowledge)
                        
                        if success:
                            print(f"Updated knowledge: {knowledge_id}")
                        else:
                            print(f"Failed to update knowledge: {knowledge_id}")
                    else:
                        print(f"Knowledge not found: {knowledge_id}")
            
            elif args[0] == "delete":
                # Delete knowledge
                if len(args) < 2:
                    print("Usage: /knowledge delete <id>")
                else:
                    knowledge_id = args[1]
                    success = await assistant.delete_knowledge(knowledge_id)
                    
                    if success:
                        print(f"Deleted knowledge: {knowledge_id}")
                    else:
                        print(f"Failed to delete knowledge: {knowledge_id}")
            
            elif args[0] == "search":
                # Search knowledge
                if len(args) < 2:
                    print("Usage: /knowledge search <query>")
                else:
                    query = " ".join(args[1:])
                    results = await assistant.search_knowledge(query)
                    
                    print(f"Search results ({len(results)}):")
                    for result in results:
                        print(f"  ID: {result['id']}")
                        print(f"  Title: {result['knowledge']['title']}")
                        print(f"  Score: {result['score']}")
                        print()
            
            elif args[0] == "list":
                # List all knowledge
                knowledge_items = await assistant.knowledge_manager.get_all_knowledge()
                
                print(f"Knowledge items ({len(knowledge_items)}):")
                for item in knowledge_items:
                    print(f"  ID: {item['id']}")
                    print(f"  Title: {item['knowledge']['title']}")
                    print()
            
            else:
                print(f"Unknown knowledge command: {args[0]}")
        
        elif cmd == "user":
            # User commands
            if not args:
                # Get user info
                user_model = await assistant.get_user_model(user_id)
                
                if user_model:
                    print(f"User ID: {user_model['user_id']}")
                    print(f"Created at: {user_model['created_at']}")
                    print(f"Last updated: {user_model['last_updated']}")
                    print(f"Preferences: {user_model['preferences']}")
                    print(f"Interactions: {len(user_model['interactions'])}")
                    
                    # Get top topics
                    topics = sorted(user_model['topics'].items(), key=lambda x: x[1], reverse=True)[:5]
                    if topics:
                        print("Top topics:")
                        for topic, score in topics:
                            print(f"  {topic}: {score}")
                else:
                    print(f"User not found: {user_id}")
            
            elif args[0] == "preference":
                # User preference commands
                if len(args) < 2:
                    print("Usage: /user preference [get|set] <key> [value]")
                
                elif args[1] == "get":
                    # Get preference
                    if len(args) < 3:
                        print("Usage: /user preference get <key>")
                    else:
                        key = args[2]
                        value = await assistant.get_user_preference(user_id, key)
                        print(f"Preference {key}: {value}")
                
                elif args[1] == "set":
                    # Set preference
                    if len(args) < 4:
                        print("Usage: /user preference set <key> <value>")
                    else:
                        key = args[2]
                        value = " ".join(args[3:])
                        
                        success = await assistant.update_user_preference(user_id, key, value)
                        
                        if success:
                            print(f"Set preference {key} = {value}")
                        else:
                            print(f"Failed to set preference {key}")
                
                else:
                    print(f"Unknown preference command: {args[1]}")
            
            elif args[0] == "interactions":
                # Get user interactions
                count = int(args[1]) if len(args) > 1 else 5
                interactions = await assistant.get_user_interactions(user_id, count=count)
                
                print(f"User interactions ({len(interactions)}):")
                for interaction in interactions:
                    print(f"  {interaction['type']} at {interaction['timestamp']}")
                    print(f"    {interaction['content']}")
                    print()
            
            else:
                print(f"Unknown user command: {args[0]}")
        
        elif cmd == "config":
            # Configuration commands
            if not args:
                # Show configuration
                print("Configuration:")
                for key, value in assistant.config.items():
                    print(f"  {key}: {value}")
            
            elif len(args) >= 2:
                # Set configuration
                key = args[0]
                value = " ".join(args[1:])
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        value = float(value)
                except:
                    pass
                
                # Update configuration
                assistant.config[key] = value
                
                # Save configuration
                await assistant.config_manager.save_config(assistant.config)
                
                print(f"Set {key} = {value}")
            
            else:
                print("Usage: /config [key value]")
        
        elif cmd == "help":
            # Show help
            print_help()
        
        else:
            print(f"Unknown command: {cmd}")
    
    except Exception as e:
        logger.error(f"Error handling command: {e}")
        print(f"Error: {e}")

def print_help():
    """Print help information."""
    print("AI assistant commands:")
    print("  /voice - Process voice query")
    print("  /visual - Process visual query")
    print("  /session - Session commands")
    print("    /session - Show current session info")
    print("    /session new - Create new session")
    print("    /session list - List active sessions")
    print("    /session end - End current session")
    print("    /session messages - Show session messages")
    print("  /knowledge - Knowledge commands")
    print("    /knowledge add <title> <content> - Add knowledge")
    print("    /knowledge get <id> - Get knowledge")
    print("    /knowledge update <id> <title> <content> - Update knowledge")
    print("    /knowledge delete <id> - Delete knowledge")
    print("    /knowledge search <query> - Search knowledge")
    print("    /knowledge list - List all knowledge")
    print("  /user - User commands")
    print("    /user - Show user info")
    print("    /user preference get <key> - Get user preference")
    print("    /user preference set <key> <value> - Set user preference")
    print("    /user interactions [count] - Show user interactions")
    print("  /config - Configuration commands")
    print("    /config - Show configuration")
    print("    /config <key> <value> - Set configuration")
    print("  /help - Show help")
    print("  exit - Exit")

if __name__ == "__main__":
    asyncio.run(main())
"""
Vision processing for the AI assistant.

This module provides vision processing functionality for the AI assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import capture_screen, extract_text_from_image

# Configure logging
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Vision processor for the AI assistant.
    
    This class provides vision processing functionality for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the vision processor.
        
        Args:
            model_path: Path to vision models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.object_detector = None
        self.face_detector = None
        self.text_detector = None
    
    async def initialize(self):
        """Initialize the vision processor."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            await self._initialize_models()
            
            logger.info("Vision processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vision processor: {e}")
    
    async def _initialize_models(self):
        """Initialize vision models."""
        try:
            # Initialize object detector
            self._initialize_object_detector()
            
            # Initialize face detector
            self._initialize_face_detector()
            
            # Initialize text detector
            self._initialize_text_detector()
            
        except Exception as e:
            logger.error(f"Error initializing vision models: {e}")
    
    def _initialize_object_detector(self):
        """Initialize object detector."""
        try:
            # Check if OpenCV DNN module is available
            if hasattr(cv2, "dnn"):
                # Load YOLO model
                yolo_cfg = self.model_path / "yolov3.cfg"
                yolo_weights = self.model_path / "yolov3.weights"
                
                if yolo_cfg.exists() and yolo_weights.exists():
                    self.object_detector = cv2.dnn.readNetFromDarknet(
                        str(yolo_cfg),
                        str(yolo_weights)
                    )
                    
                    # Load COCO class names
                    coco_names = self.model_path / "coco.names"
                    if coco_names.exists():
                        with open(coco_names, "r") as f:
                            self.object_classes = f.read().strip().split("\n")
                    else:
                        # Default COCO class names
                        self.object_classes = [
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                            "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                        ]
                    
                    logger.info("Object detector initialized")
                else:
                    logger.warning("YOLO model files not found, object detection disabled")
            else:
                logger.warning("OpenCV DNN module not available, object detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
    
    def _initialize_face_detector(self):
        """Initialize face detector."""
        try:
            # Load Haar cascade for face detection
           # Load Haar cascade for face detection
            face_cascade_path = self.model_path / "haarcascade_frontalface_default.xml"
            
            if face_cascade_path.exists():
                self.face_detector = cv2.CascadeClassifier(str(face_cascade_path))
                logger.info("Face detector initialized")
            else:
                # Try to use OpenCV's built-in cascades
                builtin_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(builtin_cascade):
                    self.face_detector = cv2.CascadeClassifier(builtin_cascade)
                    logger.info("Face detector initialized using built-in cascade")
                else:
                    logger.warning("Face cascade file not found, face detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
    
    def _initialize_text_detector(self):
        """Initialize text detector."""
        try:
            # Check if OpenCV text detection is available
            if hasattr(cv2, "text") and hasattr(cv2.text, "TextDetectorCNN_create"):
                # Load EAST text detector
                east_model_path = self.model_path / "frozen_east_text_detection.pb"
                
                if east_model_path.exists():
                    self.text_detector = cv2.dnn.readNet(str(east_model_path))
                    logger.info("Text detector initialized")
                else:
                    logger.warning("EAST model file not found, advanced text detection disabled")
            else:
                logger.warning("OpenCV text detection not available, advanced text detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing text detector: {e}")
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Screen analysis result
        """
        try:
            # Capture screen
            screen = capture_screen()
            
            if screen is None:
                return {
                    "success": False,
                    "error": "Failed to capture screen"
                }
            
            # Analyze screen
            analysis = await self.analyze_image_data(screen)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error capturing and analyzing screen: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image analysis result
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to read image"
                }
            
            # Analyze image
            analysis = await self.analyze_image_data(image)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_image_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image data.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            Image analysis result
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Extract text
            text = extract_text_from_image(image)
            
            # Detect objects
            objects = self._detect_objects(image) if self.object_detector else []
            
            # Detect faces
            faces = self._detect_faces(image) if self.face_detector else []
            
            # Detect colors
            colors = self._detect_dominant_colors(image)
            
            # Basic image properties
            properties = {
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height > 0 else 0,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "is_color": len(image.shape) > 2 and image.shape[2] > 1
            }
            
            return {
                "properties": properties,
                "text": text,
                "objects": objects,
                "faces": faces,
                "colors": colors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image data: {e}")
            
            return {
                "properties": {
                    "width": 0,
                    "height": 0,
                    "aspect_ratio": 0,
                    "channels": 0,
                    "is_color": False
                },
                "text": [],
                "objects": [],
                "faces": [],
                "colors": []
            }
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            List of detected objects
        """
        try:
            if self.object_detector is None:
                return []
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image,
                1/255.0,
                (416, 416),
                swapRB=True,
                crop=False
            )
            
            # Set input
            self.object_detector.setInput(blob)
            
            # Get output layer names
            layer_names = self.object_detector.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.object_detector.getUnconnectedOutLayers()]
            
            # Forward pass
            outputs = self.object_detector.forward(output_layers)
            
            # Process outputs
            class_ids = []
            confidences = []
            boxes = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Prepare results
            objects = []
            
            for i in indices:
                if isinstance(i, tuple):
                    i = i[0]  # For OpenCV 3
                
                box = boxes[i]
                x, y, w, h = box
                
                # Get class name
                class_id = class_ids[i]
                class_name = self.object_classes[class_id] if class_id < len(self.object_classes) else f"unknown_{class_id}"
                
                # Add object
                objects.append({
                    "class": class_name,
                    "confidence": confidences[i],
                    "box": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "center": {
                        "x": x + w // 2,
                        "y": y + h // 2
                    },
                    "area": w * h,
                    "relative_area": (w * h) / (width * height)
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            if self.face_detector is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rect = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Prepare results
            faces = []
            
            for (x, y, w, h) in faces_rect:
                # Add face
                faces.append({
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "center": {
                        "x": int(x + w // 2),
                        "y": int(y + h // 2)
                    },
                    "area": int(w * h),
                    "relative_area": float((w * h) / (width * height))
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _detect_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Dict[str, Any]]:
        """
        Detect dominant colors in an image.
        
        Args:
            image: Image data as numpy array
            num_colors: Number of dominant colors to detect
            
        Returns:
            List of dominant colors
        """
        try:
            # Reshape image
            pixels = image.reshape(-1, 3)
            
            # Convert to float32
            pixels = np.float32(pixels)
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Apply k-means clustering
            _, labels, centers = cv2.kmeans(
                pixels,
                num_colors,
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert centers to uint8
            centers = np.uint8(centers)
            
            # Count labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            
            # Prepare results
            colors = []
            
            for i in sorted_indices:
                # Get color
                color = centers[i].tolist()
                
                # Calculate percentage
                percentage = counts[i] / len(labels)
                
                # Add color
                colors.append({
                    "color": {
                        "b": color[0],
                        "g": color[1],
                        "r": color[2],
                        "hex": f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                    },
                    "percentage": float(percentage)
                })
            
            return colors
            
        except Exception as e:
            logger.error(f"Error detecting dominant colors: {e}")
            return []
"""
Voice processing for the AI assistant.

This module provides voice processing functionality for the AI assistant.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for the AI assistant.
    
    This class provides voice processing functionality for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the voice processor.
        
        Args:
            model_path: Path to voice models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.speech_recognizer = None
        self.speech_synthesizer = None
        
        # Audio device
        self.audio_device = None
    
    async def initialize(self):
        """Initialize the voice processor."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            await self._initialize_models()
            
            # Initialize audio device
            await self._initialize_audio_device()
            
            logger.info("Voice processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing voice processor: {e}")
    
    async def _initialize_models(self):
        """Initialize voice models."""
        try:
            # Initialize speech recognizer
            await self._initialize_speech_recognizer()
            
            # Initialize speech synthesizer
            await self._initialize_speech_synthesizer()
            
        except Exception as e:
            logger.error(f"Error initializing voice models: {e}")
    
    async def _initialize_speech_recognizer(self):
        """Initialize speech recognizer."""
        try:
            # Try to import speech recognition library
            import speech_recognition as sr
            
            # Create recognizer
            self.speech_recognizer = sr.Recognizer()
            
            logger.info("Speech recognizer initialized")
            
        except ImportError:
            logger.warning("speech_recognition not available, speech recognition disabled")
            
        except Exception as e:
            logger.error(f"Error initializing speech recognizer: {e}")
    
    async def _initialize_speech_synthesizer(self):
        """Initialize speech synthesizer."""
        try:
            # Try to import pyttsx3
            import pyttsx3
            
            # Create synthesizer
            self.speech_synthesizer = pyttsx3.init()
            
            logger.info("Speech synthesizer initialized")
            
        except ImportError:
            logger.warning("pyttsx3 not available, speech synthesis disabled")
            
            # Try to import gTTS as fallback
            try:
                from gtts import gTTS
                
                # Use gTTS
                self.speech_synthesizer = "gtts"
                
                logger.info("Speech synthesizer initialized using gTTS")
                
            except ImportError:
                logger.warning("gtts not available, speech synthesis disabled")
            
        except Exception as e:
            logger.error(f"Error initializing speech synthesizer: {e}")
    
    async def _initialize_audio_device(self):
        """Initialize audio device."""
        try:
            #

"""
Session management for the AI assistant.

This module provides session management functionality for the AI assistant.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class Session:
    """
    Session class for the AI assistant.
    
    This class represents a session with the AI assistant.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize a session.
        
        Args:
            session_id: Session ID
        """
        self.session_id = session_id
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.messages = []
        self.data = {}
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the session.
        
        Args:
            role: Message role (user or assistant)
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        self.last_updated = time.time()
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get session messages.
        
        Returns:
            List of messages
        """
        return self.messages
    
    def clear_messages(self):
        """Clear session messages."""
        self.messages = []
        self.last_updated = time.time()
    
    def is_expired(self, timeout: float) -> bool:
        """
        Check if the session is expired.
        
        Args:
            timeout: Session timeout in seconds
            
        Returns:
            True if the session is expired, False otherwise
        """
        return time.time() - self.last_updated > timeout

class SessionManager:
    """
    Session manager for the AI assistant.
    
    This class manages sessions for the AI assistant.
    """
    
    def __init__(self, session_timeout: float = 300.0):
        """
        Initialize the session manager.
        
        Args:
            session_timeout: Session timeout in seconds
        """
        self.sessions = {}
        self.session_timeout = session_timeout
    
    async def initialize(self):
        """Initialize the session manager."""
        try:
            logger.info("Session manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing session manager: {e}")
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        self.sessions[session_id] = Session(session_id)
        
        logger.info(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session, or None if not found
        """
        # Check if session exists
        if session_id not in self.sessions:
            return None
        
        # Get session
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.session_timeout):
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
            
            return None
        
        return session
    
    def update_session(self, session_id: str) -> bool:
        """
        Update a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.sessions:
            return False
        
        # Get session
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.session_timeout):
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
            
            return False
        
        # Update session
        session.last_updated = time.time()
        
        return True
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.sessions:
            return False
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Ended session: {session_id}")
        
        return True
    
    def get_active_sessions(self) -> List[str]:
        """
        Get active session IDs.
        
        Returns:
            List of active session IDs
        """
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        # Return active session IDs
        return list(self.sessions.keys())
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        # Get current time
        current_time = time.time()
        
        # Find expired sessions
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if current_time - session.last_updated > self.session_timeout
        ]
        
        # Remove expired sessions
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
"""
Model management for the AI assistant.

This module provides model management functionality for the AI assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for the AI assistant.
    
    This class manages AI models for the assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.models = {}
        
        # Default model
        self.default_model = None
    
    async def initialize(self):
        """Initialize the model manager."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Load models
            await self._load_models()
            
            logger.info("Model manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model manager: {e}")
    
    async def _load_models(self):
        """Load AI models."""
        try:
            # Try to import transformers
            try:
                from transformers import pipeline
                
                # Load text generation model
                try:
                    logger.info("Loading text generation model...")
                    
                    # Check if model exists locally
                    local_model_path = self.model_path / "text-generation"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["text-generation"] = pipeline(
                            "text-generation",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["text-generation"] = pipeline(
                            "text-generation",
                            model="gpt2"
                        )
                    
                    logger.info("Text generation model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading text generation model: {e}")
                
                # Load text classification model
                try:
                    logger.info("Loading text classification model...")
                    
                    # Check if model exists locally
                    local_model_path = self.model_path / "text-classification"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["text-classification"] = pipeline(
                            "text-classification",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["text-classification"] = pipeline(
                            "text-classification"
                        )
                    
                    logger.info("Text classification model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading text classification model: {e}")
                
                # Load question answering model
                try:
                    logger.info("Loading question answering model...")
                    
                    # Check if model
                   # Check if model exists locally
                    local_model_path = self.model_path / "question-answering"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["question-answering"] = pipeline(
                            "question-answering",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["question-answering"] = pipeline(
                            "question-answering"
                        )
                    
                    logger.info("Question answering model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading question answering model: {e}")
                
                # Set default model
                self.default_model = "text-generation"
                
            except ImportError:
                logger.warning("transformers not available, using fallback models")
                
                # Load fallback models
                self._load_fallback_models()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
            # Load fallback models
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback AI models."""
        try:
            # Simple rule-based model
            self.models["rule-based"] = {
                "type": "rule-based",
                "rules": [
                    {
                        "pattern": r"hello|hi|hey",
                        "response": "Hello! How can I help you today?"
                    },
                    {
                        "pattern": r"how are you",
                        "response": "I'm functioning well, thank you for asking. How can I assist you?"
                    },
                    {
                        "pattern": r"bye|goodbye",
                        "response": "Goodbye! Have a great day."
                    },
                    {
                        "pattern": r"thank you|thanks",
                        "response": "You're welcome! Is there anything else I can help with?"
                    },
                    {
                        "pattern": r"help",
                        "response": "I'm here to help. What do you need assistance with?"
                    }
                ]
            }
            
            # Set default model
            self.default_model = "rule-based"
            
            logger.info("Fallback models loaded")
            
        except Exception as e:
            logger.error(f"Error loading fallback models: {e}")
    
    async def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        try:
            # Check if text generation model is available
            if "text-generation" in self.models:
                # Generate text
                result = self.models["text-generation"](
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1
                )
                
                # Extract generated text
                generated_text = result[0]["generated_text"]
                
                return generated_text
            
            # Check if rule-based model is available
            elif "rule-based" in self.models:
                # Use rule-based model
                import re
                
                # Check rules
                for rule in self.models["rule-based"]["rules"]:
                    if re.search(rule["pattern"], prompt.lower()):
                        return rule["response"]
                
                # Default response
                return "I'm not sure how to respond to that."
            
            else:
                return "Text generation model not available."
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"
    
    async def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result
        """
        try:
            # Check if text classification model is available
            if "text-classification" in self.models:
                # Classify text
                result = self.models["text-classification"](text)
                
                # Extract classification
                if isinstance(result, list):
                    result = result[0]
                
                return {
                    "label": result["label"],
                    "score": result["score"]
                }
            
            else:
                return {
                    "label": "unknown",
                    "score": 0.0
                }
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            
            return {
                "label": "error",
                "score": 0.0,
                "error": str(e)
            }
    
    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question based on context.
        
        Args:
            question: Question to answer
            context: Context for the question
            
        Returns:
            Answer result
        """
        try:
            # Check if question answering model is available
            if "question-answering" in self.models:
                # Answer question
                result = self.models["question-answering"](
                    question=question,
                    context=context
                )
                
                return {
                    "answer": result["answer"],
                    "score": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                }
            
            else:
                return {
                    "answer": "I don't know the answer to that question.",
                    "score": 0.0,
                    "start": 0,
                    "end": 0
                }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            
            return {
                "answer": f"Error answering question: {e}",
                "score": 0.0,
                "start": 0,
                "end": 0
            }
    
    async def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_name: Model name, or None for default model
            
        Returns:
            Model information
        """
        try:
            # Get model name
            if model_name is None:
                model_name = self.default_model
            
            # Check if model exists
            if model_name not in self.models:
                return {
                    "name": model_name,
                    "available": False,
                    "error": "Model not found"
                }
            
            # Get model
            model = self.models[model_name]
            
            # Get model info
            if model_name == "rule-based":
                return {
                    "name": model_name,
                    "available": True,
                    "type": "rule-based",
                    "rules": len(model["rules"])
                }
            else:
                return {
                    "name": model_name,
                    "available": True,
                    "type": "transformer"
                }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            
            return {
                "name": model_name,
                "available": False,
                "error": str(e)
            }
    
    async def get_available_models(self) -> List[str]:
        """
        Get available models.
        
        Returns:
            List of available model names
        """
        return list(self.models.keys())
"""
Knowledge management for the AI assistant.

This module provides knowledge management functionality for the AI assistant.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge manager for the AI assistant.
    
    This class manages knowledge for the AI assistant.
    """
    
    def __init__(self, knowledge_path: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            knowledge_path: Path to knowledge base
        """
        self.knowledge_path = Path(knowledge_path)
        
        # Knowledge base
        self.knowledge_base = {}
        
        # Vector store for search
        self.vector_store = None
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        try:
            # Create knowledge directory if it doesn't exist
            self.knowledge_path.mkdir(parents=True, exist_ok=True)
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            logger.info("Knowledge manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge manager: {e}")
    
    async def _load_knowledge_base(self):
        """Load knowledge base."""
        try:
            # Knowledge base file
            knowledge_file = self.knowledge_path / "knowledge_base.json"
            
            # Check if file exists
            if knowledge_file.exists():
                # Load knowledge base
                with open(knowledge_file, "r") as f:
                    self.knowledge_base = json.load(f)
                
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} items")
            else:
                # Create empty knowledge base
                self.knowledge_base = {}
                
                logger.info("Created empty knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            
            # Create empty knowledge base
            self.knowledge_base = {}
    
    async def _initialize_vector_store(self):
        """Initialize vector store for search."""
        try:
            # Try to import sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                
                # Load model
                self.vector_store = SentenceTransformer("all-MiniLM-L6-v2")
                
                logger.info("Vector store initialized")
                
            except ImportError:
                logger.warning("sentence-transformers not available, using fallback search")
                
                # Use fallback search
                self.vector_store = None
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            
            # Use fallback search
            self.vector_store = None
    
    async def save_knowledge_base(self):
        """Save knowledge base."""
        try:
            # Knowledge base file
            knowledge_file = self.knowledge_path / "knowledge_base.json"
            
            # Save knowledge base
            with open(knowledge_file, "w") as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            logger.info(f"Saved knowledge base with {len(self.knowledge_base)} items")
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            
        Returns:
            Knowledge ID
        """
        try:
            # Generate knowledge ID
            knowledge_id = str(uuid.uuid4())
            
            # Add knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Added knowledge: {knowledge_id}")
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return None
            
            # Get knowledge
            return self.knowledge_base[knowledge_id]
            
        except Exception as e:
            logger.error(f"Error getting knowledge: {e}")
            return None
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return False
            
            # Update knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Updated knowledge: {knowledge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return False
            
            # Delete knowledge
            del self.knowledge_base[knowledge_id]
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Deleted knowledge: {knowledge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")
            return False
   async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Check if knowledge base is empty
            if not self.knowledge_base:
                return []
            
            # Check if vector store is available
            if self.vector_store is not None:
                # Use vector store for search
                return await self._vector_search(query, limit)
            else:
                # Use fallback search
                return await self._fallback_search(query, limit)
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search knowledge base using vector store.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Encode query
            query_embedding = self.vector_store.encode(query)
            
            # Encode knowledge
            knowledge_texts = []
            knowledge_ids = []
            
            for knowledge_id, knowledge in self.knowledge_base.items():
                # Get text
                text = f"{knowledge.get('title', '')} {knowledge.get('content', '')}"
                
                # Add to list
                knowledge_texts.append(text)
                knowledge_ids.append(knowledge_id)
            
            # Encode knowledge
            knowledge_embeddings = self.vector_store.encode(knowledge_texts)
            
            # Calculate similarity
            from numpy import dot
            from numpy.linalg import norm
            
            # Calculate cosine similarity
            similarities = [
                dot(query_embedding, embedding) / (norm(query_embedding) * norm(embedding))
                for embedding in knowledge_embeddings
            ]
            
            # Sort by similarity
            results = sorted(
                [
                    {
                        "id": knowledge_id,
                        "knowledge": self.knowledge_base[knowledge_id],
                        "score": float(similarity)
                    }
                    for knowledge_id, similarity in zip(knowledge_ids, similarities)
                ],
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            
            # Fallback to basic search
            return await self._fallback_search(query, limit)
    
    async def _fallback_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search knowledge base using fallback search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Simple keyword search
            query_terms = query.lower().split()
            
            # Calculate scores
            results = []
            
            for knowledge_id, knowledge in self.knowledge_base.items():
                # Get text
                title = knowledge.get("title", "").lower()
                content = knowledge.get("content", "").lower()
                
                # Calculate score
                score = 0.0
                
                for term in query_terms:
                    # Check title
                    if term in title:
                        score += 2.0
                    
                    # Check content
                    if term in content:
                        score += 1.0
                
                # Normalize score
                if len(query_terms) > 0:
                    score /= len(query_terms)
                
                # Add to results if score > 0
                if score > 0:
                    results.append({
                        "id": knowledge_id,
                        "knowledge": knowledge,
                        "score": score
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing fallback search: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge.
        
        Returns:
            List of all knowledge items
        """
        try:
            # Convert knowledge base to list
            return [
                {
                    "id": knowledge_id,
                    "knowledge": knowledge
                }
                for knowledge_id, knowledge in self.knowledge_base.items()
            ]
            
        except Exception as e:
            logger.error(f"Error getting all knowledge: {e}")
            return []
"""
Configuration management for the AI assistant.

This module provides configuration management functionality for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the AI assistant.
    
    This class manages configuration for the AI assistant.
    """
    
    def __init__(self, config_path: str = "./config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration
        """
        self.config_path = Path(config_path)
        
        # Default configuration
        self.default_config = {
            # General
            "name": "AI Assistant",
            "version": "1.0.0",
            
            # Session
            "session_timeout": 300.0,  # 5 minutes
            
            # Models
            "model_path": "./models",
            "default_model": "text-generation",
            
            # Knowledge
            "knowledge_path": "./knowledge",
            
            # Voice
            "voice_enabled": True,
            "voice_language": "en",
            
            # Vision
            "vision_enabled": True,
            
            # User
            "user_path": "./users",
            
            # Logging
            "log_level": "INFO",
            "log_file": "./logs/assistant.log",
            
            # Advanced
            "debug_mode": False,
            "max_tokens": 100,
            "temperature": 0.7
        }
    
    async def initialize(self):
        """Initialize the configuration manager."""
        try:
            # Create configuration directory if it doesn't exist
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing configuration manager: {e}")
    
    async def load_config(self) -> Dict[str, Any]:
        """
        Load configuration.
        
        Returns:
            Configuration
        """
        try:
            # Configuration file
            config_file = self.config_path / "config.json"
            
            # Check if file exists
            if config_file.exists():
                # Load configuration
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                logger.info("Loaded configuration")
                
                # Merge with default configuration
                merged_config = self.default_config.copy()
                merged_config.update(config)
                
                return merged_config
            else:
                # Create default configuration
                config = self.default_config.copy()
                
                # Save configuration
                await self.save_config(config)
                
                logger.info("Created default configuration")
                
                return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
            # Return default configuration
            return self.default_config.copy()
    
    async def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration.
        
        Args:
            config: Configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Configuration file
            config_file = self.config_path / "config.json"
            
            # Save configuration
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info("Saved configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    async def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        try:
            # Load configuration
            config = await self.load_config()
            
            # Get value
            return config.get(key, default)
            
        except Exception as e:
            logger.error(f"Error getting configuration value: {e}")
            
            # Return default value
            return default
    
    async def set_config_value(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load configuration
            config = await self.load_config()
            
            # Set value
            config[key] = value
            
            # Save configuration
            return await self.save_config(config)
            
        except Exception as e:
            logger.error(f"Error setting configuration value: {e}")
            return False
    
    async def reset_config(self) -> bool:
        """
        Reset configuration to default.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save default configuration
            return await self.save_config(self.default_config.copy())
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False





from typing import Dict, Any, Optional, List, Set, Tuple, Union
from loguru import logger
import asyncio
import uuid
import json
import time
from datetime import datetime
import numpy as np
import threading
import queue
import os
import re
from pathlib import Path
import cv2
import pyaudio
import wave
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import whisper
from PIL import Image
import pytesseract
from pydantic import BaseModel, Field

from edgenative_umaas.utils.event_bus import EventBus
from edgenative_umaas.security.security_manager import SecurityManager

class ContextWindow(BaseModel):
    """Context window for maintaining conversation state and context."""
    session_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context_embeddings: Optional[List[float]] = None
    active_tasks: Dict[str, Any] = Field(default_factory=dict)
    screen_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    system_state: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message to the context window."""
        self.messages.append(message)
        # Limit the number of messages to prevent context explosion
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

class CognitiveCollaborationSystem:
    """
    Cognitive Collaboration System.
    
    A revolutionary yet attainable AI collaboration system that enables:
    1. Continuous contextual awareness through multimodal inputs
    2. Proactive assistance based on user behavior and system state
    3. Seamless voice-driven interaction with natural conversation
    4. Real-time screen understanding and intelligent automation
    5. Adaptive learning from user interactions and feedback
    """
    
    def __init__(self, event_bus: EventBus, security_manager: SecurityManager, config: Dict[str, Any]):
        """Initialize the Cognitive Collaboration System."""
        self.event_bus = event_bus
        self.security_manager = security_manager
        self.config = config
        
        # Core components
        self.active_sessions = {}  # session_id -> ContextWindow
        self.voice_streams = {}    # session_id -> audio stream
        self.screen_streams = {}   # session_id -> screen capture stream
        
        # AI models
        self.speech_recognizer = None  # Whisper model for speech recognition
        self.text_generator = None     # LLM for text generation
        self.vision_analyzer = None    # Vision model for screen understanding
        self.embedding_model = None    # Model for semantic embeddings
        
        # Processing queues
        self.voice_queue = queue.Queue()
        self.screen_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Knowledge base
        self.knowledge_vectors = []
        self.knowledge_texts = []
        
        # User behavior patterns
        self.user_patterns = {}  # user_id -> patterns
        
        # System capabilities registry
        self.capabilities = {
            "voice_commands": self._process_voice_command,
            "screen_analysis": self._analyze_screen,
            "task_automation": self._automate_task,
            "information_retrieval": self._retrieve_information,
            "code_generation": self._generate_code,
            "data_visualization": self._visualize_data,
            "system_monitoring": self._monitor_system,
            "predictive_assistance": self._provide_predictive_assistance
        }
        
        # Worker threads
        self.workers = []
    
    async def initialize(self) -> bool:
        """Initialize the Cognitive Collaboration System."""
        logger.info("Initializing Cognitive Collaboration System")
        
        try:
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Start worker threads
            self._start_workers()
            
            # Subscribe to events
            await self.event_bus.subscribe("user.joined", self._handle_user_joined)
            await self.event_bus.subscribe("user.left", self._handle_user_left)
            await self.event_bus.subscribe("user.message", self._handle_user_message)
            await self.event_bus.subscribe("system.state_changed", self._handle_system_state_changed)
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            logger.info("Cognitive Collaboration System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Collaboration System: {e}")
            return False
    
    async def _initialize_ai_models(self):
        """Initialize AI models using the most efficient approach for edge deployment."""
        # Use a background thread for model loading to avoid blocking
        def load_models():
            try:
                # 1. Initialize Whisper for speech recognition (small model for edge devices)
                self.speech_recognizer = whisper.load_model("base")
                
                # 2. Initialize vision model for screen understanding
                self.vision_analyzer = pipeline("image-classification", 
                                               model="microsoft/resnet-50")
                
                # 3. Initialize text embedding model
                self.embedding_model = pipeline("feature-extraction", 
                                               model="sentence-transformers/all-MiniLM-L6-v2")
                
                # 4. Initialize text generation model
                # Use a quantized model for edge efficiency
                model_name = "TheBloke/Llama-2-7B-Chat-GGML"
                self.text_generator = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto",
                    load_in_8bit=True  # 8-bit quantization for memory efficiency
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                logger.info("AI models loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading AI models: {e}")
                raise
        
        # Start model loading in background
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        
        # Wait for models to load with a timeout
        for _ in range(30):  # 30 second timeout
            if self.speech_recognizer and self.vision_analyzer and self.embedding_model and self.text_generator:
                return
            await asyncio.sleep(1)
        
        if not (self.speech_recognizer and self.vision_analyzer and self.embedding_model and self.text_generator):
            raise TimeoutError("Timed out waiting for AI models to load")
    
    def _start_workers(self):
        """Start worker threads for processing different input streams."""
        # 1. Voice processing worker
        voice_worker = threading.Thread(target=self._voice_processing_worker)
        voice_worker.daemon = True
        voice_worker.start()
        self.workers.append(voice_worker)
        
        # 2. Screen analysis worker
        screen_worker = threading.Thread(target=self._screen_processing_worker)
        screen_worker.daemon = True
        screen_worker.start()
        self.workers.append(screen_worker)
        
        # 3. Command execution worker
        command_worker = threading.Thread(target=self._command_processing_worker)
        command_worker.daemon = True
        command_worker.start()
        self.workers.append(command_worker)
        
        logger.info(f"Started {len(self.workers)} worker threads")
    
    async def _load_knowledge_base(self):
        """Load and index the knowledge base for quick retrieval."""
        knowledge_dir = Path(self.config.get("knowledge_dir", "./knowledge"))
        
        if not knowledge_dir.exists():
            logger.warning(f"Knowledge directory {knowledge_dir} does not exist. Creating it.")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Load all text files in the knowledge directory
        for file_path in knowledge_dir.glob("**/*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create embedding for the content
                embedding = self.embedding_model(content)
                embedding_vector = np.mean(embedding[0], axis=0)
                
                # Store the content and its embedding
                self.knowledge_texts.append(content)
                self.knowledge_vectors.append(embedding_vector)
                
            except Exception as e:
                logger.error(f"Error loading knowledge file {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge items")
    
    async def start_session(self, user_id: str) -> str:
        """Start a new collaboration session."""
        # Create a new session
        session_id = str(uuid.uuid4())
        
        # Initialize context window
        self.active_sessions[session_id] = ContextWindow(
            session_id=session_id,
            user_preferences=await self._load_user_preferences(user_id),
            system_state={"start_time": datetime.now().isoformat()}
        )
        
        # Start voice and screen streams
        await self._start_voice_stream(session_id)
        await self._start_screen_stream(session_id)
        
        logger.info(f"Started collaboration session {session_id} for user {user_id}")
        
        # Send welcome message
        await self.send_assistant_message(
            session_id,
            "I'm ready to collaborate with you. You can speak naturally or type commands. "
            "I'll observe your screen to provide contextual assistance when needed."
        )
        
        return session_id
    
    async def _start_voice_stream(self, session_id: str):
        """Start voice input stream for a session."""
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
            stream_callback=lambda in_data, frame_count, time_info, status_flags: 
                self._voice_callback(session_id, in_data, frame_count, time_info, status_flags)
        )
        
        # Store the stream
        self.voice_streams[session_id] = {
            "stream": stream,
            "pyaudio": p,
            "buffer": bytearray(),
            "last_process_time": time.time()
        }
        
        logger.info(f"Started voice stream for session {session_id}")
    
    def _voice_callback(self, session_id, in_data, frame_count, time_info, status_flags):
        """Callback for voice stream data."""
        if session_id not in self.voice_streams:
            return (in_data, pyaudio.paContinue)
        
        # Add data to buffer
        self.voice_streams[session_id]["buffer"].extend(in_data)
        
        # Process buffer if it's been at least 1 second since last processing
        current_time = time.time()
        if current_time - self.voice_streams[session_id]["last_process_time"] >= 1.0:
            # Copy buffer and clear it
            buffer_copy = self.voice_streams[session_id]["buffer"].copy()
            self.voice_streams[session_id]["buffer"] = bytearray()
            
            # Add to processing queue
            self.voice_queue.put((session_id, buffer_copy))
            
            # Update last process time
            self.voice_streams[session_id]["last_process_time"] = current_time
        
        return (in_data, pyaudio.paContinue)
    
    async def _start_screen_stream(self, session_id: str):
        """Start screen capture stream for a session."""
        # Initialize screen capture
        self.screen_streams[session_id] = {
            "active": True,
            "last_capture_time": 0,
            "capture_interval": 1.0,  # Capture every 1 second
            "last_image": None
        }
        
        # Start screen capture thread
        thread = threading.Thread(
            target=self._screen_capture_worker,
            args=(session_id,)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started screen capture for session {session_id}")
    
    def _screen_capture_worker(self, session_id: str):
        """Worker thread for screen capture."""
        try:
            import mss
            
            with mss.mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1]
                
                while session_id in self.screen_streams and self.screen_streams[session_id]["active"]:
                    current_time = time.time()
                    
                    # Capture screen at specified interval
                    if current_time - self.screen_streams[session_id]["last_capture_time"] >= self.screen_streams[session_id]["capture_interval"]:
                        # Capture screen
                        screenshot = sct.grab(monitor)
                        
                        # Convert to numpy array
                        img = np.array(screenshot)
                        
                        # Resize to reduce processing load
                        img = cv2.resize(img, (800, 600))
                        
                        # Store the image
                        self.screen_streams[session_id]["last_image"] = img
                        
                        # Add to processing queue
                        self.screen_queue.put((session_id, img.copy()))
                        
                        # Update last capture time
                        self.screen_streams[session_id]["last_capture_time"] = current_time
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in screen capture worker for session {session_id}: {e}")
    
    def _voice_processing_worker(self):
        """Worker thread for processing voice input."""
        while True:
            try:
                # Get item from queue
                session_id, audio_data = self.voice_queue.get(timeout=1.0)
                
                # Skip if speech recognizer is not initialized
                if not self.speech_recognizer:
                    self.voice_queue.task_done()
                    continue
                
                # Convert audio data to WAV format for Whisper
                with wave.open("temp_audio.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                
                # Transcribe audio
                result = self.speech_recognizer.transcribe("temp_audio.wav")
                transcription = result["text"].strip()
                
                # Skip if empty
                if not transcription:
                    self.voice_queue.task_done()
                    continue
                
                # Process the transcription
                logger.info(f"Voice input from session {session_id}: {transcription}")
                
                # Create a task to handle the voice input
                asyncio.run_coroutine_threadsafe(
                    self._handle_voice_input(session_id, transc
               # Create a task to handle the voice input
                asyncio.run_coroutine_threadsafe(
                    self._handle_voice_input(session_id, transcription),
                    asyncio.get_event_loop()
                )
                
                # Clean up
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in voice processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.voice_queue.task_done()
    
    def _screen_processing_worker(self):
        """Worker thread for processing screen captures."""
        while True:
            try:
                # Get item from queue
                session_id, image = self.screen_queue.get(timeout=1.0)
                
                # Skip if vision analyzer is not initialized
                if not self.vision_analyzer:
                    self.screen_queue.task_done()
                    continue
                
                # Process the screen capture
                screen_context = self._extract_screen_context(image)
                
                # Update context window with screen context
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].screen_context = screen_context
                
                # Check for significant changes that might require proactive assistance
                if self._should_provide_assistance(session_id, screen_context):
                    # Create a task to provide assistance
                    asyncio.run_coroutine_threadsafe(
                        self._provide_proactive_assistance(session_id, screen_context),
                        asyncio.get_event_loop()
                    )
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in screen processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.screen_queue.task_done()
    
    def _command_processing_worker(self):
        """Worker thread for processing commands."""
        while True:
            try:
                # Get item from queue
                session_id, command, args = self.command_queue.get(timeout=1.0)
                
                # Process the command
                logger.info(f"Processing command from session {session_id}: {command} {args}")
                
                # Execute the command
                result = None
                if command in self.capabilities:
                    try:
                        # Execute the command handler
                        result = self.capabilities[command](session_id, args)
                        
                        # Create a task to send the result
                        if result:
                            asyncio.run_coroutine_threadsafe(
                                self.send_assistant_message(session_id, result),
                                asyncio.get_event_loop()
                            )
                    
                    except Exception as e:
                        error_message = f"Error executing command {command}: {e}"
                        logger.error(error_message)
                        
                        # Send error message
                        asyncio.run_coroutine_threadsafe(
                            self.send_assistant_message(session_id, error_message),
                            asyncio.get_event_loop()
                        )
                else:
                    # Unknown command
                    unknown_command_message = f"Unknown command: {command}. Type 'help' for available commands."
                    
                    # Send unknown command message
                    asyncio.run_coroutine_threadsafe(
                        self.send_assistant_message(session_id, unknown_command_message),
                        asyncio.get_event_loop()
                    )
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in command processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.command_queue.task_done()
    
    def _extract_screen_context(self, image) -> Dict[str, Any]:
        """
        Extract context from screen capture using computer vision.
        
        This is a revolutionary feature that provides real-time understanding
        of what the user is seeing and doing on their screen.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "elements": [],
            "text": [],
            "recognized_apps": [],
            "active_window": None
        }
        
        try:
            # 1. Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Extract text using OCR
            text = pytesseract.image_to_string(Image.fromarray(rgb_image))
            if text:
                context["text"] = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 3. Detect UI elements using edge detection and contour analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours to identify UI elements
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify the element
                element_type = self._classify_ui_element(rgb_image[y:y+h, x:x+w])
                
                # Add to elements list
                context["elements"].append({
                    "id": f"element_{i}",
                    "type": element_type,
                    "bounds": {"x": x, "y": y, "width": w, "height": h}
                })
            
            # 4. Recognize applications using the vision analyzer
            # Convert image to PIL format for the vision model
            pil_image = Image.fromarray(rgb_image)
            
            # Get predictions
            predictions = self.vision_analyzer(pil_image)
            
            # Extract recognized applications
            for prediction in predictions:
                if prediction["score"] > 0.7:  # Confidence threshold
                    context["recognized_apps"].append({
                        "name": prediction["label"],
                        "confidence": prediction["score"]
                    })
            
            # 5. Determine active window based on UI analysis
            if context["elements"]:
                # Heuristic: The largest element near the top of the screen is likely the active window
                top_elements = sorted(context["elements"], key=lambda e: e["bounds"]["y"])[:5]
                largest_element = max(top_elements, key=lambda e: e["bounds"]["width"] * e["bounds"]["height"])
                context["active_window"] = largest_element["id"]
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting screen context: {e}")
            return context
    
    def _classify_ui_element(self, element_image) -> str:
        """Classify a UI element based on its appearance."""
        # This is a simplified version - in a real implementation, 
        # we would use a trained model for more accurate classification
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple heuristic classification
        if aspect_ratio > 5:
            return "menu_bar"
        elif aspect_ratio > 3:
            return "toolbar"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif w > 300 and h > 200:
            if avg_brightness > 200:
                return "dialog"
            else:
                return "window"
        elif w < 100 and h < 50:
            if avg_brightness > 200:
                return "button"
            else:
                return "icon"
        elif w > 200 and h < 50:
            return "text_field"
        else:
            return "panel"
    
    def _should_provide_assistance(self, session_id: str, screen_context: Dict[str, Any]) -> bool:
        """
        Determine if proactive assistance should be provided based on screen context.
        
        This is a revolutionary feature that allows the AI to offer help exactly
        when needed without explicit requests.
        """
        if session_id not in self.active_sessions:
            return False
        
        context_window = self.active_sessions[session_id]
        previous_context = context_window.screen_context
        
        # If no previous context, don't provide assistance yet
        if not previous_context:
            return False
        
        # Check for significant changes in active window
        if (previous_context.get("active_window") != screen_context.get("active_window") and
            screen_context.get("active_window") is not None):
            return True
        
        # Check for error messages in text
        error_patterns = ["error", "exception", "failed", "warning", "invalid"]
        for text in screen_context.get("text", []):
            if any(pattern in text.lower() for pattern in error_patterns):
                return True
        
        # Check for complex UI with many elements (user might need help navigating)
        if len(screen_context.get("elements", [])) > 15:
            # But only if this is a change from before
            if len(previous_context.get("elements", [])) < 10:
                return True
        
        # Check for recognized applications that might need assistance
        assistance_apps = ["terminal", "code editor", "database", "configuration"]
        for app in screen_context.get("recognized_apps", []):
            if any(assist_app in app["name"].lower() for assist_app in assistance_apps):
                # Only provide assistance if this is a newly detected app
                previous_apps = [a["name"] for a in previous_context.get("recognized_apps", [])]
                if app["name"] not in previous_apps:
                    return True
        
        return False
    
    async def _provide_proactive_assistance(self, session_id: str, screen_context: Dict[str, Any]):
        """
        Provide proactive assistance based on screen context.
        
        This is where the AI becomes truly collaborative by offering
        timely and relevant help without explicit requests.
        """
        if session_id not in self.active_sessions:
            return
        
        context_window = self.active_sessions[session_id]
        
        # Generate assistance message based on screen context
        assistance_message = await self._generate_assistance_message(session_id, screen_context)
        
        # Send the assistance message
        if assistance_message:
            await self.send_assistant_message(
                session_id,
                f"I noticed you might need help: {assistance_message}\n\nWould you like me to assist with this?"
            )
    
    async def _generate_assistance_message(self, session_id: str, screen_context: Dict[str, Any]) -> str:
        """Generate a contextual assistance message."""
        # Extract key information from screen context
        active_app = None
        if screen_context.get("recognized_apps"):
            active_app = screen_context["recognized_apps"][0]["name"]
        
        error_text = None
        for text in screen_context.get("text", []):
            error_patterns = ["error", "exception", "failed", "warning", "invalid"]
            if any(pattern in text.lower() for pattern in error_patterns):
                error_text = text
                break
        
        # Generate appropriate assistance
        if error_text:
            # Search knowledge base for similar errors
            similar_knowledge = await self._search_knowledge_base(error_text)
            if similar_knowledge:
                return f"I see an error: '{error_text}'. Based on my knowledge, this might be related to: {similar_knowledge}"
            else:
                return f"I notice you're encountering an error: '{error_text}'. Would you like me to help troubleshoot this?"
        
        elif active_app:
            if "terminal" in active_app.lower():
                return "I see you're working in the terminal. I can help with command suggestions or explain command outputs."
            
            elif "code" in active_app.lower() or "editor" in active_app.lower():
                return "I notice you're coding. I can help with code suggestions, debugging, or explaining concepts."
            
            elif "browser" in active_app.lower():
                return "I see you're browsing. I can help search for information or explain concepts on the current page."
            
            elif "database" in active_app.lower():
                return "I notice you're working with a database. I can help with query optimization or data modeling."
        
        # Default assistance based on UI complexity
        element_count = len(screen_context.get("elements", []))
        if element_count > 15:
            return "I notice you're working with a complex interface. I can help navigate or explain functionality."
        
        return None
    
    async def _search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for relevant information."""
        if not self.knowledge_vectors or not self.embedding_model:
            return None
        
        # Generate embedding for the query
        query_embedding = self.embedding_model(query)
        query_vector = np.mean(query_embedding[0], axis=0)
        
        # Calculate similarity with all knowledge items
        similarities = []
        for i, vector in enumerate(self.knowledge_vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most similar item if similarity is above threshold
        if similarities and similarities[0][1] > 0.7:
            index = similarities[0][0]
            # Return a snippet of the knowledge text
            text = self.knowledge_texts[index]
            # Extract a relevant snippet (first 200 characters)
            return text[:200] + "..." if len(text) > 200 else text
        
        return None
    
    async def _handle_voice_input(self, session_id: str, text: str):
        """Handle voice input from the user."""
        if session_id not in self.active_sessions:
            logger.warning(f"Received voice input for unknown session {session_id}")
            return
        
        logger.info(f"Processing voice input for session {session_id}: {text}")
        
        # Add to context window
        context_window = self.active_sessions[session_id]
        context_window.add_message({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat(),
            "type": "voice"
        })
        
        # Check if it's a command
        command_match = re.match(r"^([\w]+)(?:\s+(.+))?$", text.strip().lower())
        if command_match and command_match.group(1) in self.capabilities:
       # Check if it's a command
        command_match = re.match(r"^([\w]+)(?:\s+(.+))?$", text.strip().lower())
        if command_match and command_match.group(1) in self.capabilities:
            command = command_match.group(1)
            args = command_match.group(2) or ""
            
            # Add to command queue
            self.command_queue.put((session_id, command, args))
            
            return
        
        # Process as natural language
        await self._process_natural_language(session_id, text)
    
    async def _process_natural_language(self, session_id: str, text: str):
        """
        Process natural language input from the user.
        
        This is where the AI understands and responds to conversational input,
        making the interaction feel natural and human-like.
        """
        if session_id not in self.active_sessions:
            return
        
        context_window = self.active_sessions[session_id]
        
        # Prepare context for the language model
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps users with their tasks. You can see their screen and understand their voice commands. Be concise, helpful, and proactive."}
        ]
        
        # Add recent conversation history
        for msg in context_window.messages[-10:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # Add screen context if available
        if context_window.screen_context:
            screen_summary = self._summarize_screen_context(context_window.screen_context)
            messages.append({"role": "system", "content": f"Current screen context: {screen_summary}"})
        
        try:
            # Generate response using the language model
            response = await self._generate_response(messages)
            
            # Send the response
            await self.send_assistant_message(session_id, response)
            
            # Check for actionable insights in the response
            await self._extract_and_execute_actions(session_id, response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await self.send_assistant_message(
                session_id,
                "I'm sorry, I encountered an error while processing your request. Please try again."
            )
    
    def _summarize_screen_context(self, screen_context: Dict[str, Any]) -> str:
        """Summarize screen context for inclusion in LLM prompt."""
        summary_parts = []
        
        # Add active window
        if screen_context.get("active_window"):
            active_element = next(
                (e for e in screen_context.get("elements", []) 
                 if e.get("id") == screen_context["active_window"]),
                None
            )
            if active_element:
                summary_parts.append(f"Active window: {active_element.get('type', 'window')}")
        
        # Add recognized applications
        if screen_context.get("recognized_apps"):
            apps = [app["name"] for app in screen_context["recognized_apps"][:3]]
            summary_parts.append(f"Applications: {', '.join(apps)}")
        
        # Add UI element summary
        if screen_context.get("elements"):
            element_types = {}
            for element in screen_context["elements"]:
                element_type = element.get("type", "unknown")
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            element_summary = ", ".join(f"{count} {element_type}s" for element_type, count in element_types.items())
            summary_parts.append(f"UI elements: {element_summary}")
        
        # Add text summary (first few items)
        if screen_context.get("text"):
            text_items = screen_context["text"][:3]
            if text_items:
                text_summary = "; ".join(text_items)
                if len(text_summary) > 100:
                    text_summary = text_summary[:100] + "..."
                summary_parts.append(f"Visible text: {text_summary}")
        
        return " | ".join(summary_parts)
    
    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the language model."""
        if not self.text_generator or not self.tokenizer:
            return "I'm still initializing my language capabilities. Please try again in a moment."
        
        try:
            # Convert messages to a prompt
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.text_generator.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.text_generator.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            
            # Remove any assistant prefix
            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages to a prompt for the language model."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _extract_and_execute_actions(self, session_id: str, response: str):
        """
        Extract and execute actions from the assistant's response.
        
        This enables the AI to not just talk about actions but actually perform them,
        making it a true collaborative partner.
        """
        # Look for action patterns in the response
        action_patterns = [
            (r"I'll search for\s+(.+?)[\.\n]", "search"),
            (r"I'll analyze\s+(.+?)[\.\n]", "analyze_data"),
            (r"I'll create\s+(.+?)[\.\n]", "create"),
            (r"I'll generate\s+(.+?)[\.\n]", "generate_code"),
            (r"I'll show you\s+(.+?)[\.\n]", "visualize_data"),
            (r"I'll monitor\s+(.+?)[\.\n]", "monitor_system")
        ]
        
        for pattern, action in action_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                # Add to command queue
                self.command_queue.put((session_id, action, match))
                
                logger.info(f"Extracted action from response: {action} {match}")
    
    async def send_assistant_message(self, session_id: str, content: str):
        """Send a message from the assistant to the user."""
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot send message to unknown session {session_id}")
            return
        
        # Create message
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "text"
        }
        
        # Add to context window
        self.active_sessions[session_id].add_message(message)
        
        # Publish message event
        await self.event_bus.publish("assistant.message", {
            "session_id": session_id,
            "message": message
        })
        
        logger.info(f"Sent assistant message to session {session_id}: {content[:50]}...")
    
    async def end_session(self, session_id: str):
        """End a collaboration session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot end unknown session {session_id}")
            return False
        
        # Stop voice stream
        if session_id in self.voice_streams:
            try:
                self.voice_streams[session_id]["stream"].stop_stream()
                self.voice_streams[session_id]["stream"].close()
                self.voice_streams[session_id]["pyaudio"].terminate()
                del self.voice_streams[session_id]
            except Exception as e:
                logger.error(f"Error stopping voice stream: {e}")
        
        # Stop screen stream
        if session_id in self.screen_streams:
            try:
                self.screen_streams[session_id]["active"] = False
                del self.screen_streams[session_id]
            except Exception as e:
                logger.error(f"Error stopping screen stream: {e}")
        
        # Remove session
        del self.active_sessions[session_id]
        
        logger.info(f"Ended collaboration session {session_id}")
        return True
    
    # Command handlers
    def _process_voice_command(self, session_id: str, args: str) -> str:
        """Process a voice command."""
        return f"Processing voice command: {args}"
    
    def _analyze_screen(self, session_id: str, args: str) -> str:
        """Analyze the current screen."""
        if session_id not in self.screen_streams or not self.screen_streams[session_id].get("last_image") is not None:
            return "No screen capture available to analyze."
        
        # Get the last captured image
        image = self.screen_streams[session_id]["last_image"]
        
        # Extract context
        context = self._extract_screen_context(image)
        
        # Generate a human-readable analysis
        analysis = []
        
        if context.get("recognized_apps"):
            apps = [f"{app['name']} ({app['confidence']:.2f})" for app in context["recognized_apps"]]
            analysis.append(f"Recognized applications: {', '.join(apps)}")
        
        if context.get("elements"):
            element_types = {}
            for element in context["elements"]:
                element_type = element.get("type", "unknown")
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            element_summary = ", ".join(f"{count} {element_type}s" for element_type, count in element_types.items())
            analysis.append(f"UI elements: {element_summary}")
        
        if context.get("text"):
            text_count = len(context["text"])
            analysis.append(f"Detected {text_count} text elements")
            
            # Include a few examples
            if text_count > 0:
                examples = context["text"][:3]
                analysis.append(f"Text examples: {'; '.join(examples)}")
        
        return "\n".join(analysis)
    
    def _automate_task(self, session_id: str, args: str) -> str:
        """Automate a task based on screen context."""
        return f"Automating task: {args}"
    
    def _retrieve_information(self, session_id: str, args: str) -> str:
        """Retrieve information from knowledge base."""
        # Search knowledge base
        result = asyncio.run(self._search_knowledge_base(args))
        
        if result:
            return f"Found relevant information: {result}"
        else:
            return f"No relevant information found for: {args}"
    
    def _generate_code(self, session_id: str, args: str) -> str:
        """Generate code based on description."""
        return f"Generating code for: {args}"
    
    def _visualize_data(self, session_id: str, args: str) -> str:
        """Visualize data."""
        return f"Visualizing data: {args}"
    
    def _monitor_system(self, session_id: str, args: str) -> str:
        """Monitor system metrics."""
        return f"Monitoring system: {args}"
    
    def _provide_predictive_assistance(self, session_id: str, args: str) -> str:
        """Provide predictive assistance based on user patterns."""
        return f"Providing predictive assistance: {args}"
    
    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences."""
        # In a real implementation, this would load from a database
        return {
            "voice_enabled": True,
            "screen_capture_interval": 1.0,
            "proactive_assistance": True,
            "preferred_communication_style": "conversational"
        }
    
    async def _handle_user_joined(self, event_data: Dict[str, Any]):
        """Handle user joined event."""
        user_id = event_data.get("user_id")
        if not user_id:
            return
        
        # Start a session for the user
        session_id = await self.start_session(user_id)
        
        # Publish session started event
        await self.event_bus.publish("assistant.session_started", {
            "user_id": user_id,
            "session_id": session_id
        })
    
    async def _handle_user_left(self, event_data: Dict[str, Any]):
        """Handle user left event."""
        user_id = event_data.get("user_id")
        session_id = event_data.get("session_id")
        
        if not user_id or not session_id:
            return
        
        # End the session
        await self.end_session(session_id)
    
    async def _handle_user_message(self, event_data: Dict[str, Any]):
        """Handle user message event."""
        session_id = event_data.get("session_id")
        message = event_data.get("message")
        
        if not session_id or not message:
            return
        
        # Add to context window
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_message(message)
        
        # Process the message
        content = message.get("content", "")
        content_type = message.get("content_type", "text")
        
        if content_type == "text":
            await self._process_natural_language(session_id, content)
        elif content_type == "command":
            # Parse command
            parts = content.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Add to command queue
            self.command_queue.put((session_id, command, args))
    
    async def _handle_system_state_changed(self, event_data: Dict[str, Any]):
        """Handle system state changed event."""
        # Update system state for all active sessions
        for session_i
   async def _handle_system_state_changed(self, event_data: Dict[str, Any]):
        """Handle system state changed event."""
        # Update system state for all active sessions
        for session_id, context_window in self.active_sessions.items():
            # Update system state
            context_window.system_state.update(event_data)
            
            # Check if proactive notification is needed
            if self._should_notify_system_change(session_id, event_data):
                await self.send_assistant_message(
                    session_id,
                    f"System update: {self._format_system_change(event_data)}"
                )
    
    def _should_notify_system_change(self, session_id: str, event_data: Dict[str, Any]) -> bool:
        """Determine if a system change should trigger a notification."""
        # Check user preferences
        if session_id in self.active_sessions:
            user_preferences = self.active_sessions[session_id].user_preferences
            if not user_preferences.get("proactive_assistance", True):
                return False
        
        # Check importance of the change
        importance = event_data.get("importance", "low")
        if importance == "high":
            return True
        elif importance == "medium":
            # Only notify if the user isn't actively engaged
            if session_id in self.active_sessions:
                # Check if there was recent user activity (within last 5 minutes)
                recent_messages = [
                    msg for msg in self.active_sessions[session_id].messages[-10:]
                    if msg.get("role") == "user"
                ]
                
                if recent_messages:
                    last_message_time = datetime.fromisoformat(recent_messages[-1].get("timestamp", ""))
                    now = datetime.now()
                    time_diff = (now - last_message_time).total_seconds()
                    
                    # If user was active in the last 5 minutes, don't interrupt
                    if time_diff < 300:
                        return False
            
            return True
        
        # Low importance changes don't trigger notifications
        return False
    
    def _format_system_change(self, event_data: Dict[str, Any]) -> str:
        """Format a system change event for user notification."""
        event_type = event_data.get("type", "unknown")
        
        if event_type == "resource_warning":
            resource = event_data.get("resource", "unknown")
            level = event_data.get("level", "warning")
            details = event_data.get("details", "")
            return f"{level.upper()}: {resource} resource issue. {details}"
        
        elif event_type == "task_completed":
            task = event_data.get("task", "unknown")
            result = event_data.get("result", "completed")
            return f"Task '{task}' has been completed with result: {result}"
        
        elif event_type == "security_alert":
            alert = event_data.get("alert", "unknown")
            severity = event_data.get("severity", "medium")
            return f"{severity.upper()} security alert: {alert}"
        
        elif event_type == "update_available":
            component = event_data.get("component", "system")
            version = event_data.get("version", "unknown")
            return f"Update available for {component}: version {version}"
        
        else:
            # Generic formatting for unknown event types
            return ", ".join(f"{k}: {v}" for k, v in event_data.items() if k != "type")

class VoiceProcessor:
    """
    Voice processor for continuous speech recognition.
    
    This component enables the revolutionary natural voice interaction
    with the AI assistant.
    """
    
    def __init__(self, session_id: str, callback):
        """
        Initialize the voice processor.
        
        Args:
            session_id: Session ID
            callback: Callback function to call with transcribed text
        """
        self.session_id = session_id
        self.callback = callback
        self.active = False
        self.thread = None
        self.audio_queue = queue.Queue()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize Whisper model (will be loaded in the thread)
        self.model = None
    
    def start(self):
        """Start the voice processor."""
        if self.active:
            return
        
        self.active = True
        
        # Start processing thread
        self.thread = threading.Thread(target=self._processing_thread)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started voice processor for session {self.session_id}")
    
    def stop(self):
        """Stop the voice processor."""
        if not self.active:
            return
        
        self.active = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        # Clean up PyAudio
        self.p.terminate()
        self.p = None
        
        logger.info(f"Stopped voice processor for session {self.session_id}")
    
    def _processing_thread(self):
        """Processing thread for voice recognition."""
        try:
            # Load Whisper model
            self.model = whisper.load_model("base")
            
            # Open audio stream
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            stream.start_stream()
            
            # Process audio chunks
            while self.active:
                try:
                    # Get audio chunk from queue with timeout
                    audio_data = self.audio_queue.get(timeout=0.5)
                    
                    # Save to temporary file
                    with wave.open("temp_voice.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)
                    
                    # Transcribe
                    result = self.model.transcribe("temp_voice.wav")
                    text = result["text"].strip()
                    
                    # Call callback if text is not empty
                    if text:
                        self.callback(self.session_id, text)
                    
                    # Clean up
                    if os.path.exists("temp_voice.wav"):
                        os.remove("temp_voice.wav")
                    
                except queue.Empty:
                    # No audio data, continue
                    continue
                
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error in voice processing thread: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status_flags):
        """Callback for audio stream."""
        if self.active:
            # Add audio data to queue
            self.audio_queue.put(in_data)
        
        return (in_data, pyaudio.paContinue)

class ScreenAnalyzer:
    """
    Screen analyzer for understanding user's visual context.
    
    This component enables the revolutionary screen understanding
    capabilities of the AI assistant.
    """
    
    def __init__(self, vision_model, ocr_engine=None):
        """
        Initialize the screen analyzer.
        
        Args:
            vision_model: Vision model for image analysis
            ocr_engine: OCR engine for text extraction
        """
        self.vision_model = vision_model
        self.ocr_engine = ocr_engine or pytesseract
        
        # Initialize element classifiers
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize UI element classifiers."""
        # In a real implementation, this would load trained models
        # for UI element classification
        pass
    
    def analyze_screen(self, image) -> Dict[str, Any]:
        """
        Analyze a screen capture.
        
        Args:
            image: Screen capture image
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "elements": [],
            "text": [],
            "recognized_apps": [],
            "active_window": None
        }
        
        try:
            # 1. Convert to RGB for processing
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Extract text using OCR
            text = self.ocr_engine.image_to_string(Image.fromarray(rgb_image))
            if text:
                results["text"] = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 3. Detect UI elements
            elements = self._detect_ui_elements(rgb_image)
            results["elements"] = elements
            
            # 4. Recognize applications
            apps = self._recognize_applications(rgb_image)
            results["recognized_apps"] = apps
            
            # 5. Determine active window
            active_window = self._determine_active_window(elements)
            if active_window:
                results["active_window"] = active_window
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing screen: {e}")
            return results
    
    def _detect_ui_elements(self, image) -> List[Dict[str, Any]]:
        """Detect UI elements in the image."""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract element image
                element_image = image[y:y+h, x:x+w]
                
                # Classify element
                element_type = self._classify_element(element_image)
                
                # Add to elements list
                elements.append({
                    "id": f"element_{i}",
                    "type": element_type,
                    "bounds": {"x": x, "y": y, "width": w, "height": h}
                })
            
            return elements
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
            return elements
    
    def _classify_element(self, element_image) -> str:
        """Classify a UI element."""
        # This is a simplified version - in a real implementation, 
        # we would use a trained model for more accurate classification
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple heuristic classification
        if aspect_ratio > 5:
            return "menu_bar"
        elif aspect_ratio > 3:
            return "toolbar"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif w > 300 and h > 200:
            if avg_brightness > 200:
                return "dialog"
            else:
                return "window"
        elif w < 100 and h < 50:
            if avg_brightness > 200:
                return "button"
            else:
                return "icon"
        elif w > 200 and h < 50:
            return "text_field"
        else:
            return "panel"
    
    def _recognize_applications(self, image) -> List[Dict[str, Any]]:
        """Recognize applications in the image."""
        apps = []
        
        try:
            # Convert to PIL image for the vision model
            pil_image = Image.fromarray(image)
            
            # Get predictions from vision model
            predictions = self.vision_model(pil_image)
            
            # Process predictions
            for prediction in predictions:
                if prediction["score"] > 0.7:  # Confidence threshold
                    apps.append({
                        "name": prediction["label"],
                        "confidence": prediction["score"]
                    })
            
            return apps
            
        except Exception as e:
            logger.error(f"Error recognizing applications: {e}")
            return apps
    
    def _determine_active_window(self, elements: List[Dict[str, Any]]) -> Optional[str]:
        """Determine the active window from detected elements."""
        if not elements:
            return None
        
        # Heuristic: The largest element near the top of the screen is likely the active window
        window_elements = [e for e in elements if e["type"] in ["window", "dialog"]]
        
        if not window_elements:
            return None
        
        # Sort by y-coordinate (top to bottom)
        top_elements = sorted(window_elements, key=lambda e: e["bounds"]["y"])[:5]
        
        # Get the largest element
        largest_element = max(top_elements, key=lambda e: e["bounds"]["width"] * e["bounds"]["height"])
        
        return largest_element["id"]
class KnowledgeManager:
    """
    Knowledge manager for storing and retrieving information.
    
    This component enables the revolutionary knowledge capabilities
    of the AI assistant, allowing it to learn and adapt over time.
    """
    
    def __init__(self, embedding_model, storage_dir: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            embedding_model: Model for generating embeddings
            storage_dir: Directory for storing knowledge
        """
        self.embedding_model = embedding_model
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base
        self.knowledge_texts = []
        self.knowledge_vectors = []
        self.knowledge_metadata = []
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        # Load existing knowledge
        await self.load_knowledge()
    
    async def load_knowledge(self):
        """Load knowledge from storage."""
        # Clear existing knowledge
        self.knowledge_texts = []
        self.knowledge_vectors = []
        self.knowledge_metadata = []
        
        # Load knowledge index if it exists
        index_path = self.storage_dir / "knowledge_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                
                # Load knowledge items
                for item in index:
                    item_path = self.storage_dir / item["file"]
                    if item_path.exists():
                        with open(item_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        
                        # Load vector
                        vector_path = self.storage_dir / item["vector_file"]
                        if vector_path.exists():
                            vector = np.load(vector_path)
                            
                            # Add to knowledge base
                            self.knowledge_texts.append(text)
                            self.knowledge_vectors.append(vector)
                            self.knowledge_metadata.append(item["metadata"])
            
            except Exception as e:
                logger.error(f"Error loading knowledge index: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge items")
    
    async def add_knowledge(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add knowledge to the knowledge base.
        
        Args:
            text: Knowledge text
            metadata: Metadata for the knowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_model(text)
            vector = np.mean(embedding[0], axis=0)
            
            # Generate unique ID
            knowledge_id = str(uuid.uuid4())
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {}
            
            metadata["id"] = knowledge_id
            metadata["timestamp"] = datetime.now().isoformat()
            
            # Save text
            text_file = f"knowledge_{knowledge_id}.txt"
            with open(self.storage_dir / text_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Save vector
            vector_file = f"vector_{knowledge_id}.npy"
            np.save(self.storage_dir / vector_file, vector)
            
            # Add to knowledge base
            self.knowledge_texts.append(text)
            self.knowledge_vectors.append(vector)
            self.knowledge_metadata.append(metadata)
            
            # Update index
            await self._update_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        if not self.knowledge_vectors:
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model(query)
            query_vector = np.mean(query_embedding[0], axis=0)
            
            # Calculate similarity with all knowledge items
            similarities = []
            for i, vector in enumerate(self.knowledge_vectors):
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k results
            results = []
            for i, similarity in similarities[:top_k]:
                if similarity > 0.5:  # Similarity threshold
                    results.append({
                        "text": self.knowledge_texts[i],
                        "similarity": float(similarity),
                        "metadata": self.knowledge_metadata[i]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def _update_index(self):
        """Update the knowledge index."""
        try:
            # Create index
            index = []
            for i, metadata in enumerate(self.knowledge_metadata):
                knowledge_id = metadata.get("id", str(uuid.uuid4()))
                index.append({
                    "id": knowledge_id,
                    "file": f"knowledge_{knowledge_id}.txt",
                    "vector_file": f"vector_{knowledge_id}.npy",
                    "metadata": metadata
                })
            
            # Save index
            with open(self.storage_dir / "knowledge_index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating knowledge index: {e}")

class TaskAutomator:
    """
    Task automator for automating user tasks.
    
    This component enables the revolutionary automation capabilities
    of the AI assistant, allowing it to perform tasks on behalf of the user.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the task automator.
        
        Args:
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus
        
        # Automation capabilities
        self.capabilities = {
            "open_application": self._open_application,
            "close_application": self._close_application,
            "click": self._click,
            "type_text": self._type_text,
            "copy_text": self._copy_text,
            "paste_text": self._paste_text,
            "save_file": self._save_file,
            "open_file": self._open_file,
            "search": self._search,
            "navigate_to": self._navigate_to
        }
    
    async def execute_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task_type: Type of task to execute
            params: Task parameters
            
        Returns:
            Task result
        """
        if task_type not in self.capabilities:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
        
        try:
            # Execute the task
            result = await self.capabilities[task_type](params)
            
            # Publish task completed event
            await self.event_bus.publish("task.completed", {
                "type": "task_completed",
                "task": task_type,
                "params": params,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error executing task {task_type}: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _open_application(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open an application."""
        app_name = params.get("name")
        if not app_name:
            raise ValueError("Application name is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to open the application
        
        return {
            "status": "opened",
            "app_name": app_name
        }
    
    async def _close_application(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Close an application."""
        app_name = params.get("name")
        if not app_name:
            raise ValueError("Application name is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to close the application
        
        return {
            "status": "closed",
            "app_name": app_name
        }
    
    async def _click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Click at a position or on an element."""
        # Check if we have coordinates
        if "x" in params and "y" in params:
            x = params.get("x")
            y = params.get("y")
            
            # In a real implementation, this would use platform-specific APIs
            # to click at the specified coordinates
            
            return {
                "status": "clicked",
                "position": {"x": x, "y": y}
            }
        
        # Check if we have an element
        elif "element_id" in params:
            element_id = params.get("element_id")
            
            # In a real implementation, this would use platform-specific APIs
            # to click on the specified element
            
            return {
                "status": "clicked",
                "element_id": element_id
            }
        
        else:
            raise ValueError("Either coordinates (x, y) or element_id is required")
    
    async def _type_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Type text."""
        text = params.get("text")
        if not text:
            raise ValueError("Text is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to type the specified text
        
        return {
            "status": "typed",
            "text": text
        }
    
    async def _copy_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Copy text to clipboard."""
        text = params.get("text")
        if not text:
            raise ValueError("Text is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to copy the specified text to the clipboard
        
        return {
            "status": "copied",
            "text": text
        }
    
    async def _paste_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Paste text from clipboard."""
        # In a real implementation, this would use platform-specific APIs
        # to paste text from the clipboard
        
        return {
            "status": "pasted"
        }
    
    async def _save_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Save a file."""
        path = params.get("path")
        content = params.get("content")
        
        if not path:
            raise ValueError("File path is required")
        
        if content is None:
            raise ValueError("File content is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to save the file
        
        return {
            "status": "saved",
            "path": path
        }
    
    async def _open_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a file."""
        path = params.get("path")
        if not path:
            raise ValueError("File path is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to open the file
        
        return {
            "status": "opened",
            "path": path
        }
    
    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a search."""
        query = params.get("query")
        if not query:
            raise ValueError("Search query is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to perform the search
        
        return {
            "status": "searched",
            "query": query
        }
    
    async def _navigate_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a URL or location."""
        url = params.get("url")
        if not url:
            raise ValueError("URL is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to navigate to the URL
        
        return {
            "status": "navigated",
            "url": url
        }

class UserModelManager:
    """
    User model manager for tracking user preferences and behavior.
    
    This component enables the revolutionary personalization capabilities
    of the AI assistant, allowing it to adapt to each user's unique needs.
    """
    
    def __init__(self, storage_dir: str = "./user_models"):
        """
        Initialize the user model manager.
        
        Args:
            storage_dir: Directory for storing user models
        """
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # User models
        self.user_models = {}
    
    async def initialize(self):
        """Initialize the user model manager."""
        # Load existing user models
        await self.load_user_models()
    
    async def load_user_models(self):
        """Load user models from storage."""
        # Clear existing models
        self.user_models = {}
        
        # Load user models
        for file_path in self.storage_dir.glob("user_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    user_model = json.load(f)
                
                user_id = user_model.get("user_id")
                if user_id:
                    self.user_models[user_id] = user_model
            
            except Exception as e:
                logger.error(f"Error loading user model {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.user_models)} user models")
    
    async def get_user_model(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model
        """
        # Return existing model if available
        if user_id in self.user_models:
            return self.user_models[user_id]
        
        # Create new model
        user_model = {
            "user_id": user_id,
            "preferences": {
                "voice_enabled": True,
                "screen_capture_interval": 1.0,
                "proactive_assistance": True,
                "preferred_communication_style": "conversational"
            },
            "behavior_patterns": {},
            "interaction_history": [],
            "created_at": datetime.now().isoformat(),
            "
           "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save the model
        self.user_models[user_id] = user_model
        await self._save_user_model(user_id)
        
        return user_model
    
    async def update_user_preference(self, user_id: str, preference: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            preference: Preference name
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        # Get user model
        user_model = await self.get_user_model(user_id)
        
        # Update preference
        user_model["preferences"][preference] = value
        user_model["updated_at"] = datetime.now().isoformat()
        
        # Save the model
        return await self._save_user_model(user_id)
    
    async def record_interaction(self, user_id: str, interaction_type: str, details: Dict[str, Any]) -> bool:
        """
        Record a user interaction.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            details: Interaction details
            
        Returns:
            True if successful, False otherwise
        """
        # Get user model
        user_model = await self.get_user_model(user_id)
        
        # Add interaction to history
        interaction = {
            "type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        user_model["interaction_history"].append(interaction)
        
        # Limit history size
        if len(user_model["interaction_history"]) > 1000:
            user_model["interaction_history"] = user_model["interaction_history"][-1000:]
        
        # Update behavior patterns
        await self._update_behavior_patterns(user_model)
        
        # Save the model
        return await self._save_user_model(user_id)
    
    async def _update_behavior_patterns(self, user_model: Dict[str, Any]):
        """Update behavior patterns based on interaction history."""
        # Extract interaction history
        interactions = user_model["interaction_history"]
        
        # Skip if not enough interactions
        if len(interactions) < 10:
            return
        
        # Update patterns
        patterns = {}
        
        # 1. Preferred interaction times
        hour_counts = {}
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction["timestamp"])
            hour = timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours
        total_interactions = len(interactions)
        peak_hours = [hour for hour, count in hour_counts.items() if count > total_interactions * 0.1]
        patterns["preferred_hours"] = peak_hours
        
        # 2. Preferred interaction types
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction["type"]
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        # Find preferred types
        preferred_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        patterns["preferred_types"] = [t[0] for t in preferred_types]
        
        # 3. Response patterns
        response_times = []
        for i in range(1, len(interactions)):
            if interactions[i]["type"] == "assistant_message" and interactions[i-1]["type"] == "user_message":
                prev_time = datetime.fromisoformat(interactions[i-1]["timestamp"])
                curr_time = datetime.fromisoformat(interactions[i]["timestamp"])
                response_time = (curr_time - prev_time).total_seconds()
                response_times.append(response_time)
        
        if response_times:
            patterns["avg_response_time"] = sum(response_times) / len(response_times)
        
        # 4. Common queries
        queries = [
            interaction["details"].get("content", "")
            for interaction in interactions
            if interaction["type"] == "user_message"
        ]
        
        # Extract common words
        word_counts = {}
        for query in queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns["common_words"] = [w[0] for w in common_words]
        
        # Update user model
        user_model["behavior_patterns"] = patterns
    
    async def _save_user_model(self, user_id: str) -> bool:
        """Save a user model to storage."""
        if user_id not in self.user_models:
            return False
        
        try:
            # Get user model
            user_model = self.user_models[user_id]
            
            # Save to file
            file_path = self.storage_dir / f"user_{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_model, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user model for {user_id}: {e}")
            return False

class PredictiveEngine:
    """
    Predictive engine for anticipating user needs.
    
    This component enables the revolutionary predictive capabilities
    of the AI assistant, allowing it to anticipate user needs before
    they are explicitly expressed.
    """
    
    def __init__(self, user_model_manager: UserModelManager, knowledge_manager: KnowledgeManager):
        """
        Initialize the predictive engine.
        
        Args:
            user_model_manager: User model manager
            knowledge_manager: Knowledge manager
        """
        self.user_model_manager = user_model_manager
        self.knowledge_manager = knowledge_manager
    
    async def predict_user_needs(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict user needs based on context and user model.
        
        Args:
            user_id: User ID
            context: Current context
            
        Returns:
            List of predicted needs
        """
        # Get user model
        user_model = await self.user_model_manager.get_user_model(user_id)
        
        # Extract relevant information
        current_time = datetime.now()
        screen_context = context.get("screen_context", {})
        recent_messages = context.get("recent_messages", [])
        
        # Predictions
        predictions = []
        
        # 1. Time-based predictions
        time_predictions = await self._predict_time_based_needs(user_model, current_time)
        predictions.extend(time_predictions)
        
        # 2. Context-based predictions
        context_predictions = await self._predict_context_based_needs(user_model, screen_context)
        predictions.extend(context_predictions)
        
        # 3. Conversation-based predictions
        conversation_predictions = await self._predict_conversation_based_needs(user_model, recent_messages)
        predictions.extend(conversation_predictions)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predictions
    
    async def _predict_time_based_needs(self, user_model: Dict[str, Any], current_time: datetime) -> List[Dict[str, Any]]:
        """Predict needs based on time patterns."""
        predictions = []
        
        # Get behavior patterns
        patterns = user_model.get("behavior_patterns", {})
        preferred_hours = patterns.get("preferred_hours", [])
        
        # Check if current hour is a preferred hour
        current_hour = current_time.hour
        if current_hour in preferred_hours:
            # Predict based on common activities during this hour
            hour_interactions = [
                interaction for interaction in user_model.get("interaction_history", [])
                if datetime.fromisoformat(interaction["timestamp"]).hour == current_hour
            ]
            
            if hour_interactions:
                # Count interaction types
                type_counts = {}
                for interaction in hour_interactions:
                    interaction_type = interaction["type"]
                    type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
                
                # Find most common type
                most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                # Add prediction
                predictions.append({
                    "type": "time_based",
                    "need": f"typical_{most_common_type}",
                    "description": f"User typically performs {most_common_type} at this time",
                    "confidence": 0.7
                })
        
        return predictions
    
    async def _predict_context_based_needs(self, user_model: Dict[str, Any], screen_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict needs based on screen context."""
        predictions = []
        
        # Check for recognized applications
        recognized_apps = screen_context.get("recognized_apps", [])
        if recognized_apps:
            app_name = recognized_apps[0]["name"].lower()
            
            # Predict based on application
            if "code" in app_name or "editor" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "coding_assistance",
                    "description": "User might need help with coding",
                    "confidence": 0.8
                })
            
            elif "terminal" in app_name or "command" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "command_assistance",
                    "description": "User might need help with terminal commands",
                    "confidence": 0.8
                })
            
            elif "browser" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "information_search",
                    "description": "User might need help finding information",
                    "confidence": 0.7
                })
            
            elif "document" in app_name or "word" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "writing_assistance",
                    "description": "User might need help with writing",
                    "confidence": 0.7
                })
        
        # Check for error messages
        error_detected = False
        for text in screen_context.get("text", []):
            error_patterns = ["error", "exception", "failed", "warning", "invalid"]
            if any(pattern in text.lower() for pattern in error_patterns):
                error_detected = True
                break
        
        if error_detected:
            predictions.append({
                "type": "context_based",
                "need": "error_resolution",
                "description": "User might need help resolving an error",
                "confidence": 0.9
            })
        
        return predictions
    
    async def _predict_conversation_based_needs(self, user_model: Dict[str, Any], recent_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict needs based on recent conversation."""
        predictions = []
        
        # Skip if no recent messages
        if not recent_messages:
            return predictions
        
        # Extract user messages
        user_messages = [
            msg["content"] for msg in recent_messages
            if msg.get("role") == "user"
        ]
        
        # Skip if no user messages
        if not user_messages:
            return predictions
        
        # Analyze last user message
        last_message = user_messages[-1].lower()
        
        # Check for question patterns
        question_patterns = ["how", "what", "why", "when", "where", "can", "could", "would", "will", "?"]
        is_question = any(pattern in last_message for pattern in question_patterns)
        
        if is_question:
            # Search knowledge base for relevant information
            knowledge_results = await self.knowledge_manager.search_knowledge(last_message)
            
            if knowledge_results:
                predictions.append({
                    "type": "conversation_based",
                    "need": "knowledge_retrieval",
                    "description": "User is asking a question that might be answered from knowledge base",
                    "confidence": 0.8,
                    "knowledge": knowledge_results[0]
                })
        
        # Check for request patterns
        request_patterns = ["can you", "could you", "please", "help me", "i need", "show me"]
        is_request = any(pattern in last_message for pattern in request_patterns)
        
        if is_request:
            predictions.append({
                "type": "conversation_based",
                "need": "task_assistance",
                "description": "User is requesting assistance with a task",
                "confidence": 0.7
            })
        
        return predictions

class ContextWindow:
    """
    Context window for tracking conversation and context.
    
    This component enables the AI assistant to maintain context
    across interactions, providing a more coherent and personalized
    experience.
    """
    
    def __init__(self, user_id: str, session_id: str, max_messages: int = 100):
        """
        Initialize the context window.
        
        Args:
            user_id: User ID
            session_id: Session ID
            max_messages: Maximum number of messages to keep
        """
        self.user_id = user_id
        self.session_id = session_id
        self.max_messages = max_messages
        
        # Messages
        self.messages = []
        
        # Context
        self.screen_context = {}
        self.system_state = {}
        self.user_preferences = {}
    
    def add_message(self, message: Dict[str, Any]):
        """
        Add a message to the context window.
        
        Args:
            message: Message to add
        """
        # Add message
        self.messages.append(message)
        
        # Trim if necessary
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages.
        
        Args:
            count: Number of messages to get
            
        Returns:
            List of recent messages
        """
        return self.messages[-count:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.
        
        Returns:
            Context summary
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "recent_messages": self.get_recent_messages(5),
            "screen_context": self.screen_context,
            "system_state": self.system_state,
            "user_preferences": self.user_preferences
        }

# Main entry point
def create_cognitive_collaboration_system(config: Dict[str, Any] = None) -> CognitiveCollaborationSystem:
    """
    Create a cognitive collaboration system.
    
    Args:
        config: Configuration
   Create a cognitive collaboration system.
    
    Args:
        config: Configuration
        
    Returns:
        Cognitive collaboration system
    """
    # Default configuration
    default_config = {
        "model_path": "./models",
        "knowledge_dir": "./knowledge",
        "user_models_dir": "./user_models",
        "log_level": "INFO"
    }
    
    # Merge configurations
    if config is None:
        config = {}
    
    merged_config = {**default_config, **config}
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, merged_config["log_level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create event bus
    event_bus = EventBus()
    
    # Create system
    system = CognitiveCollaborationSystem(
        model_path=merged_config["model_path"],
        event_bus=event_bus
    )
    
    return system

# Example usage
if __name__ == "__main__":
    # Create system
    system = create_cognitive_collaboration_system()
    
    # Initialize system
    asyncio.run(system.initialize())
    
    # Start system
    asyncio.run(system.start())
    
    try:
        # Run forever
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        # Stop system
        asyncio.run(system.stop())
"""
Event bus for communication between components.

This module provides a simple event bus implementation for asynchronous
communication between components of the AI assistant.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class EventBus:
    """
    Event bus for communication between components.
    
    This class provides a simple publish-subscribe mechanism for
    asynchronous communication between components.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.event_history = {}
        self.max_history = 100
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        # Add to history
        if event_type not in self.event_history:
            self.event_history[event_type] = []
        
        self.event_history[event_type].append(event_data)
        
        # Trim history if necessary
        if len(self.event_history[event_type]) > self.max_history:
            self.event_history[event_type] = self.event_history[event_type][-self.max_history:]
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event subscriber for {event_type}: {e}")
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        """
        Subscribe to an event.
        
        Args:
            event_type: Type of event
            callback: Callback function
            
        Returns:
            Unsubscribe function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        
        # Return unsubscribe function
        def unsubscribe():
            if event_type in self.subscribers and callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
        
        return unsubscribe
    
    def get_recent_events(self, event_type: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events of a specific type.
        
        Args:
            event_type: Type of event
            count: Number of events to get
            
        Returns:
            List of recent events
        """
        if event_type not in self.event_history:
            return []
        
        return self.event_history[event_type][-count:]
"""
AI Assistant package for Edge-Native UMaaS.

This package provides the AI assistant capabilities for the Edge-Native
Universal Multimodal Assistant as a Service (UMaaS) platform.
"""

from .cognitive_collaboration_system import CognitiveCollaborationSystem, create_cognitive_collaboration_system
from .event_bus import EventBus

__all__ = [
    'CognitiveCollaborationSystem',
    'create_cognitive_collaboration_system',
    'EventBus'
]
#!/usr/bin/env python3
"""
Command-line interface for Edge-Native UMaaS.

This module provides a command-line interface for interacting with
the Edge-Native Universal Multimodal Assistant as a Service (UMaaS) platform.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any

from edgenative_umaas.ai_assistant import create_cognitive_collaboration_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class UMaaSCLI:
    """Command-line interface for UMaaS."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.system = None
        self.session_id = None
        self.user_id = os.environ.get("USER", "default_user")
    
    async def initialize(self, config_path: str = None):
        """
        Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Create system
        self.system = create_cognitive_collaboration_system(config)
        
        # Initialize system
        await self.system.initialize()
        
        # Start system
        await self.system.start()
        
        # Subscribe to assistant messages
        self.system.event_bus.subscribe("assistant.message", self._handle_assistant_message)
        
        # Start session
        self.session_id = await self.system.start_session(self.user_id)
        
        logger.info(f"Started session {self.session_id} for user {self.user_id}")
    
    async def _handle_assistant_message(self, event_data: Dict[str, Any]):
        """Handle assistant message event."""
        message = event_data.get("message", {})
        content = message.get("content", "")
        
        # Print message
        print(f"\nAssistant: {content}\n")
    
    async def process_command(self, command: str):
        """
        Process a command.
        
        Args:
            command: Command to process
        """
        if not self.system or not self.session_id:
            print("System not initialized")
            return
        
        if command.lower() in ["exit", "quit"]:
            # End session
            await self.system.end_session(self.session_id)
            
            # Stop system
            await self.system.stop()
            
            return False
        
        # Process command
        await self.system.process_user_input(self.session_id, command)
        
        return True
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("Edge-Native UMaaS CLI")
        print("Type 'exit' or 'quit' to exit")
        print()
        
        # Welcome message
        print("Assistant: Hello! I'm your AI assistant. How can I help you today?")
        
        # Main loop
        running = True
        while running:
            try:
                # Get user input
                user_input = input("\nYou: ")
                
                # Process command
                result = await self.process_command(user_input)
                if result is False:
                    running = False
            
            except KeyboardInterrupt:
                print("\nExiting...")
                
                # End session
                await self.system.end_session(self.session_id)
                
                # Stop system
                await self.system.stop()
                
                running = False
            
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(f"\nError: {e}")

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Edge-Native UMaaS CLI")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create CLI
    cli = UMaaSCLI()
    
    # Run
    try:
        # Initialize
        asyncio.run(cli.initialize(args.config))
        
        # Run in interactive mode
        asyncio.run(cli.interactive_mode())
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
from setuptools import setup, find_packages

setup(
    name="edgenative-umaas",
    version="0.1.0",
    description="Edge-Native Universal Multimodal Assistant as a Service",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "opencv-python",
        "pillow",
        "pytesseract",
        "pyaudio",
        "whisper",
    ],
    entry_points={
        "console_scripts": [
            "umaas-cli=edgenative_umaas.cli:main",
        ],
    },
    python_requires=">=3.8",
)
"""
Utility functions for the AI assistant.

This module provides utility functions used by various components
of the AI assistant.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False

def capture_screen() -> Optional[np.ndarray]:
    """
    Capture the screen.
    
    Returns:
        Screen image as numpy array, or None if failed
    """
    try:
        # This is a simplified implementation that works on most platforms
        # In a real implementation, we would use platform-specific APIs
        # for better performance and reliability
        
        # Try to use mss for fast screen capture
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # Primary monitor
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except ImportError:
            pass
        
        # Fall back to PIL and ImageGrab
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return np.array(img)
        except ImportError:
            pass
        
        logger.warning("No screen capture method available")
        return None
        
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return None

def save_image(image: np.ndarray, directory: str, prefix: str = "image") -> Optional[str]:
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array
        directory: Directory to save to
        prefix: Filename prefix
        
    Returns:
        Path to saved image, or None if failed
    """
    try:
        # Ensure directory exists
        if not ensure_directory(directory):
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
    """
    Crop an image.
    
    Args:
        image: Image as numpy array
        x: X coordinate
        y: Y coordinate
        width: Width
        height: Height
        
    Returns:
        Cropped image, or None if failed
    """
    try:
        # Check bounds
        img_height, img_width = image.shape[:2]
        
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            logger.warning(f"Crop region ({x}, {y}, {width}, {height}) out of bounds ({img_width}, {img_height})")
            return None
        
        # Crop image
        return image[y:y+height, x:x+width]
        
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return None

def resize_image(image: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """
    Resize an image.
    
    Args:
        image: Image as numpy array
        width: Width
        height: Height
        
    Returns:
        Resized image, or None if failed
    """
    try:
        return cv2.resize(image, (width, height))
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

def extract_text_from_image(image: np.ndarray, lang: str = "eng") -> List[str]:
    """
    Extract text from an image using OCR.
    
    Args:
        image: Image as numpy array
        lang: Language code
        
    Returns:
        List of extracted text lines
    """
    try:
        # Convert to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Extract text using pytesseract
        import pytesseract
        text = pytesseract.image_to_string(pil_image, lang=lang)
        
        # Split into lines and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return lines
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return []

def measure_execution_time(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper

def format_timestamp(timestamp: float) -> str:
    """
    Format a timestamp.
    
    Args:
        timestamp: Timestamp in seconds
        
    Returns:
        Formatted timestamp
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    try:
        # Split into words
        words = text.lower().split()
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "that", "this", "these", "those", "it", "they", 
                        "he", "she", "we", "you", "i", "me", "my", "your", "our", "their"}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    try:
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def parse_command(text: str) -> Tuple[str, str]:
    """
    Parse a command.
    
    Args:
        text: Command text
        
    Returns:
        Tuple of (command, arguments)
    """
    parts = text.strip().split(maxsplit=1)
    
    if len(parts) == 0:
        return "", ""
    
    if len(parts) == 1:
        return parts[0].lower(), ""
    
    return parts[0].lower(), parts[1]

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    
    hours = minutes / 60
    return f"{hours:.1f} hours"

def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.1f} GB"
"""
Model management for the AI assistant.

This module provides functionality for loading and managing AI models
used by the assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for loading and managing AI models.
    
    This class provides functionality for loading and managing AI models
    used by the assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Loaded models
        self.models = {}
        self.tokenizers = {}
    
    async def load_model(self, model_name: str, model_type: str = "causal_lm", device: str = None) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """
        Load a model.
        
        Args:
            model_name: Model name or path
            model_type: Model type (causal_lm or seq2seq_lm)
            device: Device to load model on (cpu, cuda, or None for auto)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if model is already loaded
        if model_name in self.models and model_name in self.tokenizers:
            return self.models[model_name], self.tokenizers[model_name]
        
        try:
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} on {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_name)
            elif model_type == "seq2seq_lm":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Move model to device
            model = model.to(device)
            
            # Save model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, None
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.models:
            return False
        
        try:
            # Remove model and tokenizer
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def generate_text(self, model_name: str, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using a model.
        
        Args:
            model_name: Model name
            prompt: Prompt text
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated text
        """
        if model_name not in self.models or model_name not in self.tokenizers:
            # Try to load the model
            model, tokenizer = await self.load_model(model_name)
            if model is None or tokenizer is None:
                return ""
        else:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate text
"""
Configuration management for the AI assistant.

This module provides functionality for loading and managing configuration
for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the AI assistant.
    
    This class provides functionality for loading and managing configuration
    for the AI assistant.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        
        # Default configuration
        self.default_config = {
            "model_path": "./models",
            "knowledge_dir": "./knowledge",
            "user_models_dir": "./user_models",
            "log_level": "INFO",
            "voice": {
                "enabled": True,
                "model": "base",
                "language": "en"
            },
            "screen_capture": {
                "enabled": True,
                "interval": 1.0
            },
            "models": {
                "assistant": "gpt2",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "vision": "facebook/detr-resnet-50"
            },
            "system": {
                "max_sessions": 10,
                "session_timeout": 3600,
                "max_messages": 100
            }
        }
    
    async def load_config(self) -> Dict[str, Any]:
        """
        Load configuration.
        
        Returns:
            Configuration dictionary
        """
        # Start with default configuration
        self.config = self.default_config.copy()
        
        # Load configuration from file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                
                # Merge configurations
                self._merge_config(self.config, file_config)
                
                logger.info(f"Loaded configuration from {self.config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")
        
        return self.config
    
    async def save_config(self) -> bool:
        """
        Save configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        # Split key into parts
        parts = key.split(".")
        
        # Navigate through configuration
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split key into parts
            parts = key.split(".")
            
            # Navigate through configuration
            config = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set value
            config[parts[-1]] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value for {key}: {e}")
            return False
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Merge two configuration dictionaries.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._merge_config(target[key], value)
            else:
                # Set value
                target[key] = value
   def _cleanup_sessions(self):
        """Clean up expired sessions."""
        # Find expired sessions
        expired_session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        
        # End expired sessions
        for session_id in expired_session_ids:
            self.end_session(session_id)
        
        if expired_session_ids:
            logger.info(f"Cleaned up {len(expired_session_ids)} expired sessions")
"""
Voice processing for the AI assistant.

This module provides functionality for speech recognition and synthesis
for the AI assistant.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for the AI assistant.
    
    This class provides functionality for speech recognition and synthesis
    for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the voice processor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Speech recognition model
        self.stt_model = None
        
        # Speech synthesis model
        self.tts_model = None
    
    async def initialize(self):
        """Initialize the voice processor."""
        try:
            # Initialize speech recognition
            await self._initialize_stt()
            
            # Initialize speech synthesis
            await self._initialize_tts()
            
        except Exception as e:
            logger.error(f"Error initializing voice processor: {e}")
    
    async def _initialize_stt(self):
        """Initialize speech recognition."""
        try:
            # Try to import whisper
            import whisper
            
            # Load model
            model_name = "base"
            logger.info(f"Loading Whisper model: {model_name}")
            self.stt_model = whisper.load_model(model_name)
            
            logger.info("Speech recognition initialized")
            
        except ImportError:
            logger.warning("Whisper not available, speech recognition disabled")
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    async def _initialize_tts(self):
        """Initialize speech synthesis."""
        try:
            # Try to import TTS
            from TTS.api import TTS
            
            # Load model
            logger.info("Loading TTS model")
            self.tts_model = TTS(gpu=torch.cuda.is_available())
            
            logger.info("Speech synthesis initialized")
            
        except ImportError:
            logger.warning("TTS not available, speech synthesis disabled")
        except Exception as e:
            logger.error(f"Error initializing speech synthesis: {e}")
    
    async def recognize_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            Recognition result
        """
        if self.stt_model is None:
            return {
                "success": False,
                "error": "Speech recognition not available"
            }
        
        try:
            # Recognize speech
            result = self.stt_model.transcribe(audio_data)
            
            return {
                "success": True,
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
            
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def synthesize_speech(self, text: str, language: str = "en") -> Tuple[Optional[np.ndarray], int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        if self.tts_model is None:
            return None, 0
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize speech
            self.tts_model.tts_to_file(
                text=text,
                file_path=temp_path,
                language=language
            )
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None, 0
    
    async def record_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], int]:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        try:
            # Try to import pyaudio
            import pyaudio
            import wave
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            # Record audio
            logger.info(f"Recording audio for {duration} seconds")
            frames = []
            for _ in range(0, int(sample_rate / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to file
            wf = wave.open(temp_path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except ImportError:
            logger.warning("PyAudio not available, audio recording disabled")
            return None, 0
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None, 0
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Play audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to import pyaudio
            import pyaudio
            import wave
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to file
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Open wave file
            wf = wave.open(temp_path, "rb")
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Play audio
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return True
            
        except ImportError:
            logger.warning("PyAudio not available, audio playback disabled")
            return False
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
"""
Vision processing for the AI assistant.

This module provides functionality for computer vision tasks
for the AI assistant.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import capture_screen, extract_text_from_image

# Configure logging
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Vision processor for the AI assistant.
    
    This class provides functionality for computer vision tasks
    for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the vision processor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Object detection model
        self.detection_model = None
        
        # Image classification model
        self.classification_model = None
    
    async def initialize(self):
        """Initialize the vision processor."""
        try:
            # Initialize object detection
            await self._initialize_detection()
            
            # Initialize image classification
            await self._initialize_classification()
            
        except Exception as e:
            logger.error(f"Error initializing vision processor: {e}")
    
    async def _initialize_detection(self):
        """Initialize object detection."""
        try:
            # Try to import transformers
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            # Load model
            model_name = "facebook/detr-resnet-50"
            logger.info(f"Loading object detection model: {model_name}")
            
            self.detection_processor = DetrImageProcessor.from_pretrained(model_name)
            self.detection_model = DetrForObjectDetection.from_pretrained(model_name)
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detection_model = self.detection_model.to(device)
            
            logger.info("Object detection initialized")
            
        except ImportError:
            logger.warning("Transformers not available, object detection disabled")
        except Exception as e:
            logger.error(f"Error initializing object detection: {e}")
    
    async def _initialize_classification(self):
        """Initialize image classification."""
        try:
            # Try to import transformers
            from transformers import ViTForImageClassification, ViTImageProcessor
            
            # Load model
            model_name = "google/vit-base-patch16-224"
            logger.info(f"Loading image classification model: {model_name}")
            
            self.classification_processor = ViTImageProcessor.from_pretrained(model_name)
            self.classification_model = ViTForImageClassification.from_pretrained(model_name)
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classification_model = self.classification_model.to(device)
            
            logger.info("Image classification initialized")
            
        except ImportError:
            logger.warning("Transformers not available, image classification disabled")
        except Exception as e:
            logger.error(f"Error initializing image classification: {e}")
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Analysis result
        """
        # Capture screen
        screen = capture_screen()
        if screen is None:
            return {
                "success": False,
                "error": "Failed to capture screen"
            }
        
        # Analyze screen
        result = await self.analyze_image(screen)
        
        # Add screen capture
        result["screen"] = screen
        
        return result
    
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Analysis result
        """
        result = {
            "success": True,
            "text": [],
            "objects": [],
            "classification": []
        }
        
        try:
            # Extract text
            text = extract_text_from_image(image)
            result["text"] = text
            
            # Detect objects
            if self.detection_model is not None:
                objects = await self._detect_objects(image)
                result["objects"] = objects
            
            # Classify image
            if self.classification_model is not None:
                classification = await self._classify_image(image)
                result["classification"] = classification
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected objects
        """
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare image for model
            inputs = self.detection_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.detection_model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.tensor([pil_image.size[::-1]])
            target_sizes = target_sizes.to(self.detection_model.device)
            results = self.detection_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )[0]
            
            # Extract results
            objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                objects.append({
                    "label": self.detection_model
                   "label": self.detection_model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": {
                        "x": box[0].item(),
                        "y": box[1].item(),
                        "width": box[2].item() - box[0].item(),
                        "height": box[3].item() - box[1].item()
                    }
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    async def _classify_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of classifications
        """
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare image for model
            inputs = self.classification_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.classification_model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Extract results
            classifications = []
            for prob, idx in zip(top5_prob, top5_indices):
                classifications.append({
                    "label": self.classification_model.config.id2label[idx.item()],
                    "score": prob.item()
                })
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return []
    
    async def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Extract results
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
            
            return face_list
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    async def recognize_app_windows(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize application windows in a screen image.
        
        Args:
            image: Screen image as numpy array
            
        Returns:
            List of recognized application windows
        """
        try:
            # This is a simplified implementation that uses basic image processing
            # In a real implementation, we would use platform-specific APIs
            # to get accurate window information
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            windows = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 100 or h < 100:
                    continue
                
                # Extract window region
                window_region = image[y:y+h, x:x+w]
                
                # Extract text from window title bar
                title_bar_region = window_region[:30, :]
                title_text = extract_text_from_image(title_bar_region)
                
                # Add window
                windows.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "title": title_text[0] if title_text else "Unknown"
                })
            
            return windows
            
        except Exception as e:
            logger.error(f"Error recognizing app windows: {e}")
            return []
    
    async def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected text regions
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            text_regions = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 10:
                    continue
                
                # Extract text region
                text_region = image[y:y+h, x:x+w]
                
                # Extract text
                text = extract_text_from_image(text_region)
                
                # Add text region
                if text:
                    text_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "text": text[0]
                    })
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    async def recognize_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize UI elements in a screen image.
        
        Args:
            image: Screen image as numpy array
            
        Returns:
            List of recognized UI elements
        """
        try:
            # This is a simplified implementation that uses basic image processing
            # In a real implementation, we would use more sophisticated techniques
            # or platform-specific APIs
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            ui_elements = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 10:
                    continue
                
                # Extract UI element region
                element_region = image[y:y+h, x:x+w]
                
                # Determine element type
                element_type = "unknown"
                
                # Check if it's a button
                if 20 <= w <= 200 and 20 <= h <= 50:
                    element_type = "button"
                
                # Check if it's a text field
                elif 100 <= w <= 400 and 20 <= h <= 40:
                    element_type = "text_field"
                
                # Check if it's a checkbox
                elif 10 <= w <= 30 and 10 <= h <= 30:
                    element_type = "checkbox"
                
                # Extract text
                text = extract_text_from_image(element_region)
                
                # Add UI element
                ui_elements.append({
                    "type": element_type,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "text": text[0] if text else ""
                })
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error recognizing UI elements: {e}")
            return []
"""
Knowledge management for the AI assistant.

This module provides functionality for managing knowledge
for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge manager for the AI assistant.
    
    This class provides functionality for managing knowledge
    for the AI assistant.
    """
    
    def __init__(self, knowledge_dir: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            knowledge_dir: Path to knowledge directory
        """
        self.knowledge_dir = Path(knowledge_dir)
        
        # Create knowledge directory if it doesn't exist
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base
        self.knowledge_base = {}
        
        # Embeddings
        self.embeddings = {}
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        try:
            # Load knowledge base
            await self.load_knowledge_base()
            
        except Exception as e:
            logger.error(f"Error initializing knowledge manager: {e}")
    
    async def load_knowledge_base(self) -> bool:
        """
        Load knowledge base from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear knowledge base
            self.knowledge_base = {}
            self.embeddings = {}
            
            # Load knowledge files
            for file_path in self.knowledge_dir.glob("*.json"):
                try:
                    # Load knowledge file
                    with open(file_path, "r", encoding="utf-8") as f:
                        knowledge = json.load(f)
                    
                    # Add to knowledge base
                    knowledge_id = file_path.stem
                    self.knowledge_base[knowledge_id] = knowledge
                    
                    # Load embedding if available
                    embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
                    if embedding_path.exists():
                        self.embeddings[knowledge_id] = np.load(embedding_path)
                    
                except Exception as e:
                    logger.error(f"Error loading knowledge file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    async def save_knowledge_base(self) -> bool:
        """
        Save knowledge base to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save knowledge files
            for knowledge_id, knowledge in self.knowledge_base.items():
                try:
                    # Save knowledge file
                    file_path = self.knowledge_dir / f"{knowledge_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(knowledge, f, indent=2)
                    
                    # Save embedding if available
                    if knowledge_id in self.embeddings:
                        embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
                        np.save(embedding_path, self.embeddings[knowledge_id])
                    
                except Exception as e:
                    logger.error(f"Error saving knowledge file {file_path}: {e}")
            
            logger.info(f"Saved {len(self.knowledge_base)} knowledge items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    async def add_knowledge(self, knowledge: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            embedding: Knowledge embedding
            
        Returns:
            Knowledge ID
        """
        try:
            # Generate knowledge ID
            import uuid
            knowledge_id = str(uuid.uuid4())
            
            # Add to knowledge base
            self.knowledge_base[knowledge_id] = knowledge
            
            # Add embedding if available
            if embedding is not None:
                self.embeddings[knowledge_id] = embedding
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        return self.knowledge_base.get(knowledge_id)
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            embedding: Updated knowledge embedding
            
        Returns:
            True if successful, False otherwise
        """
        if knowledge_id not in self.knowledge_base:
            return False
        
        try:
            # Update knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Update embedding if available
            if embedding is not None:
                self.embeddings[knowledge_id] = embedding
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge {knowledge_id}: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        
       Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        if knowledge_id not in self.knowledge_base:
            return False
        
        try:
            # Remove from knowledge base
            del self.knowledge_base[knowledge_id]
            
            # Remove embedding if available
            if knowledge_id in self.embeddings:
                del self.embeddings[knowledge_id]
            
            # Remove knowledge files
            file_path = self.knowledge_dir / f"{knowledge_id}.json"
            if file_path.exists():
                os.remove(file_path)
            
            embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
            if embedding_path.exists():
                os.remove(embedding_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge {knowledge_id}: {e}")
            return False
    
    async def search_knowledge(self, query: str, embedding: Optional[np.ndarray] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            embedding: Query embedding
            max_results: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        try:
            results = []
            
            # Search by embedding if available
            if embedding is not None and self.embeddings:
                # Calculate similarity scores
                scores = {}
                for knowledge_id, knowledge_embedding in self.embeddings.items():
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, knowledge_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(knowledge_embedding)
                    )
                    scores[knowledge_id] = similarity
                
                # Sort by similarity
                sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
                
                # Add top results
                for knowledge_id in sorted_ids[:max_results]:
                    knowledge = self.knowledge_base[knowledge_id]
                    results.append({
                        "id": knowledge_id,
                        "knowledge": knowledge,
                        "score": float(scores[knowledge_id])
                    })
            
            # Search by text if no results or no embedding
            if not results:
                # Convert query to lowercase
                query_lower = query.lower()
                
                # Search knowledge base
                for knowledge_id, knowledge in self.knowledge_base.items():
                    # Check if query matches knowledge
                    score = 0.0
                    
                    # Check title
                    if "title" in knowledge and query_lower in knowledge["title"].lower():
                        score += 0.8
                    
                    # Check content
                    if "content" in knowledge and query_lower in knowledge["content"].lower():
                        score += 0.5
                    
                    # Check tags
                    if "tags" in knowledge:
                        for tag in knowledge["tags"]:
                            if query_lower in tag.lower():
                                score += 0.3
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append({
                            "id": knowledge_id,
                            "knowledge": knowledge,
                            "score": score
                        })
                
                # Sort by score
                results = sorted(results, key=lambda r: r["score"], reverse=True)
                
                # Limit results
                results = results[:max_results]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge from the knowledge base.
        
        Returns:
            List of all knowledge items
        """
        return [
            {
                "id": knowledge_id,
                "knowledge": knowledge
            }
            for knowledge_id, knowledge in self.knowledge_base.items()
        ]
    
    async def import_knowledge(self, file_path: str) -> int:
        """
        Import knowledge from a file.
        
        Args:
            file_path: Path to knowledge file
            
        Returns:
            Number of imported knowledge items
        """
        try:
            # Check file extension
            if file_path.endswith(".json"):
                # Load JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check if it's a list or a single item
                if isinstance(data, list):
                    # Import multiple knowledge items
                    count = 0
                    for item in data:
                        await self.add_knowledge(item)
                        count += 1
                    
                    return count
                else:
                    # Import single knowledge item
                    await self.add_knowledge(data)
                    return 1
            
            elif file_path.endswith(".txt"):
                # Load text file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create knowledge item
                knowledge = {
                    "title": os.path.basename(file_path),
                    "content": content,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                }
                
                # Add knowledge
                await self.add_knowledge(knowledge)
                
                return 1
            
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return 0
            
        except Exception as e:
            logger.error(f"Error importing knowledge from {file_path}: {e}")
            return 0
    
    async def export_knowledge(self, file_path: str) -> bool:
        """
        Export knowledge to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all knowledge
            knowledge_items = [
                knowledge for knowledge_id, knowledge in self.knowledge_base.items()
            ]
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(knowledge_items, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge to {file_path}: {e}")
            return False
"""
User modeling for the AI assistant.

This module provides functionality for modeling user preferences
and behavior for the AI assistant.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class UserModel:
    """
    User model for the AI assistant.
    
    This class represents a model of a user's preferences and behavior.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the user model.
        
        Args:
            user_id: User ID
        """
        self.user_id = user_id
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        
        # User preferences
        self.preferences = {}
        
        # User interaction history
        self.interactions = []
        
        # User topics of interest
        self.topics = {}
    
    def update_preference(self, key: str, value: Any):
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.last_updated = datetime.now().isoformat()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        return self.preferences.get(key, default)
    
    def add_interaction(self, interaction_type: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a user interaction.
        
        Args:
            interaction_type: Type of interaction
            content: Interaction content
            metadata: Interaction metadata
        """
        # Create interaction
        interaction = {
            "type": interaction_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to interactions
        self.interactions.append(interaction)
        
        # Update last updated
        self.last_updated = datetime.now().isoformat()
        
        # Update topics of interest
        if interaction_type == "query":
            self._update_topics(content)
    
    def get_interactions(self, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        # Filter by type if specified
        if interaction_type:
            filtered = [i for i in self.interactions if i["type"] == interaction_type]
        else:
            filtered = self.interactions
        
        # Limit count if specified
        if count is not None:
            filtered = filtered[-count:]
        
        return filtered
    
    def get_top_topics(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top topics of interest.
        
        Args:
            count: Number of topics to get
            
        Returns:
            List of topics
        """
        # Sort topics by score
        sorted_topics = sorted(
            self.topics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to list of dictionaries
        return [
            {"topic": topic, "score": score}
            for topic, score in sorted_topics[:count]
        ]
    
    def _update_topics(self, content: str):
        """
        Update topics of interest based on content.
        
        Args:
            content: Content to analyze
        """
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Update topic scores
        for keyword in keywords:
            if keyword in self.topics:
                self.topics[keyword] += 1
            else:
                self.topics[keyword] = 1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Split into words
        words = text.lower().split()
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "that", "this", "these", "those", "it", "they", 
                        "he", "she", "we", "you", "i", "me", "my", "your", "our", "their"}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:10]]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the user model to a dictionary.
        
        Returns:
            User model dictionary
        """
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "preferences": self.preferences,
            "interactions": self.interactions,
            "topics": self.topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserModel':
        """
        Create a user model from a dictionary.
        
        Args:
            data: User model dictionary
            
        Returns:
            User model
        """
        user_model = cls(data["user_id"])
        user_model.created_at = data["created_at"]
        user_model.last_updated = data["last_updated"]
        user_model.preferences = data["preferences"]
        user_model.interactions = data["interactions"]
        user_model.topics = data["topics"]
        
        return user_model

class UserModelManager:
    """
    User model manager for the AI assistant.
    
    This class manages user models for the AI assistant.
    """
    
    def __init__(self, user_models_dir: str = "./user_models"):
        """
        Initialize the user model manager.
        
        Args:
            user_models_dir: Path to user models directory
        """
        self.user_models_dir = Path(user_models_dir)
        
        # Create user models directory if it doesn't exist
        self.user_models_dir.mkdir(parents=True, exist_ok=True)
        
        # User models
        self.user_models = {}
    
    async def initialize(self):
        """Initialize the user model manager."""
        try:
            # Load user models
            await self.load_user_models()
            
        except Exception as e:
            logger.error(f"Error initializing user model manager: {e}")
    
    async def load_user_models(self) -> bool:
        """
        Load user models from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear user models
            self.user_models = {}
            
            # Load user model files
            for file_path in self.user_models_dir.glob("*.json"):
                try:
                    # Load user model file
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Create user model
                    user_model = UserModel.from_dict(data)
                    
                    # Add to user models
                    self.user_models[user_model.user_id] = user_model
                    
                except Exception as e:
                    logger.error(f"Error loading user model file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.user_models)} user models")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading user models: {e}")
            return False
    
    async def save_user_model(self, user_id: str) -> bool:
        """
        Save a user model to disk.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.user_models:
            return False
        
        try:
            # Get user model
            user_model = self.user_models[user_id]
            
            # Save user model file
            file_path = self.user_models_dir / f"{user_i
           # Save user model file
            file_path = self.user_models_dir / f"{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_model.to_dict(), f, indent=2)
            
            logger.info(f"Saved user model for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user model for user {user_id}: {e}")
            return False
    
    async def save_all_user_models(self) -> bool:
        """
        Save all user models to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save each user model
            for user_id in self.user_models:
                await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving all user models: {e}")
            return False
    
    async def get_user_model(self, user_id: str) -> UserModel:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model
        """
        # Check if user model exists
        if user_id not in self.user_models:
            # Create new user model
            self.user_models[user_id] = UserModel(user_id)
        
        return self.user_models[user_id]
    
    async def delete_user_model(self, user_id: str) -> bool:
        """
        Delete a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.user_models:
            return False
        
        try:
            # Remove from user models
            del self.user_models[user_id]
            
            # Remove user model file
            file_path = self.user_models_dir / f"{user_id}.json"
            if file_path.exists():
                os.remove(file_path)
            
            logger.info(f"Deleted user model for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user model for user {user_id}: {e}")
            return False
    
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List of user IDs
        """
        return list(self.user_models.keys())
    
    async def add_user_interaction(self, user_id: str, interaction_type: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a user interaction.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            content: Interaction content
            metadata: Interaction metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Add interaction
            user_model.add_interaction(interaction_type, content, metadata)
            
            # Save user model
            await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding interaction for user {user_id}: {e}")
            return False
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Update preference
            user_model.update_preference(key, value)
            
            # Save user model
            await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference for user {user_id}: {e}")
            return False
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get preference
            return user_model.get_preference(key, default)
            
        except Exception as e:
            logger.error(f"Error getting preference for user {user_id}: {e}")
            return default
    
    async def get_user_interactions(self, user_id: str, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get interactions
            return user_model.get_interactions(interaction_type, count)
            
        except Exception as e:
            logger.error(f"Error getting interactions for user {user_id}: {e}")
            return []
    
    async def get_user_top_topics(self, user_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top topics of interest for a user.
        
        Args:
            user_id: User ID
            count: Number of topics to get
            
        Returns:
            List of topics
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get top topics
            return user_model.get_top_topics(count)
            
        except Exception as e:
            logger.error(f"Error getting top topics for user {user_id}: {e}")
            return []
"""
Utility functions for the AI assistant.

This module provides utility functions for the AI assistant.
"""

import logging
import os
import platform
import subprocess
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

# Configure logging
logger = logging.getLogger(__name__)

def capture_screen() -> Optional[np.ndarray]:
    """
    Capture the screen.
    
    Returns:
        Screen image as numpy array, or None if failed
    """
    try:
        # Try to import mss
        import mss
        
        # Capture screen
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]
            
            # Capture screenshot
            sct_img = sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(sct_img)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
    except ImportError:
        logger.warning("mss not available, trying alternative method")
        
        # Try alternative method based on platform
        system = platform.system()
        
        if system == "Windows":
            # Use PIL on Windows
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab()
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Error capturing screen on Windows: {e}")
                return None
                
        elif system == "Darwin":  # macOS
            try:
                # Use screencapture on macOS
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Capture screen to temporary file
                subprocess.run(["screencapture", "-x", temp_path], check=True)
                
                # Read image
                img = cv2.imread(temp_path)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                return img
                
            except Exception as e:
                logger.error(f"Error capturing screen on macOS: {e}")
                return None
                
        elif system == "Linux":
            try:
                # Use scrot on Linux
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Capture screen to temporary file
                subprocess.run(["scrot", temp_path], check=True)
                
                # Read image
                img = cv2.imread(temp_path)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                return img
                
            except Exception as e:
                logger.error(f"Error capturing screen on Linux: {e}")
                return None
        
        else:
            logger.error(f"Unsupported platform: {system}")
            return None
    
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return None

def extract_text_from_image(image: np.ndarray) -> List[str]:
    """
    Extract text from an image.
    
    Args:
        image: Image as numpy array
        
    Returns:
        List of extracted text
    """
    try:
        # Check if pytesseract is available
        if not pytesseract.get_tesseract_version():
            logger.warning("Tesseract not available, text extraction disabled")
            return []
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Split into lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        return lines
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return []

def execute_command(command: str) -> dict:
    """
    Execute a shell command.
    
    Args:
        command: Command to execute
        
    Returns:
        Dictionary with command result
    """
    try:
        # Execute command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Get output
        stdout, stderr = process.communicate()
        
        # Get return code
        return_code = process.returncode
        
        return {
            "success": return_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code
        }
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    try:
        # Get system information
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        # Get memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory_total"] = memory.total
            info["memory_available"] = memory.available
            info["memory_percent"] = memory.percent
            
            # Get disk information
            disk = psutil.disk_usage("/")
            info["disk_total"] = disk.total
            info["disk_free"] = disk.free
            info["disk_percent"] = disk.percent
            
            # Get CPU information
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
        except ImportError:
            logger.warning("psutil not available, limited system information")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system information: {e}")
        
        return {
            "error": str(e)
        }

def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"
"""
AI assistant main module.

This module provides the main AI assistant functionality.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import ConfigManager
from .knowledge import KnowledgeManager
from .models import ModelManager
from .session import SessionManager
from .user_model import UserModelManager
from .vision import VisionProcessor
from .voice import VoiceProcessor

# Configure logging
logger = logging.getLogger(__name__)

class AIAssistant:
    """
    AI assistant.
    
    This class provides the main AI assistant functionality.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the AI assistant.
        
        Args:
            config_path: Path to configuration file
        """
        # Configuration
        self.config_manager = ConfigManager(config_path)
        self.config = {}
        
        # Components
        self.model_manager = None
        self.session_manager = None
        self.knowledge_manager = None
        self.user_model_manager = None
        self.vision_processor = None
        self.voice_processor = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize the AI assistant."""
        try:
            # Load configuration
            self.config = await self.config_manager.load_config()
            
            # Configure logging
            log_level = self.config.get("log_level", "INFO")
            logging.basicConfig(
                level=getattr(logging, log_level),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
            # Initialize components
            await self._initialize_components()
            
            self.initialized = True
            logger.info("AI assistant initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI assistant: {e}")
    
    async def _initialize_components(self):
        """Initialize AI assistant components."""
        # Initialize model manager
        model_path
       # Initialize model manager
        model_path = self.config.get("model_path", "./models")
        self.model_manager = ModelManager(model_path)
        await self.model_manager.initialize()
        
        # Initialize session manager
        session_timeout = self.config.get("session_timeout", 300)
        self.session_manager = SessionManager(session_timeout)
        await self.session_manager.initialize()
        
        # Initialize knowledge manager
        knowledge_dir = self.config.get("knowledge_dir", "./knowledge")
        self.knowledge_manager = KnowledgeManager(knowledge_dir)
        await self.knowledge_manager.initialize()
        
        # Initialize user model manager
        user_models_dir = self.config.get("user_models_dir", "./user_models")
        self.user_model_manager = UserModelManager(user_models_dir)
        await self.user_model_manager.initialize()
        
        # Initialize vision processor
        self.vision_processor = VisionProcessor(model_path)
        await self.vision_processor.initialize()
        
        # Initialize voice processor
        self.voice_processor = VoiceProcessor(model_path)
        await self.voice_processor.initialize()
    
    async def process_query(self, query: str, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Start time
            start_time = time.time()
            
            # Get or create session
            if session_id is None:
                session_id = self.session_manager.create_session()
            else:
                # Update session
                self.session_manager.update_session(session_id)
            
            # Get session
            session = self.session_manager.get_session(session_id)
            if session is None:
                return {
                    "success": False,
                    "error": "Invalid session ID"
                }
            
            # Add query to session history
            session.add_message("user", query)
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "query",
                    query,
                    {"session_id": session_id}
                )
            
            # Process query
            response = await self._generate_response(query, session, user_id)
            
            # Add response to session history
            session.add_message("assistant", response["text"])
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "response",
                    response["text"],
                    {"session_id": session_id}
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return response
            return {
                "success": True,
                "session_id": session_id,
                "text": response["text"],
                "processing_time": processing_time,
                "actions": response.get("actions", []),
                "context": response.get("context", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_response(self, query: str, session, user_id: str = None) -> Dict[str, Any]:
        """
        Generate a response to a user query.
        
        Args:
            query: User query
            session: Session object
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        try:
            # Get conversation history
            history = session.get_messages()
            
            # Get user preferences if user ID is provided
            user_preferences = {}
            if user_id:
                user_model = await self.user_model_manager.get_user_model(user_id)
                user_preferences = user_model.preferences
            
            # Search knowledge base
            knowledge_results = await self.knowledge_manager.search_knowledge(query)
            
            # Prepare context
            context = {
                "knowledge": knowledge_results,
                "user_preferences": user_preferences,
                "session_data": session.data
            }
            
            # Generate response
            response = await self.model_manager.generate_response(
                query,
                history,
                context
            )
            
            # Extract actions
            actions = self._extract_actions(response)
            
            # Execute actions
            action_results = await self._execute_actions(actions, session, user_id)
            
            # Update response if needed
            if action_results:
                # Generate updated response
                updated_response = await self.model_manager.generate_response(
                    query,
                    history,
                    {
                        **context,
                        "action_results": action_results
                    }
                )
                
                response = updated_response
            
            return {
                "text": response,
                "actions": actions,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            return {
                "text": "I'm sorry, I encountered an error while processing your request."
            }
    
    def _extract_actions(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract actions from response.
        
        Args:
            response: Response text
            
        Returns:
            List of actions
        """
        # This is a simplified implementation
        # In a real implementation, we would use a more sophisticated approach
        # to extract actions from the response
        
        actions = []
        
        # Check for action markers
        if "[ACTION:" in response:
            # Extract action blocks
            action_blocks = response.split("[ACTION:")[1:]
            
            for block in action_blocks:
                # Extract action type and parameters
                action_end = block.find("]")
                if action_end != -1:
                    action_type = block[:action_end].strip()
                    
                    # Create action
                    action = {
                        "type": action_type,
                        "parameters": {}
                    }
                    
                    # Extract parameters
                    param_start = block.find("(")
                    param_end = block.find(")")
                    
                    if param_start != -1 and param_end != -1:
                        param_str = block[param_start+1:param_end]
                        
                        # Parse parameters
                        params = param_str.split(",")
                        for param in params:
                            if "=" in param:
                                key, value = param.split("=", 1)
                                action["parameters"][key.strip()] = value.strip()
                    
                    # Add action
                    actions.append(action)
        
        return actions
    
    async def _execute_actions(self, actions: List[Dict[str, Any]], session, user_id: str = None) -> Dict[str, Any]:
        """
        Execute actions.
        
        Args:
            actions: List of actions
            session: Session object
            user_id: User ID (optional)
            
        Returns:
            Action results
        """
        results = {}
        
        for action in actions:
            action_type = action["type"]
            parameters = action["parameters"]
            
            try:
                if action_type == "search_knowledge":
                    # Search knowledge base
                    query = parameters.get("query", "")
                    results["search_knowledge"] = await self.knowledge_manager.search_knowledge(query)
                
                elif action_type == "capture_screen":
                    # Capture and analyze screen
                    results["capture_screen"] = await self.vision_processor.capture_and_analyze_screen()
                
                elif action_type == "record_audio":
                    # Record audio
                    duration = float(parameters.get("duration", 5.0))
                    audio_data, sample_rate = await self.voice_processor.record_audio(duration)
                    
                    # Recognize speech
                    if audio_data is not None:
                        speech_result = await self.voice_processor.recognize_speech(audio_data, sample_rate)
                        results["record_audio"] = speech_result
                
                elif action_type == "synthesize_speech":
                    # Synthesize speech
                    text = parameters.get("text", "")
                    language = parameters.get("language", "en")
                    
                    audio_data, sample_rate = await self.voice_processor.synthesize_speech(text, language)
                    
                    # Play audio
                    if audio_data is not None:
                        await self.voice_processor.play_audio(audio_data, sample_rate)
                        
                        results["synthesize_speech"] = {
                            "success": True,
                            "text": text
                        }
                
                elif action_type == "update_user_preference":
                    # Update user preference
                    if user_id:
                        key = parameters.get("key", "")
                        value = parameters.get("value", "")
                        
                        if key:
                            success = await self.user_model_manager.update_user_preference(user_id, key, value)
                            results["update_user_preference"] = {
                                "success": success,
                                "key": key,
                                "value": value
                            }
                
                elif action_type == "store_session_data":
                    # Store session data
                    key = parameters.get("key", "")
                    value = parameters.get("value", "")
                    
                    if key:
                        session.data[key] = value
                        results["store_session_data"] = {
                            "success": True,
                            "key": key,
                            "value": value
                        }
                
                elif action_type == "execute_command":
                    # Execute command
                    command = parameters.get("command", "")
                    
                    if command:
                        from .utils import execute_command
                        result = execute_command(command)
                        results["execute_command"] = result
                
            except Exception as e:
                logger.error(f"Error executing action {action_type}: {e}")
                results[action_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def process_voice_query(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a voice query.
        
        Args:
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Record audio
            audio_data, sample_rate = await self.voice_processor.record_audio()
            
            if audio_data is None:
                return {
                    "success": False,
                    "error": "Failed to record audio"
                }
            
            # Recognize speech
            speech_result = await self.voice_processor.recognize_speech(audio_data, sample_rate)
            
            if not speech_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to recognize speech"
                }
            
            # Get recognized text
            query = speech_result["text"]
            
            # Process query
            response = await self.process_query(query, session_id, user_id)
            
            # Synthesize speech
            if response["success"]:
                audio_data, sample_rate = await self.voice_processor.synthesize_speech(response["text"])
                
                if audio_data is not None:
                    # Play audio
                    await self.voice_processor.play_audio(audio_data, sample_rate)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_visual_query(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a visual query.
        
        Args:
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Capture and analyze screen
            screen_result = await self.vision_processor.capture_and_analyze_screen()
            
            if not screen_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to capture and analyze screen"
                }
            
            # Get or create session
            if session_id is None:
                session_id = self.session_manager.create_session()
            else:
                # Update session
                self.session_manager.update_session(session_id)
            
            # Get session
            session = self.session_manager.get_session(session_id)
            if session is None:
                return {
                    "success": False,
                    "error": "Invalid session ID"
                }
            
            # Store screen analysis in session data
            session.data["screen_analysis"] = screen_result
            
            # Generate response
            response = await self._generate_response(
                "What do you see on my screen?",
                session,
                user_id
            )
            
            # Add response to session history
            session.add_message("assistant", response["text"])
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "visual_query",
                    "What do you see on my screen?",
                    {"session_id": session_id}
                )
                
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "response",
                    response["text"],
                    {"session_id": session_id}
                )
            
            return {
                "success": True,
                "session_id": session_id,
                "text": response["text"],
                "screen_analysis": screen_result,
                "actions": response.get("actions", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing visual query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return self.session_manager.end_session(session_id)
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information, or None if session doesn't exist
        """
       """
        if not self.initialized:
            return None
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        
        return {
            "session_id": session_id,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "message_count": len(session.messages),
            "data": session.data
        }
    
    async def get_session_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get session messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of messages, or None if session doesn't exist
        """
        if not self.initialized:
            return None
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        
        return session.get_messages()
    
    async def get_active_sessions(self) -> List[str]:
        """
        Get active session IDs.
        
        Returns:
            List of active session IDs
        """
        if not self.initialized:
            return []
        
        return self.session_manager.get_active_sessions()
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            
        Returns:
            Knowledge ID
        """
        if not self.initialized:
            return ""
        
        return await self.knowledge_manager.add_knowledge(knowledge)
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        if not self.initialized:
            return None
        
        return await self.knowledge_manager.get_knowledge(knowledge_id)
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.knowledge_manager.update_knowledge(knowledge_id, knowledge)
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.knowledge_manager.delete_knowledge(knowledge_id)
    
    async def search_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        if not self.initialized:
            return []
        
        return await self.knowledge_manager.search_knowledge(query, max_results=max_results)
    
    async def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model, or None if not found
        """
        if not self.initialized:
            return None
        
        user_model = await self.user_model_manager.get_user_model(user_id)
        return user_model.to_dict()
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.user_model_manager.update_user_preference(user_id, key, value)
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        if not self.initialized:
            return default
        
        return await self.user_model_manager.get_user_preference(user_id, key, default)
    
    async def get_user_interactions(self, user_id: str, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        if not self.initialized:
            return []
        
        return await self.user_model_manager.get_user_interactions(user_id, interaction_type, count)
    
    async def delete_user_model(self, user_id: str) -> bool:
        """
        Delete a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.user_model_manager.delete_user_model(user_id)
    
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List of user IDs
        """
        if not self.initialized:
            return []
        
        return await self.user_model_manager.get_all_user_ids()
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Screen analysis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.vision_processor.capture_and_analyze_screen()
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image analysis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.vision_processor.analyze_image(image_path)
    
    async def record_audio(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Record audio.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Audio recording result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        audio_data, sample_rate = await self.voice_processor.record_audio(duration)
        
        if audio_data is None:
            return {
                "success": False,
                "error": "Failed to record audio"
            }
        
        return {
            "success": True,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    
    async def recognize_speech(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Speech recognition result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.voice_processor.recognize_speech(audio_data, sample_rate)
    
    async def synthesize_speech(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Synthesize speech.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Speech synthesis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        audio_data, sample_rate = await self.voice_processor.synthesize_speech(text, language)
        
        if audio_data is None:
            return {
                "success": False,
                "error": "Failed to synthesize speech"
            }
        
        return {
            "success": True,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    
    async def play_audio(self, audio_data: bytes, sample_rate: int) -> bool:
        """
        Play audio.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.voice_processor.play_audio(audio_data, sample_rate)
    
    async def shutdown(self):
        """Shut down the AI assistant."""
        try:
            # Save user models
            if self.user_model_manager:
                await self.user_model_manager.save_all_user_models()
            
            # Save knowledge base
            if self.knowledge_manager:
                await self.knowledge_manager.save_knowledge_base()
            
            # Save configuration
            if self.config_manager:
                await self.config_manager.save_config(self.config)
            
            logger.info("AI assistant shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down AI assistant: {e}")
"""
AI assistant main entry point.

This module provides the main entry point for the AI assistant.
"""

import argparse
import asyncio
import logging
import sys

from .assistant import AIAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI assistant")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create AI assistant
    assistant = AIAssistant(args.config)
    
    try:
        # Initialize AI assistant
        await assistant.initialize()
        
        # Start interactive mode
        await interactive_mode(assistant)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error: {e}")
    
    finally:
        # Shut down AI assistant
        await assistant.shutdown()

async def interactive_mode(assistant: AIAssistant):
    """
    Interactive mode.
    
    Args:
        assistant: AI assistant
    """
    print("AI assistant interactive mode")
    print("Type 'exit' to exit")
    print("Type 'help' for help")
    print()
    
    session_id = None
    user_id = "interactive_user"
    
    while True:
        try:
            # Get user input
            user_input = input("> ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Check for help command
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Check for special commands
            if user_input.startswith("/"):
                await handle_command(assistant, user_input, session_id, user_id)
                continue
            
            # Process query
            response = await assistant.process_query(user_input, session_id, user_id)
            
            # Update session ID
            if response["success"]:
                session_id = response["session_id"]
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            logger.error(f"Error: {e}")

async def handle_command(assistant: AIAssistant, command: str, session_id: str, user_id: str):
    """
    Handle a special command.
    
    Args:
        assistant: AI assistant
        command: Command string
        session_id: Session ID
        user_id: User ID
    """
    # Parse command
    parts = command[1:].split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]
    
    try:
        if cmd == "voice":
            # Process voice query
            print("Listening...")
            response = await assistant.process_voice_query(session_id, user_id)
            
            if response["success"]:
                session_id = response["session_id"]
                print(f"You said: {response.get('query', '')}")
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        elif cmd == "visual":
            # Process visual query
            print("Capturing screen...")
            response = await assistant.process_visual_query(session_id, user_id)
            
            if response["success"]:
                session_id = response["session_id"]
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        elif cmd == "session":
            # Session commands
            if not args:
                # Get session info
                if session_id:
                    info = await assistant.get_session_info(session_id)
                    print(f"Session ID: {session_id}")
                    print(f"Created at: {info['created_at']}")
                    print(f"Last updated: {info['last_updated']}")
                    print(f"Message count: {info['message_count']}")
                else:
                    print("No active session")
            
            elif args[0] == "new":
                # Create new session
                session_id = assistant.session_manager.create_session()
                print(f"Created new session: {session_id}")
           elif args[0] == "list":
                # List active sessions
                sessions = await assistant.get_active_sessions()
                print(f"Active sessions ({len(sessions)}):")
                for s in sessions:
                    info = await assistant.get_session_info(s)
                    print(f"  {s} - {info['message_count']} messages, last updated: {info['last_updated']}")
            
            elif args[0] == "end":
                # End session
                if session_id:
                    success = await assistant.end_session(session_id)
                    if success:
                        print(f"Ended session: {session_id}")
                        session_id = None
                    else:
                        print(f"Failed to end session: {session_id}")
                else:
                    print("No active session")
            
            elif args[0] == "messages":
                # Get session messages
                if session_id:
                    messages = await assistant.get_session_messages(session_id)
                    print(f"Session messages ({len(messages)}):")
                    for msg in messages:
                        print(f"  {msg['role']}: {msg['content']}")
                else:
                    print("No active session")
            
            else:
                print(f"Unknown session command: {args[0]}")
        
        elif cmd == "knowledge":
            # Knowledge commands
            if not args:
                print("Usage: /knowledge [add|get|update|delete|search|list]")
            
            elif args[0] == "add":
                # Add knowledge
                if len(args) < 3:
                    print("Usage: /knowledge add <title> <content>")
                else:
                    title = args[1]
                    content = " ".join(args[2:])
                    
                    knowledge = {
                        "title": title,
                        "content": content,
                        "source": "interactive",
                        "tags": []
                    }
                    
                    knowledge_id = await assistant.add_knowledge(knowledge)
                    print(f"Added knowledge: {knowledge_id}")
            
            elif args[0] == "get":
                # Get knowledge
                if len(args) < 2:
                    print("Usage: /knowledge get <id>")
                else:
                    knowledge_id = args[1]
                    knowledge = await assistant.get_knowledge(knowledge_id)
                    
                    if knowledge:
                        print(f"Knowledge ID: {knowledge_id}")
                        print(f"Title: {knowledge['title']}")
                        print(f"Content: {knowledge['content']}")
                        print(f"Source: {knowledge.get('source', 'unknown')}")
                        print(f"Tags: {', '.join(knowledge.get('tags', []))}")
                    else:
                        print(f"Knowledge not found: {knowledge_id}")
            
            elif args[0] == "update":
                # Update knowledge
                if len(args) < 4:
                    print("Usage: /knowledge update <id> <title> <content>")
                else:
                    knowledge_id = args[1]
                    title = args[2]
                    content = " ".join(args[3:])
                    
                    # Get existing knowledge
                    knowledge = await assistant.get_knowledge(knowledge_id)
                    
                    if knowledge:
                        # Update knowledge
                        knowledge["title"] = title
                        knowledge["content"] = content
                        
                        success = await assistant.update_knowledge(knowledge_id, knowledge)
                        
                        if success:
                            print(f"Updated knowledge: {knowledge_id}")
                        else:
                            print(f"Failed to update knowledge: {knowledge_id}")
                    else:
                        print(f"Knowledge not found: {knowledge_id}")
            
            elif args[0] == "delete":
                # Delete knowledge
                if len(args) < 2:
                    print("Usage: /knowledge delete <id>")
                else:
                    knowledge_id = args[1]
                    success = await assistant.delete_knowledge(knowledge_id)
                    
                    if success:
                        print(f"Deleted knowledge: {knowledge_id}")
                    else:
                        print(f"Failed to delete knowledge: {knowledge_id}")
            
            elif args[0] == "search":
                # Search knowledge
                if len(args) < 2:
                    print("Usage: /knowledge search <query>")
                else:
                    query = " ".join(args[1:])
                    results = await assistant.search_knowledge(query)
                    
                    print(f"Search results ({len(results)}):")
                    for result in results:
                        print(f"  ID: {result['id']}")
                        print(f"  Title: {result['knowledge']['title']}")
                        print(f"  Score: {result['score']}")
                        print()
            
            elif args[0] == "list":
                # List all knowledge
                knowledge_items = await assistant.knowledge_manager.get_all_knowledge()
                
                print(f"Knowledge items ({len(knowledge_items)}):")
                for item in knowledge_items:
                    print(f"  ID: {item['id']}")
                    print(f"  Title: {item['knowledge']['title']}")
                    print()
            
            else:
                print(f"Unknown knowledge command: {args[0]}")
        
        elif cmd == "user":
            # User commands
            if not args:
                # Get user info
                user_model = await assistant.get_user_model(user_id)
                
                if user_model:
                    print(f"User ID: {user_model['user_id']}")
                    print(f"Created at: {user_model['created_at']}")
                    print(f"Last updated: {user_model['last_updated']}")
                    print(f"Preferences: {user_model['preferences']}")
                    print(f"Interactions: {len(user_model['interactions'])}")
                    
                    # Get top topics
                    topics = sorted(user_model['topics'].items(), key=lambda x: x[1], reverse=True)[:5]
                    if topics:
                        print("Top topics:")
                        for topic, score in topics:
                            print(f"  {topic}: {score}")
                else:
                    print(f"User not found: {user_id}")
            
            elif args[0] == "preference":
                # User preference commands
                if len(args) < 2:
                    print("Usage: /user preference [get|set] <key> [value]")
                
                elif args[1] == "get":
                    # Get preference
                    if len(args) < 3:
                        print("Usage: /user preference get <key>")
                    else:
                        key = args[2]
                        value = await assistant.get_user_preference(user_id, key)
                        print(f"Preference {key}: {value}")
                
                elif args[1] == "set":
                    # Set preference
                    if len(args) < 4:
                        print("Usage: /user preference set <key> <value>")
                    else:
                        key = args[2]
                        value = " ".join(args[3:])
                        
                        success = await assistant.update_user_preference(user_id, key, value)
                        
                        if success:
                            print(f"Set preference {key} = {value}")
                        else:
                            print(f"Failed to set preference {key}")
                
                else:
                    print(f"Unknown preference command: {args[1]}")
            
            elif args[0] == "interactions":
                # Get user interactions
                count = int(args[1]) if len(args) > 1 else 5
                interactions = await assistant.get_user_interactions(user_id, count=count)
                
                print(f"User interactions ({len(interactions)}):")
                for interaction in interactions:
                    print(f"  {interaction['type']} at {interaction['timestamp']}")
                    print(f"    {interaction['content']}")
                    print()
            
            else:
                print(f"Unknown user command: {args[0]}")
        
        elif cmd == "config":
            # Configuration commands
            if not args:
                # Show configuration
                print("Configuration:")
                for key, value in assistant.config.items():
                    print(f"  {key}: {value}")
            
            elif len(args) >= 2:
                # Set configuration
                key = args[0]
                value = " ".join(args[1:])
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        value = float(value)
                except:
                    pass
                
                # Update configuration
                assistant.config[key] = value
                
                # Save configuration
                await assistant.config_manager.save_config(assistant.config)
                
                print(f"Set {key} = {value}")
            
            else:
                print("Usage: /config [key value]")
        
        elif cmd == "help":
            # Show help
            print_help()
        
        else:
            print(f"Unknown command: {cmd}")
    
    except Exception as e:
        logger.error(f"Error handling command: {e}")
        print(f"Error: {e}")

def print_help():
    """Print help information."""
    print("AI assistant commands:")
    print("  /voice - Process voice query")
    print("  /visual - Process visual query")
    print("  /session - Session commands")
    print("    /session - Show current session info")
    print("    /session new - Create new session")
    print("    /session list - List active sessions")
    print("    /session end - End current session")
    print("    /session messages - Show session messages")
    print("  /knowledge - Knowledge commands")
    print("    /knowledge add <title> <content> - Add knowledge")
    print("    /knowledge get <id> - Get knowledge")
    print("    /knowledge update <id> <title> <content> - Update knowledge")
    print("    /knowledge delete <id> - Delete knowledge")
    print("    /knowledge search <query> - Search knowledge")
    print("    /knowledge list - List all knowledge")
    print("  /user - User commands")
    print("    /user - Show user info")
    print("    /user preference get <key> - Get user preference")
    print("    /user preference set <key> <value> - Set user preference")
    print("    /user interactions [count] - Show user interactions")
    print("  /config - Configuration commands")
    print("    /config - Show configuration")
    print("    /config <key> <value> - Set configuration")
    print("  /help - Show help")
    print("  exit - Exit")

if __name__ == "__main__":
    asyncio.run(main())
"""
Vision processing for the AI assistant.

This module provides vision processing functionality for the AI assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import capture_screen, extract_text_from_image

# Configure logging
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Vision processor for the AI assistant.
    
    This class provides vision processing functionality for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the vision processor.
        
        Args:
            model_path: Path to vision models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.object_detector = None
        self.face_detector = None
        self.text_detector = None
    
    async def initialize(self):
        """Initialize the vision processor."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            await self._initialize_models()
            
            logger.info("Vision processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vision processor: {e}")
    
    async def _initialize_models(self):
        """Initialize vision models."""
        try:
            # Initialize object detector
            self._initialize_object_detector()
            
            # Initialize face detector
            self._initialize_face_detector()
            
            # Initialize text detector
            self._initialize_text_detector()
            
        except Exception as e:
            logger.error(f"Error initializing vision models: {e}")
    
    def _initialize_object_detector(self):
        """Initialize object detector."""
        try:
            # Check if OpenCV DNN module is available
            if hasattr(cv2, "dnn"):
                # Load YOLO model
                yolo_cfg = self.model_path / "yolov3.cfg"
                yolo_weights = self.model_path / "yolov3.weights"
                
                if yolo_cfg.exists() and yolo_weights.exists():
                    self.object_detector = cv2.dnn.readNetFromDarknet(
                        str(yolo_cfg),
                        str(yolo_weights)
                    )
                    
                    # Load COCO class names
                    coco_names = self.model_path / "coco.names"
                    if coco_names.exists():
                        with open(coco_names, "r") as f:
                            self.object_classes = f.read().strip().split("\n")
                    else:
                        # Default COCO class names
                        self.object_classes = [
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                            "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                        ]
                    
                    logger.info("Object detector initialized")
                else:
                    logger.warning("YOLO model files not found, object detection disabled")
            else:
                logger.warning("OpenCV DNN module not available, object detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
    
    def _initialize_face_detector(self):
        """Initialize face detector."""
        try:
            # Load Haar cascade for face detection
           # Load Haar cascade for face detection
            face_cascade_path = self.model_path / "haarcascade_frontalface_default.xml"
            
            if face_cascade_path.exists():
                self.face_detector = cv2.CascadeClassifier(str(face_cascade_path))
                logger.info("Face detector initialized")
            else:
                # Try to use OpenCV's built-in cascades
                builtin_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(builtin_cascade):
                    self.face_detector = cv2.CascadeClassifier(builtin_cascade)
                    logger.info("Face detector initialized using built-in cascade")
                else:
                    logger.warning("Face cascade file not found, face detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
    
    def _initialize_text_detector(self):
        """Initialize text detector."""
        try:
            # Check if OpenCV text detection is available
            if hasattr(cv2, "text") and hasattr(cv2.text, "TextDetectorCNN_create"):
                # Load EAST text detector
                east_model_path = self.model_path / "frozen_east_text_detection.pb"
                
                if east_model_path.exists():
                    self.text_detector = cv2.dnn.readNet(str(east_model_path))
                    logger.info("Text detector initialized")
                else:
                    logger.warning("EAST model file not found, advanced text detection disabled")
            else:
                logger.warning("OpenCV text detection not available, advanced text detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing text detector: {e}")
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Screen analysis result
        """
        try:
            # Capture screen
            screen = capture_screen()
            
            if screen is None:
                return {
                    "success": False,
                    "error": "Failed to capture screen"
                }
            
            # Analyze screen
            analysis = await self.analyze_image_data(screen)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error capturing and analyzing screen: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image analysis result
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to read image"
                }
            
            # Analyze image
            analysis = await self.analyze_image_data(image)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_image_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image data.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            Image analysis result
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Extract text
            text = extract_text_from_image(image)
            
            # Detect objects
            objects = self._detect_objects(image) if self.object_detector else []
            
            # Detect faces
            faces = self._detect_faces(image) if self.face_detector else []
            
            # Detect colors
            colors = self._detect_dominant_colors(image)
            
            # Basic image properties
            properties = {
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height > 0 else 0,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "is_color": len(image.shape) > 2 and image.shape[2] > 1
            }
            
            return {
                "properties": properties,
                "text": text,
                "objects": objects,
                "faces": faces,
                "colors": colors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image data: {e}")
            
            return {
                "properties": {
                    "width": 0,
                    "height": 0,
                    "aspect_ratio": 0,
                    "channels": 0,
                    "is_color": False
                },
                "text": [],
                "objects": [],
                "faces": [],
                "colors": []
            }
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            List of detected objects
        """
        try:
            if self.object_detector is None:
                return []
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image,
                1/255.0,
                (416, 416),
                swapRB=True,
                crop=False
            )
            
            # Set input
            self.object_detector.setInput(blob)
            
            # Get output layer names
            layer_names = self.object_detector.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.object_detector.getUnconnectedOutLayers()]
            
            # Forward pass
            outputs = self.object_detector.forward(output_layers)
            
            # Process outputs
            class_ids = []
            confidences = []
            boxes = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Prepare results
            objects = []
            
            for i in indices:
                if isinstance(i, tuple):
                    i = i[0]  # For OpenCV 3
                
                box = boxes[i]
                x, y, w, h = box
                
                # Get class name
                class_id = class_ids[i]
                class_name = self.object_classes[class_id] if class_id < len(self.object_classes) else f"unknown_{class_id}"
                
                # Add object
                objects.append({
                    "class": class_name,
                    "confidence": confidences[i],
                    "box": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "center": {
                        "x": x + w // 2,
                        "y": y + h // 2
                    },
                    "area": w * h,
                    "relative_area": (w * h) / (width * height)
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            if self.face_detector is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rect = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Prepare results
            faces = []
            
            for (x, y, w, h) in faces_rect:
                # Add face
                faces.append({
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "center": {
                        "x": int(x + w // 2),
                        "y": int(y + h // 2)
                    },
                    "area": int(w * h),
                    "relative_area": float((w * h) / (width * height))
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _detect_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Dict[str, Any]]:
        """
        Detect dominant colors in an image.
        
        Args:
            image: Image data as numpy array
            num_colors: Number of dominant colors to detect
            
        Returns:
            List of dominant colors
        """
        try:
            # Reshape image
            pixels = image.reshape(-1, 3)
            
            # Convert to float32
            pixels = np.float32(pixels)
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Apply k-means clustering
            _, labels, centers = cv2.kmeans(
                pixels,
                num_colors,
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert centers to uint8
            centers = np.uint8(centers)
            
            # Count labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            
            # Prepare results
            colors = []
            
            for i in sorted_indices:
                # Get color
                color = centers[i].tolist()
                
                # Calculate percentage
                percentage = counts[i] / len(labels)
                
                # Add color
                colors.append({
                    "color": {
                        "b": color[0],
                        "g": color[1],
                        "r": color[2],
                        "hex": f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                    },
                    "percentage": float(percentage)
                })
            
            return colors
            
        except Exception as e:
            logger.error(f"Error detecting dominant colors: {e}")
            return []
"""
Voice processing for the AI assistant.

This module provides voice processing functionality for the AI assistant.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for the AI assistant.
    
    This class provides voice processing functionality for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the voice processor.
        
        Args:
            model_path: Path to voice models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.speech_recognizer = None
        self.speech_synthesizer = None
        
        # Audio device
        self.audio_device = None
    
    async def initialize(self):
        """Initialize the voice processor."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            await self._initialize_models()
            
            # Initialize audio device
            await self._initialize_audio_device()
            
            logger.info("Voice processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing voice processor: {e}")
    
    async def _initialize_models(self):
        """Initialize voice models."""
        try:
            # Initialize speech recognizer
            await self._initialize_speech_recognizer()
            
            # Initialize speech synthesizer
            await self._initialize_speech_synthesizer()
            
        except Exception as e:
            logger.error(f"Error initializing voice models: {e}")
    
    async def _initialize_speech_recognizer(self):
        """Initialize speech recognizer."""
        try:
            # Try to import speech recognition library
            import speech_recognition as sr
            
            # Create recognizer
            self.speech_recognizer = sr.Recognizer()
            
            logger.info("Speech recognizer initialized")
            
        except ImportError:
            logger.warning("speech_recognition not available, speech recognition disabled")
            
        except Exception as e:
            logger.error(f"Error initializing speech recognizer: {e}")
    
    async def _initialize_speech_synthesizer(self):
        """Initialize speech synthesizer."""
        try:
            # Try to import pyttsx3
            import pyttsx3
            
            # Create synthesizer
            self.speech_synthesizer = pyttsx3.init()
            
            logger.info("Speech synthesizer initialized")
            
        except ImportError:
            logger.warning("pyttsx3 not available, speech synthesis disabled")
            
            # Try to import gTTS as fallback
            try:
                from gtts import gTTS
                
                # Use gTTS
                self.speech_synthesizer = "gtts"
                
                logger.info("Speech synthesizer initialized using gTTS")
                
            except ImportError:
                logger.warning("gtts not available, speech synthesis disabled")
            
        except Exception as e:
            logger.error(f"Error initializing speech synthesizer: {e}")
    
    async def _initialize_audio_device(self):
        """Initialize audio device."""
        try:
            #

"""
Session management for the AI assistant.

This module provides session management functionality for the AI assistant.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class Session:
    """
    Session class for the AI assistant.
    
    This class represents a session with the AI assistant.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize a session.
        
        Args:
            session_id: Session ID
        """
        self.session_id = session_id
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.messages = []
        self.data = {}
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the session.
        
        Args:
            role: Message role (user or assistant)
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        self.last_updated = time.time()
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get session messages.
        
        Returns:
            List of messages
        """
        return self.messages
    
    def clear_messages(self):
        """Clear session messages."""
        self.messages = []
        self.last_updated = time.time()
    
    def is_expired(self, timeout: float) -> bool:
        """
        Check if the session is expired.
        
        Args:
            timeout: Session timeout in seconds
            
        Returns:
            True if the session is expired, False otherwise
        """
        return time.time() - self.last_updated > timeout

class SessionManager:
    """
    Session manager for the AI assistant.
    
    This class manages sessions for the AI assistant.
    """
    
    def __init__(self, session_timeout: float = 300.0):
        """
        Initialize the session manager.
        
        Args:
            session_timeout: Session timeout in seconds
        """
        self.sessions = {}
        self.session_timeout = session_timeout
    
    async def initialize(self):
        """Initialize the session manager."""
        try:
            logger.info("Session manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing session manager: {e}")
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        self.sessions[session_id] = Session(session_id)
        
        logger.info(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session, or None if not found
        """
        # Check if session exists
        if session_id not in self.sessions:
            return None
        
        # Get session
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.session_timeout):
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
            
            return None
        
        return session
    
    def update_session(self, session_id: str) -> bool:
        """
        Update a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.sessions:
            return False
        
        # Get session
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.session_timeout):
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
            
            return False
        
        # Update session
        session.last_updated = time.time()
        
        return True
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.sessions:
            return False
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Ended session: {session_id}")
        
        return True
    
    def get_active_sessions(self) -> List[str]:
        """
        Get active session IDs.
        
        Returns:
            List of active session IDs
        """
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        # Return active session IDs
        return list(self.sessions.keys())
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        # Get current time
        current_time = time.time()
        
        # Find expired sessions
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if current_time - session.last_updated > self.session_timeout
        ]
        
        # Remove expired sessions
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
"""
Model management for the AI assistant.

This module provides model management functionality for the AI assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for the AI assistant.
    
    This class manages AI models for the assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.models = {}
        
        # Default model
        self.default_model = None
    
    async def initialize(self):
        """Initialize the model manager."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Load models
            await self._load_models()
            
            logger.info("Model manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model manager: {e}")
    
    async def _load_models(self):
        """Load AI models."""
        try:
            # Try to import transformers
            try:
                from transformers import pipeline
                
                # Load text generation model
                try:
                    logger.info("Loading text generation model...")
                    
                    # Check if model exists locally
                    local_model_path = self.model_path / "text-generation"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["text-generation"] = pipeline(
                            "text-generation",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["text-generation"] = pipeline(
                            "text-generation",
                            model="gpt2"
                        )
                    
                    logger.info("Text generation model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading text generation model: {e}")
                
                # Load text classification model
                try:
                    logger.info("Loading text classification model...")
                    
                    # Check if model exists locally
                    local_model_path = self.model_path / "text-classification"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["text-classification"] = pipeline(
                            "text-classification",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["text-classification"] = pipeline(
                            "text-classification"
                        )
                    
                    logger.info("Text classification model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading text classification model: {e}")
                
                # Load question answering model
                try:
                    logger.info("Loading question answering model...")
                    
                    # Check if model
                   # Check if model exists locally
                    local_model_path = self.model_path / "question-answering"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["question-answering"] = pipeline(
                            "question-answering",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["question-answering"] = pipeline(
                            "question-answering"
                        )
                    
                    logger.info("Question answering model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading question answering model: {e}")
                
                # Set default model
                self.default_model = "text-generation"
                
            except ImportError:
                logger.warning("transformers not available, using fallback models")
                
                # Load fallback models
                self._load_fallback_models()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
            # Load fallback models
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback AI models."""
        try:
            # Simple rule-based model
            self.models["rule-based"] = {
                "type": "rule-based",
                "rules": [
                    {
                        "pattern": r"hello|hi|hey",
                        "response": "Hello! How can I help you today?"
                    },
                    {
                        "pattern": r"how are you",
                        "response": "I'm functioning well, thank you for asking. How can I assist you?"
                    },
                    {
                        "pattern": r"bye|goodbye",
                        "response": "Goodbye! Have a great day."
                    },
                    {
                        "pattern": r"thank you|thanks",
                        "response": "You're welcome! Is there anything else I can help with?"
                    },
                    {
                        "pattern": r"help",
                        "response": "I'm here to help. What do you need assistance with?"
                    }
                ]
            }
            
            # Set default model
            self.default_model = "rule-based"
            
            logger.info("Fallback models loaded")
            
        except Exception as e:
            logger.error(f"Error loading fallback models: {e}")
    
    async def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        try:
            # Check if text generation model is available
            if "text-generation" in self.models:
                # Generate text
                result = self.models["text-generation"](
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1
                )
                
                # Extract generated text
                generated_text = result[0]["generated_text"]
                
                return generated_text
            
            # Check if rule-based model is available
            elif "rule-based" in self.models:
                # Use rule-based model
                import re
                
                # Check rules
                for rule in self.models["rule-based"]["rules"]:
                    if re.search(rule["pattern"], prompt.lower()):
                        return rule["response"]
                
                # Default response
                return "I'm not sure how to respond to that."
            
            else:
                return "Text generation model not available."
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"
    
    async def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result
        """
        try:
            # Check if text classification model is available
            if "text-classification" in self.models:
                # Classify text
                result = self.models["text-classification"](text)
                
                # Extract classification
                if isinstance(result, list):
                    result = result[0]
                
                return {
                    "label": result["label"],
                    "score": result["score"]
                }
            
            else:
                return {
                    "label": "unknown",
                    "score": 0.0
                }
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            
            return {
                "label": "error",
                "score": 0.0,
                "error": str(e)
            }
    
    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question based on context.
        
        Args:
            question: Question to answer
            context: Context for the question
            
        Returns:
            Answer result
        """
        try:
            # Check if question answering model is available
            if "question-answering" in self.models:
                # Answer question
                result = self.models["question-answering"](
                    question=question,
                    context=context
                )
                
                return {
                    "answer": result["answer"],
                    "score": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                }
            
            else:
                return {
                    "answer": "I don't know the answer to that question.",
                    "score": 0.0,
                    "start": 0,
                    "end": 0
                }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            
            return {
                "answer": f"Error answering question: {e}",
                "score": 0.0,
                "start": 0,
                "end": 0
            }
    
    async def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_name: Model name, or None for default model
            
        Returns:
            Model information
        """
        try:
            # Get model name
            if model_name is None:
                model_name = self.default_model
            
            # Check if model exists
            if model_name not in self.models:
                return {
                    "name": model_name,
                    "available": False,
                    "error": "Model not found"
                }
            
            # Get model
            model = self.models[model_name]
            
            # Get model info
            if model_name == "rule-based":
                return {
                    "name": model_name,
                    "available": True,
                    "type": "rule-based",
                    "rules": len(model["rules"])
                }
            else:
                return {
                    "name": model_name,
                    "available": True,
                    "type": "transformer"
                }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            
            return {
                "name": model_name,
                "available": False,
                "error": str(e)
            }
    
    async def get_available_models(self) -> List[str]:
        """
        Get available models.
        
        Returns:
            List of available model names
        """
        return list(self.models.keys())
"""
Knowledge management for the AI assistant.

This module provides knowledge management functionality for the AI assistant.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge manager for the AI assistant.
    
    This class manages knowledge for the AI assistant.
    """
    
    def __init__(self, knowledge_path: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            knowledge_path: Path to knowledge base
        """
        self.knowledge_path = Path(knowledge_path)
        
        # Knowledge base
        self.knowledge_base = {}
        
        # Vector store for search
        self.vector_store = None
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        try:
            # Create knowledge directory if it doesn't exist
            self.knowledge_path.mkdir(parents=True, exist_ok=True)
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            logger.info("Knowledge manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge manager: {e}")
    
    async def _load_knowledge_base(self):
        """Load knowledge base."""
        try:
            # Knowledge base file
            knowledge_file = self.knowledge_path / "knowledge_base.json"
            
            # Check if file exists
            if knowledge_file.exists():
                # Load knowledge base
                with open(knowledge_file, "r") as f:
                    self.knowledge_base = json.load(f)
                
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} items")
            else:
                # Create empty knowledge base
                self.knowledge_base = {}
                
                logger.info("Created empty knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            
            # Create empty knowledge base
            self.knowledge_base = {}
    
    async def _initialize_vector_store(self):
        """Initialize vector store for search."""
        try:
            # Try to import sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                
                # Load model
                self.vector_store = SentenceTransformer("all-MiniLM-L6-v2")
                
                logger.info("Vector store initialized")
                
            except ImportError:
                logger.warning("sentence-transformers not available, using fallback search")
                
                # Use fallback search
                self.vector_store = None
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            
            # Use fallback search
            self.vector_store = None
    
    async def save_knowledge_base(self):
        """Save knowledge base."""
        try:
            # Knowledge base file
            knowledge_file = self.knowledge_path / "knowledge_base.json"
            
            # Save knowledge base
            with open(knowledge_file, "w") as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            logger.info(f"Saved knowledge base with {len(self.knowledge_base)} items")
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            
        Returns:
            Knowledge ID
        """
        try:
            # Generate knowledge ID
            knowledge_id = str(uuid.uuid4())
            
            # Add knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Added knowledge: {knowledge_id}")
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return None
            
            # Get knowledge
            return self.knowledge_base[knowledge_id]
            
        except Exception as e:
            logger.error(f"Error getting knowledge: {e}")
            return None
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return False
            
            # Update knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Updated knowledge: {knowledge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return False
            
            # Delete knowledge
            del self.knowledge_base[knowledge_id]
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Deleted knowledge: {knowledge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")
            return False
   async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Check if knowledge base is empty
            if not self.knowledge_base:
                return []
            
            # Check if vector store is available
            if self.vector_store is not None:
                # Use vector store for search
                return await self._vector_search(query, limit)
            else:
                # Use fallback search
                return await self._fallback_search(query, limit)
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search knowledge base using vector store.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Encode query
            query_embedding = self.vector_store.encode(query)
            
            # Encode knowledge
            knowledge_texts = []
            knowledge_ids = []
            
            for knowledge_id, knowledge in self.knowledge_base.items():
                # Get text
                text = f"{knowledge.get('title', '')} {knowledge.get('content', '')}"
                
                # Add to list
                knowledge_texts.append(text)
                knowledge_ids.append(knowledge_id)
            
            # Encode knowledge
            knowledge_embeddings = self.vector_store.encode(knowledge_texts)
            
            # Calculate similarity
            from numpy import dot
            from numpy.linalg import norm
            
            # Calculate cosine similarity
            similarities = [
                dot(query_embedding, embedding) / (norm(query_embedding) * norm(embedding))
                for embedding in knowledge_embeddings
            ]
            
            # Sort by similarity
            results = sorted(
                [
                    {
                        "id": knowledge_id,
                        "knowledge": self.knowledge_base[knowledge_id],
                        "score": float(similarity)
                    }
                    for knowledge_id, similarity in zip(knowledge_ids, similarities)
                ],
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            
            # Fallback to basic search
            return await self._fallback_search(query, limit)
    
    async def _fallback_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search knowledge base using fallback search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Simple keyword search
            query_terms = query.lower().split()
            
            # Calculate scores
            results = []
            
            for knowledge_id, knowledge in self.knowledge_base.items():
                # Get text
                title = knowledge.get("title", "").lower()
                content = knowledge.get("content", "").lower()
                
                # Calculate score
                score = 0.0
                
                for term in query_terms:
                    # Check title
                    if term in title:
                        score += 2.0
                    
                    # Check content
                    if term in content:
                        score += 1.0
                
                # Normalize score
                if len(query_terms) > 0:
                    score /= len(query_terms)
                
                # Add to results if score > 0
                if score > 0:
                    results.append({
                        "id": knowledge_id,
                        "knowledge": knowledge,
                        "score": score
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing fallback search: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge.
        
        Returns:
            List of all knowledge items
        """
        try:
            # Convert knowledge base to list
            return [
                {
                    "id": knowledge_id,
                    "knowledge": knowledge
                }
                for knowledge_id, knowledge in self.knowledge_base.items()
            ]
            
        except Exception as e:
            logger.error(f"Error getting all knowledge: {e}")
            return []
"""
Configuration management for the AI assistant.

This module provides configuration management functionality for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the AI assistant.
    
    This class manages configuration for the AI assistant.
    """
    
    def __init__(self, config_path: str = "./config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration
        """
        self.config_path = Path(config_path)
        
        # Default configuration
        self.default_config = {
            # General
            "name": "AI Assistant",
            "version": "1.0.0",
            
            # Session
            "session_timeout": 300.0,  # 5 minutes
            
            # Models
            "model_path": "./models",
            "default_model": "text-generation",
            
            # Knowledge
            "knowledge_path": "./knowledge",
            
            # Voice
            "voice_enabled": True,
            "voice_language": "en",
            
            # Vision
            "vision_enabled": True,
            
            # User
            "user_path": "./users",
            
            # Logging
            "log_level": "INFO",
            "log_file": "./logs/assistant.log",
            
            # Advanced
            "debug_mode": False,
            "max_tokens": 100,
            "temperature": 0.7
        }
    
    async def initialize(self):
        """Initialize the configuration manager."""
        try:
            # Create configuration directory if it doesn't exist
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing configuration manager: {e}")
    
    async def load_config(self) -> Dict[str, Any]:
        """
        Load configuration.
        
        Returns:
            Configuration
        """
        try:
            # Configuration file
            config_file = self.config_path / "config.json"
            
            # Check if file exists
            if config_file.exists():
                # Load configuration
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                logger.info("Loaded configuration")
                
                # Merge with default configuration
                merged_config = self.default_config.copy()
                merged_config.update(config)
                
                return merged_config
            else:
                # Create default configuration
                config = self.default_config.copy()
                
                # Save configuration
                await self.save_config(config)
                
                logger.info("Created default configuration")
                
                return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
            # Return default configuration
            return self.default_config.copy()
    
    async def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration.
        
        Args:
            config: Configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Configuration file
            config_file = self.config_path / "config.json"
            
            # Save configuration
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info("Saved configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    async def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value


from typing import Dict, Any, Optional, List, Set, Tuple, Union
from loguru import logger
import asyncio
import uuid
import json
import time
from datetime import datetime
import numpy as np
import threading
import queue
import os
import re
from pathlib import Path
import cv2
import pyaudio
import wave
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import whisper
from PIL import Image
import pytesseract
from pydantic import BaseModel, Field

from edgenative_umaas.utils.event_bus import EventBus
from edgenative_umaas.security.security_manager import SecurityManager

class ContextWindow(BaseModel):
    """Context window for maintaining conversation state and context."""
    session_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context_embeddings: Optional[List[float]] = None
    active_tasks: Dict[str, Any] = Field(default_factory=dict)
    screen_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    system_state: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message to the context window."""
        self.messages.append(message)
        # Limit the number of messages to prevent context explosion
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

class CognitiveCollaborationSystem:
    """
    Cognitive Collaboration System.
    
    A revolutionary yet attainable AI collaboration system that enables:
    1. Continuous contextual awareness through multimodal inputs
    2. Proactive assistance based on user behavior and system state
    3. Seamless voice-driven interaction with natural conversation
    4. Real-time screen understanding and intelligent automation
    5. Adaptive learning from user interactions and feedback
    """
    
    def __init__(self, event_bus: EventBus, security_manager: SecurityManager, config: Dict[str, Any]):
        """Initialize the Cognitive Collaboration System."""
        self.event_bus = event_bus
        self.security_manager = security_manager
        self.config = config
        
        # Core components
        self.active_sessions = {}  # session_id -> ContextWindow
        self.voice_streams = {}    # session_id -> audio stream
        self.screen_streams = {}   # session_id -> screen capture stream
        
        # AI models
        self.speech_recognizer = None  # Whisper model for speech recognition
        self.text_generator = None     # LLM for text generation
        self.vision_analyzer = None    # Vision model for screen understanding
        self.embedding_model = None    # Model for semantic embeddings
        
        # Processing queues
        self.voice_queue = queue.Queue()
        self.screen_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Knowledge base
        self.knowledge_vectors = []
        self.knowledge_texts = []
        
        # User behavior patterns
        self.user_patterns = {}  # user_id -> patterns
        
        # System capabilities registry
        self.capabilities = {
            "voice_commands": self._process_voice_command,
            "screen_analysis": self._analyze_screen,
            "task_automation": self._automate_task,
            "information_retrieval": self._retrieve_information,
            "code_generation": self._generate_code,
            "data_visualization": self._visualize_data,
            "system_monitoring": self._monitor_system,
            "predictive_assistance": self._provide_predictive_assistance
        }
        
        # Worker threads
        self.workers = []
    
    async def initialize(self) -> bool:
        """Initialize the Cognitive Collaboration System."""
        logger.info("Initializing Cognitive Collaboration System")
        
        try:
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Start worker threads
            self._start_workers()
            
            # Subscribe to events
            await self.event_bus.subscribe("user.joined", self._handle_user_joined)
            await self.event_bus.subscribe("user.left", self._handle_user_left)
            await self.event_bus.subscribe("user.message", self._handle_user_message)
            await self.event_bus.subscribe("system.state_changed", self._handle_system_state_changed)
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            logger.info("Cognitive Collaboration System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Collaboration System: {e}")
            return False
    
    async def _initialize_ai_models(self):
        """Initialize AI models using the most efficient approach for edge deployment."""
        # Use a background thread for model loading to avoid blocking
        def load_models():
            try:
                # 1. Initialize Whisper for speech recognition (small model for edge devices)
                self.speech_recognizer = whisper.load_model("base")
                
                # 2. Initialize vision model for screen understanding
                self.vision_analyzer = pipeline("image-classification", 
                                               model="microsoft/resnet-50")
                
                # 3. Initialize text embedding model
                self.embedding_model = pipeline("feature-extraction", 
                                               model="sentence-transformers/all-MiniLM-L6-v2")
                
                # 4. Initialize text generation model
                # Use a quantized model for edge efficiency
                model_name = "TheBloke/Llama-2-7B-Chat-GGML"
                self.text_generator = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto",
                    load_in_8bit=True  # 8-bit quantization for memory efficiency
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                logger.info("AI models loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading AI models: {e}")
                raise
        
        # Start model loading in background
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        
        # Wait for models to load with a timeout
        for _ in range(30):  # 30 second timeout
            if self.speech_recognizer and self.vision_analyzer and self.embedding_model and self.text_generator:
                return
            await asyncio.sleep(1)
        
        if not (self.speech_recognizer and self.vision_analyzer and self.embedding_model and self.text_generator):
            raise TimeoutError("Timed out waiting for AI models to load")
    
    def _start_workers(self):
        """Start worker threads for processing different input streams."""
        # 1. Voice processing worker
        voice_worker = threading.Thread(target=self._voice_processing_worker)
        voice_worker.daemon = True
        voice_worker.start()
        self.workers.append(voice_worker)
        
        # 2. Screen analysis worker
        screen_worker = threading.Thread(target=self._screen_processing_worker)
        screen_worker.daemon = True
        screen_worker.start()
        self.workers.append(screen_worker)
        
        # 3. Command execution worker
        command_worker = threading.Thread(target=self._command_processing_worker)
        command_worker.daemon = True
        command_worker.start()
        self.workers.append(command_worker)
        
        logger.info(f"Started {len(self.workers)} worker threads")
    
    async def _load_knowledge_base(self):
        """Load and index the knowledge base for quick retrieval."""
        knowledge_dir = Path(self.config.get("knowledge_dir", "./knowledge"))
        
        if not knowledge_dir.exists():
            logger.warning(f"Knowledge directory {knowledge_dir} does not exist. Creating it.")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Load all text files in the knowledge directory
        for file_path in knowledge_dir.glob("**/*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create embedding for the content
                embedding = self.embedding_model(content)
                embedding_vector = np.mean(embedding[0], axis=0)
                
                # Store the content and its embedding
                self.knowledge_texts.append(content)
                self.knowledge_vectors.append(embedding_vector)
                
            except Exception as e:
                logger.error(f"Error loading knowledge file {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge items")
    
    async def start_session(self, user_id: str) -> str:
        """Start a new collaboration session."""
        # Create a new session
        session_id = str(uuid.uuid4())
        
        # Initialize context window
        self.active_sessions[session_id] = ContextWindow(
            session_id=session_id,
            user_preferences=await self._load_user_preferences(user_id),
            system_state={"start_time": datetime.now().isoformat()}
        )
        
        # Start voice and screen streams
        await self._start_voice_stream(session_id)
        await self._start_screen_stream(session_id)
        
        logger.info(f"Started collaboration session {session_id} for user {user_id}")
        
        # Send welcome message
        await self.send_assistant_message(
            session_id,
            "I'm ready to collaborate with you. You can speak naturally or type commands. "
            "I'll observe your screen to provide contextual assistance when needed."
        )
        
        return session_id
    
    async def _start_voice_stream(self, session_id: str):
        """Start voice input stream for a session."""
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
            stream_callback=lambda in_data, frame_count, time_info, status_flags: 
                self._voice_callback(session_id, in_data, frame_count, time_info, status_flags)
        )
        
        # Store the stream
        self.voice_streams[session_id] = {
            "stream": stream,
            "pyaudio": p,
            "buffer": bytearray(),
            "last_process_time": time.time()
        }
        
        logger.info(f"Started voice stream for session {session_id}")
    
    def _voice_callback(self, session_id, in_data, frame_count, time_info, status_flags):
        """Callback for voice stream data."""
        if session_id not in self.voice_streams:
            return (in_data, pyaudio.paContinue)
        
        # Add data to buffer
        self.voice_streams[session_id]["buffer"].extend(in_data)
        
        # Process buffer if it's been at least 1 second since last processing
        current_time = time.time()
        if current_time - self.voice_streams[session_id]["last_process_time"] >= 1.0:
            # Copy buffer and clear it
            buffer_copy = self.voice_streams[session_id]["buffer"].copy()
            self.voice_streams[session_id]["buffer"] = bytearray()
            
            # Add to processing queue
            self.voice_queue.put((session_id, buffer_copy))
            
            # Update last process time
            self.voice_streams[session_id]["last_process_time"] = current_time
        
        return (in_data, pyaudio.paContinue)
    
    async def _start_screen_stream(self, session_id: str):
        """Start screen capture stream for a session."""
        # Initialize screen capture
        self.screen_streams[session_id] = {
            "active": True,
            "last_capture_time": 0,
            "capture_interval": 1.0,  # Capture every 1 second
            "last_image": None
        }
        
        # Start screen capture thread
        thread = threading.Thread(
            target=self._screen_capture_worker,
            args=(session_id,)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started screen capture for session {session_id}")
    
    def _screen_capture_worker(self, session_id: str):
        """Worker thread for screen capture."""
        try:
            import mss
            
            with mss.mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1]
                
                while session_id in self.screen_streams and self.screen_streams[session_id]["active"]:
                    current_time = time.time()
                    
                    # Capture screen at specified interval
                    if current_time - self.screen_streams[session_id]["last_capture_time"] >= self.screen_streams[session_id]["capture_interval"]:
                        # Capture screen
                        screenshot = sct.grab(monitor)
                        
                        # Convert to numpy array
                        img = np.array(screenshot)
                        
                        # Resize to reduce processing load
                        img = cv2.resize(img, (800, 600))
                        
                        # Store the image
                        self.screen_streams[session_id]["last_image"] = img
                        
                        # Add to processing queue
                        self.screen_queue.put((session_id, img.copy()))
                        
                        # Update last capture time
                        self.screen_streams[session_id]["last_capture_time"] = current_time
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in screen capture worker for session {session_id}: {e}")
    
    def _voice_processing_worker(self):
        """Worker thread for processing voice input."""
        while True:
            try:
                # Get item from queue
                session_id, audio_data = self.voice_queue.get(timeout=1.0)
                
                # Skip if speech recognizer is not initialized
                if not self.speech_recognizer:
                    self.voice_queue.task_done()
                    continue
                
                # Convert audio data to WAV format for Whisper
                with wave.open("temp_audio.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                
                # Transcribe audio
                result = self.speech_recognizer.transcribe("temp_audio.wav")
                transcription = result["text"].strip()
                
                # Skip if empty
                if not transcription:
                    self.voice_queue.task_done()
                    continue
                
                # Process the transcription
                logger.info(f"Voice input from session {session_id}: {transcription}")
                
                # Create a task to handle the voice input
                asyncio.run_coroutine_threadsafe(
                    self._handle_voice_input(session_id, transc
               # Create a task to handle the voice input
                asyncio.run_coroutine_threadsafe(
                    self._handle_voice_input(session_id, transcription),
                    asyncio.get_event_loop()
                )
                
                # Clean up
                if os.path.exists("temp_audio.wav"):
                    os.remove("temp_audio.wav")
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in voice processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.voice_queue.task_done()
    
    def _screen_processing_worker(self):
        """Worker thread for processing screen captures."""
        while True:
            try:
                # Get item from queue
                session_id, image = self.screen_queue.get(timeout=1.0)
                
                # Skip if vision analyzer is not initialized
                if not self.vision_analyzer:
                    self.screen_queue.task_done()
                    continue
                
                # Process the screen capture
                screen_context = self._extract_screen_context(image)
                
                # Update context window with screen context
                if session_id in self.active_sessions:
                    self.active_sessions[session_id].screen_context = screen_context
                
                # Check for significant changes that might require proactive assistance
                if self._should_provide_assistance(session_id, screen_context):
                    # Create a task to provide assistance
                    asyncio.run_coroutine_threadsafe(
                        self._provide_proactive_assistance(session_id, screen_context),
                        asyncio.get_event_loop()
                    )
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in screen processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.screen_queue.task_done()
    
    def _command_processing_worker(self):
        """Worker thread for processing commands."""
        while True:
            try:
                # Get item from queue
                session_id, command, args = self.command_queue.get(timeout=1.0)
                
                # Process the command
                logger.info(f"Processing command from session {session_id}: {command} {args}")
                
                # Execute the command
                result = None
                if command in self.capabilities:
                    try:
                        # Execute the command handler
                        result = self.capabilities[command](session_id, args)
                        
                        # Create a task to send the result
                        if result:
                            asyncio.run_coroutine_threadsafe(
                                self.send_assistant_message(session_id, result),
                                asyncio.get_event_loop()
                            )
                    
                    except Exception as e:
                        error_message = f"Error executing command {command}: {e}"
                        logger.error(error_message)
                        
                        # Send error message
                        asyncio.run_coroutine_threadsafe(
                            self.send_assistant_message(session_id, error_message),
                            asyncio.get_event_loop()
                        )
                else:
                    # Unknown command
                    unknown_command_message = f"Unknown command: {command}. Type 'help' for available commands."
                    
                    # Send unknown command message
                    asyncio.run_coroutine_threadsafe(
                        self.send_assistant_message(session_id, unknown_command_message),
                        asyncio.get_event_loop()
                    )
                
            except queue.Empty:
                # No items in queue, continue
                pass
                
            except Exception as e:
                logger.error(f"Error in command processing worker: {e}")
                
            finally:
                # Mark task as done if we got one
                if 'session_id' in locals():
                    self.command_queue.task_done()
    
    def _extract_screen_context(self, image) -> Dict[str, Any]:
        """
        Extract context from screen capture using computer vision.
        
        This is a revolutionary feature that provides real-time understanding
        of what the user is seeing and doing on their screen.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "elements": [],
            "text": [],
            "recognized_apps": [],
            "active_window": None
        }
        
        try:
            # 1. Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Extract text using OCR
            text = pytesseract.image_to_string(Image.fromarray(rgb_image))
            if text:
                context["text"] = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 3. Detect UI elements using edge detection and contour analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours to identify UI elements
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify the element
                element_type = self._classify_ui_element(rgb_image[y:y+h, x:x+w])
                
                # Add to elements list
                context["elements"].append({
                    "id": f"element_{i}",
                    "type": element_type,
                    "bounds": {"x": x, "y": y, "width": w, "height": h}
                })
            
            # 4. Recognize applications using the vision analyzer
            # Convert image to PIL format for the vision model
            pil_image = Image.fromarray(rgb_image)
            
            # Get predictions
            predictions = self.vision_analyzer(pil_image)
            
            # Extract recognized applications
            for prediction in predictions:
                if prediction["score"] > 0.7:  # Confidence threshold
                    context["recognized_apps"].append({
                        "name": prediction["label"],
                        "confidence": prediction["score"]
                    })
            
            # 5. Determine active window based on UI analysis
            if context["elements"]:
                # Heuristic: The largest element near the top of the screen is likely the active window
                top_elements = sorted(context["elements"], key=lambda e: e["bounds"]["y"])[:5]
                largest_element = max(top_elements, key=lambda e: e["bounds"]["width"] * e["bounds"]["height"])
                context["active_window"] = largest_element["id"]
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting screen context: {e}")
            return context
    
    def _classify_ui_element(self, element_image) -> str:
        """Classify a UI element based on its appearance."""
        # This is a simplified version - in a real implementation, 
        # we would use a trained model for more accurate classification
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple heuristic classification
        if aspect_ratio > 5:
            return "menu_bar"
        elif aspect_ratio > 3:
            return "toolbar"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif w > 300 and h > 200:
            if avg_brightness > 200:
                return "dialog"
            else:
                return "window"
        elif w < 100 and h < 50:
            if avg_brightness > 200:
                return "button"
            else:
                return "icon"
        elif w > 200 and h < 50:
            return "text_field"
        else:
            return "panel"
    
    def _should_provide_assistance(self, session_id: str, screen_context: Dict[str, Any]) -> bool:
        """
        Determine if proactive assistance should be provided based on screen context.
        
        This is a revolutionary feature that allows the AI to offer help exactly
        when needed without explicit requests.
        """
        if session_id not in self.active_sessions:
            return False
        
        context_window = self.active_sessions[session_id]
        previous_context = context_window.screen_context
        
        # If no previous context, don't provide assistance yet
        if not previous_context:
            return False
        
        # Check for significant changes in active window
        if (previous_context.get("active_window") != screen_context.get("active_window") and
            screen_context.get("active_window") is not None):
            return True
        
        # Check for error messages in text
        error_patterns = ["error", "exception", "failed", "warning", "invalid"]
        for text in screen_context.get("text", []):
            if any(pattern in text.lower() for pattern in error_patterns):
                return True
        
        # Check for complex UI with many elements (user might need help navigating)
        if len(screen_context.get("elements", [])) > 15:
            # But only if this is a change from before
            if len(previous_context.get("elements", [])) < 10:
                return True
        
        # Check for recognized applications that might need assistance
        assistance_apps = ["terminal", "code editor", "database", "configuration"]
        for app in screen_context.get("recognized_apps", []):
            if any(assist_app in app["name"].lower() for assist_app in assistance_apps):
                # Only provide assistance if this is a newly detected app
                previous_apps = [a["name"] for a in previous_context.get("recognized_apps", [])]
                if app["name"] not in previous_apps:
                    return True
        
        return False
    
    async def _provide_proactive_assistance(self, session_id: str, screen_context: Dict[str, Any]):
        """
        Provide proactive assistance based on screen context.
        
        This is where the AI becomes truly collaborative by offering
        timely and relevant help without explicit requests.
        """
        if session_id not in self.active_sessions:
            return
        
        context_window = self.active_sessions[session_id]
        
        # Generate assistance message based on screen context
        assistance_message = await self._generate_assistance_message(session_id, screen_context)
        
        # Send the assistance message
        if assistance_message:
            await self.send_assistant_message(
                session_id,
                f"I noticed you might need help: {assistance_message}\n\nWould you like me to assist with this?"
            )
    
    async def _generate_assistance_message(self, session_id: str, screen_context: Dict[str, Any]) -> str:
        """Generate a contextual assistance message."""
        # Extract key information from screen context
        active_app = None
        if screen_context.get("recognized_apps"):
            active_app = screen_context["recognized_apps"][0]["name"]
        
        error_text = None
        for text in screen_context.get("text", []):
            error_patterns = ["error", "exception", "failed", "warning", "invalid"]
            if any(pattern in text.lower() for pattern in error_patterns):
                error_text = text
                break
        
        # Generate appropriate assistance
        if error_text:
            # Search knowledge base for similar errors
            similar_knowledge = await self._search_knowledge_base(error_text)
            if similar_knowledge:
                return f"I see an error: '{error_text}'. Based on my knowledge, this might be related to: {similar_knowledge}"
            else:
                return f"I notice you're encountering an error: '{error_text}'. Would you like me to help troubleshoot this?"
        
        elif active_app:
            if "terminal" in active_app.lower():
                return "I see you're working in the terminal. I can help with command suggestions or explain command outputs."
            
            elif "code" in active_app.lower() or "editor" in active_app.lower():
                return "I notice you're coding. I can help with code suggestions, debugging, or explaining concepts."
            
            elif "browser" in active_app.lower():
                return "I see you're browsing. I can help search for information or explain concepts on the current page."
            
            elif "database" in active_app.lower():
                return "I notice you're working with a database. I can help with query optimization or data modeling."
        
        # Default assistance based on UI complexity
        element_count = len(screen_context.get("elements", []))
        if element_count > 15:
            return "I notice you're working with a complex interface. I can help navigate or explain functionality."
        
        return None
    
    async def _search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for relevant information."""
        if not self.knowledge_vectors or not self.embedding_model:
            return None
        
        # Generate embedding for the query
        query_embedding = self.embedding_model(query)
        query_vector = np.mean(query_embedding[0], axis=0)
        
        # Calculate similarity with all knowledge items
        similarities = []
        for i, vector in enumerate(self.knowledge_vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most similar item if similarity is above threshold
        if similarities and similarities[0][1] > 0.7:
            index = similarities[0][0]
            # Return a snippet of the knowledge text
            text = self.knowledge_texts[index]
            # Extract a relevant snippet (first 200 characters)
            return text[:200] + "..." if len(text) > 200 else text
        
        return None
    
    async def _handle_voice_input(self, session_id: str, text: str):
        """Handle voice input from the user."""
        if session_id not in self.active_sessions:
            logger.warning(f"Received voice input for unknown session {session_id}")
            return
        
        logger.info(f"Processing voice input for session {session_id}: {text}")
        
        # Add to context window
        context_window = self.active_sessions[session_id]
        context_window.add_message({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat(),
            "type": "voice"
        })
        
        # Check if it's a command
        command_match = re.match(r"^([\w]+)(?:\s+(.+))?$", text.strip().lower())
        if command_match and command_match.group(1) in self.capabilities:
       # Check if it's a command
        command_match = re.match(r"^([\w]+)(?:\s+(.+))?$", text.strip().lower())
        if command_match and command_match.group(1) in self.capabilities:
            command = command_match.group(1)
            args = command_match.group(2) or ""
            
            # Add to command queue
            self.command_queue.put((session_id, command, args))
            
            return
        
        # Process as natural language
        await self._process_natural_language(session_id, text)
    
    async def _process_natural_language(self, session_id: str, text: str):
        """
        Process natural language input from the user.
        
        This is where the AI understands and responds to conversational input,
        making the interaction feel natural and human-like.
        """
        if session_id not in self.active_sessions:
            return
        
        context_window = self.active_sessions[session_id]
        
        # Prepare context for the language model
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps users with their tasks. You can see their screen and understand their voice commands. Be concise, helpful, and proactive."}
        ]
        
        # Add recent conversation history
        for msg in context_window.messages[-10:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # Add screen context if available
        if context_window.screen_context:
            screen_summary = self._summarize_screen_context(context_window.screen_context)
            messages.append({"role": "system", "content": f"Current screen context: {screen_summary}"})
        
        try:
            # Generate response using the language model
            response = await self._generate_response(messages)
            
            # Send the response
            await self.send_assistant_message(session_id, response)
            
            # Check for actionable insights in the response
            await self._extract_and_execute_actions(session_id, response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await self.send_assistant_message(
                session_id,
                "I'm sorry, I encountered an error while processing your request. Please try again."
            )
    
    def _summarize_screen_context(self, screen_context: Dict[str, Any]) -> str:
        """Summarize screen context for inclusion in LLM prompt."""
        summary_parts = []
        
        # Add active window
        if screen_context.get("active_window"):
            active_element = next(
                (e for e in screen_context.get("elements", []) 
                 if e.get("id") == screen_context["active_window"]),
                None
            )
            if active_element:
                summary_parts.append(f"Active window: {active_element.get('type', 'window')}")
        
        # Add recognized applications
        if screen_context.get("recognized_apps"):
            apps = [app["name"] for app in screen_context["recognized_apps"][:3]]
            summary_parts.append(f"Applications: {', '.join(apps)}")
        
        # Add UI element summary
        if screen_context.get("elements"):
            element_types = {}
            for element in screen_context["elements"]:
                element_type = element.get("type", "unknown")
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            element_summary = ", ".join(f"{count} {element_type}s" for element_type, count in element_types.items())
            summary_parts.append(f"UI elements: {element_summary}")
        
        # Add text summary (first few items)
        if screen_context.get("text"):
            text_items = screen_context["text"][:3]
            if text_items:
                text_summary = "; ".join(text_items)
                if len(text_summary) > 100:
                    text_summary = text_summary[:100] + "..."
                summary_parts.append(f"Visible text: {text_summary}")
        
        return " | ".join(summary_parts)
    
    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the language model."""
        if not self.text_generator or not self.tokenizer:
            return "I'm still initializing my language capabilities. Please try again in a moment."
        
        try:
            # Convert messages to a prompt
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.text_generator.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.text_generator.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            
            # Remove any assistant prefix
            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages to a prompt for the language model."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _extract_and_execute_actions(self, session_id: str, response: str):
        """
        Extract and execute actions from the assistant's response.
        
        This enables the AI to not just talk about actions but actually perform them,
        making it a true collaborative partner.
        """
        # Look for action patterns in the response
        action_patterns = [
            (r"I'll search for\s+(.+?)[\.\n]", "search"),
            (r"I'll analyze\s+(.+?)[\.\n]", "analyze_data"),
            (r"I'll create\s+(.+?)[\.\n]", "create"),
            (r"I'll generate\s+(.+?)[\.\n]", "generate_code"),
            (r"I'll show you\s+(.+?)[\.\n]", "visualize_data"),
            (r"I'll monitor\s+(.+?)[\.\n]", "monitor_system")
        ]
        
        for pattern, action in action_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                # Add to command queue
                self.command_queue.put((session_id, action, match))
                
                logger.info(f"Extracted action from response: {action} {match}")
    
    async def send_assistant_message(self, session_id: str, content: str):
        """Send a message from the assistant to the user."""
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot send message to unknown session {session_id}")
            return
        
        # Create message
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "text"
        }
        
        # Add to context window
        self.active_sessions[session_id].add_message(message)
        
        # Publish message event
        await self.event_bus.publish("assistant.message", {
            "session_id": session_id,
            "message": message
        })
        
        logger.info(f"Sent assistant message to session {session_id}: {content[:50]}...")
    
    async def end_session(self, session_id: str):
        """End a collaboration session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Cannot end unknown session {session_id}")
            return False
        
        # Stop voice stream
        if session_id in self.voice_streams:
            try:
                self.voice_streams[session_id]["stream"].stop_stream()
                self.voice_streams[session_id]["stream"].close()
                self.voice_streams[session_id]["pyaudio"].terminate()
                del self.voice_streams[session_id]
            except Exception as e:
                logger.error(f"Error stopping voice stream: {e}")
        
        # Stop screen stream
        if session_id in self.screen_streams:
            try:
                self.screen_streams[session_id]["active"] = False
                del self.screen_streams[session_id]
            except Exception as e:
                logger.error(f"Error stopping screen stream: {e}")
        
        # Remove session
        del self.active_sessions[session_id]
        
        logger.info(f"Ended collaboration session {session_id}")
        return True
    
    # Command handlers
    def _process_voice_command(self, session_id: str, args: str) -> str:
        """Process a voice command."""
        return f"Processing voice command: {args}"
    
    def _analyze_screen(self, session_id: str, args: str) -> str:
        """Analyze the current screen."""
        if session_id not in self.screen_streams or not self.screen_streams[session_id].get("last_image") is not None:
            return "No screen capture available to analyze."
        
        # Get the last captured image
        image = self.screen_streams[session_id]["last_image"]
        
        # Extract context
        context = self._extract_screen_context(image)
        
        # Generate a human-readable analysis
        analysis = []
        
        if context.get("recognized_apps"):
            apps = [f"{app['name']} ({app['confidence']:.2f})" for app in context["recognized_apps"]]
            analysis.append(f"Recognized applications: {', '.join(apps)}")
        
        if context.get("elements"):
            element_types = {}
            for element in context["elements"]:
                element_type = element.get("type", "unknown")
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            element_summary = ", ".join(f"{count} {element_type}s" for element_type, count in element_types.items())
            analysis.append(f"UI elements: {element_summary}")
        
        if context.get("text"):
            text_count = len(context["text"])
            analysis.append(f"Detected {text_count} text elements")
            
            # Include a few examples
            if text_count > 0:
                examples = context["text"][:3]
                analysis.append(f"Text examples: {'; '.join(examples)}")
        
        return "\n".join(analysis)
    
    def _automate_task(self, session_id: str, args: str) -> str:
        """Automate a task based on screen context."""
        return f"Automating task: {args}"
    
    def _retrieve_information(self, session_id: str, args: str) -> str:
        """Retrieve information from knowledge base."""
        # Search knowledge base
        result = asyncio.run(self._search_knowledge_base(args))
        
        if result:
            return f"Found relevant information: {result}"
        else:
            return f"No relevant information found for: {args}"
    
    def _generate_code(self, session_id: str, args: str) -> str:
        """Generate code based on description."""
        return f"Generating code for: {args}"
    
    def _visualize_data(self, session_id: str, args: str) -> str:
        """Visualize data."""
        return f"Visualizing data: {args}"
    
    def _monitor_system(self, session_id: str, args: str) -> str:
        """Monitor system metrics."""
        return f"Monitoring system: {args}"
    
    def _provide_predictive_assistance(self, session_id: str, args: str) -> str:
        """Provide predictive assistance based on user patterns."""
        return f"Providing predictive assistance: {args}"
    
    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences."""
        # In a real implementation, this would load from a database
        return {
            "voice_enabled": True,
            "screen_capture_interval": 1.0,
            "proactive_assistance": True,
            "preferred_communication_style": "conversational"
        }
    
    async def _handle_user_joined(self, event_data: Dict[str, Any]):
        """Handle user joined event."""
        user_id = event_data.get("user_id")
        if not user_id:
            return
        
        # Start a session for the user
        session_id = await self.start_session(user_id)
        
        # Publish session started event
        await self.event_bus.publish("assistant.session_started", {
            "user_id": user_id,
            "session_id": session_id
        })
    
    async def _handle_user_left(self, event_data: Dict[str, Any]):
        """Handle user left event."""
        user_id = event_data.get("user_id")
        session_id = event_data.get("session_id")
        
        if not user_id or not session_id:
            return
        
        # End the session
        await self.end_session(session_id)
    
    async def _handle_user_message(self, event_data: Dict[str, Any]):
        """Handle user message event."""
        session_id = event_data.get("session_id")
        message = event_data.get("message")
        
        if not session_id or not message:
            return
        
        # Add to context window
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_message(message)
        
        # Process the message
        content = message.get("content", "")
        content_type = message.get("content_type", "text")
        
        if content_type == "text":
            await self._process_natural_language(session_id, content)
        elif content_type == "command":
            # Parse command
            parts = content.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Add to command queue
            self.command_queue.put((session_id, command, args))
    
    async def _handle_system_state_changed(self, event_data: Dict[str, Any]):
        """Handle system state changed event."""
        # Update system state for all active sessions
        for session_i
   async def _handle_system_state_changed(self, event_data: Dict[str, Any]):
        """Handle system state changed event."""
        # Update system state for all active sessions
        for session_id, context_window in self.active_sessions.items():
            # Update system state
            context_window.system_state.update(event_data)
            
            # Check if proactive notification is needed
            if self._should_notify_system_change(session_id, event_data):
                await self.send_assistant_message(
                    session_id,
                    f"System update: {self._format_system_change(event_data)}"
                )
    
    def _should_notify_system_change(self, session_id: str, event_data: Dict[str, Any]) -> bool:
        """Determine if a system change should trigger a notification."""
        # Check user preferences
        if session_id in self.active_sessions:
            user_preferences = self.active_sessions[session_id].user_preferences
            if not user_preferences.get("proactive_assistance", True):
                return False
        
        # Check importance of the change
        importance = event_data.get("importance", "low")
        if importance == "high":
            return True
        elif importance == "medium":
            # Only notify if the user isn't actively engaged
            if session_id in self.active_sessions:
                # Check if there was recent user activity (within last 5 minutes)
                recent_messages = [
                    msg for msg in self.active_sessions[session_id].messages[-10:]
                    if msg.get("role") == "user"
                ]
                
                if recent_messages:
                    last_message_time = datetime.fromisoformat(recent_messages[-1].get("timestamp", ""))
                    now = datetime.now()
                    time_diff = (now - last_message_time).total_seconds()
                    
                    # If user was active in the last 5 minutes, don't interrupt
                    if time_diff < 300:
                        return False
            
            return True
        
        # Low importance changes don't trigger notifications
        return False
    
    def _format_system_change(self, event_data: Dict[str, Any]) -> str:
        """Format a system change event for user notification."""
        event_type = event_data.get("type", "unknown")
        
        if event_type == "resource_warning":
            resource = event_data.get("resource", "unknown")
            level = event_data.get("level", "warning")
            details = event_data.get("details", "")
            return f"{level.upper()}: {resource} resource issue. {details}"
        
        elif event_type == "task_completed":
            task = event_data.get("task", "unknown")
            result = event_data.get("result", "completed")
            return f"Task '{task}' has been completed with result: {result}"
        
        elif event_type == "security_alert":
            alert = event_data.get("alert", "unknown")
            severity = event_data.get("severity", "medium")
            return f"{severity.upper()} security alert: {alert}"
        
        elif event_type == "update_available":
            component = event_data.get("component", "system")
            version = event_data.get("version", "unknown")
            return f"Update available for {component}: version {version}"
        
        else:
            # Generic formatting for unknown event types
            return ", ".join(f"{k}: {v}" for k, v in event_data.items() if k != "type")

class VoiceProcessor:
    """
    Voice processor for continuous speech recognition.
    
    This component enables the revolutionary natural voice interaction
    with the AI assistant.
    """
    
    def __init__(self, session_id: str, callback):
        """
        Initialize the voice processor.
        
        Args:
            session_id: Session ID
            callback: Callback function to call with transcribed text
        """
        self.session_id = session_id
        self.callback = callback
        self.active = False
        self.thread = None
        self.audio_queue = queue.Queue()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize Whisper model (will be loaded in the thread)
        self.model = None
    
    def start(self):
        """Start the voice processor."""
        if self.active:
            return
        
        self.active = True
        
        # Start processing thread
        self.thread = threading.Thread(target=self._processing_thread)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started voice processor for session {self.session_id}")
    
    def stop(self):
        """Stop the voice processor."""
        if not self.active:
            return
        
        self.active = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        # Clean up PyAudio
        self.p.terminate()
        self.p = None
        
        logger.info(f"Stopped voice processor for session {self.session_id}")
    
    def _processing_thread(self):
        """Processing thread for voice recognition."""
        try:
            # Load Whisper model
            self.model = whisper.load_model("base")
            
            # Open audio stream
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            stream.start_stream()
            
            # Process audio chunks
            while self.active:
                try:
                    # Get audio chunk from queue with timeout
                    audio_data = self.audio_queue.get(timeout=0.5)
                    
                    # Save to temporary file
                    with wave.open("temp_voice.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)
                    
                    # Transcribe
                    result = self.model.transcribe("temp_voice.wav")
                    text = result["text"].strip()
                    
                    # Call callback if text is not empty
                    if text:
                        self.callback(self.session_id, text)
                    
                    # Clean up
                    if os.path.exists("temp_voice.wav"):
                        os.remove("temp_voice.wav")
                    
                except queue.Empty:
                    # No audio data, continue
                    continue
                
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error in voice processing thread: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status_flags):
        """Callback for audio stream."""
        if self.active:
            # Add audio data to queue
            self.audio_queue.put(in_data)
        
        return (in_data, pyaudio.paContinue)

class ScreenAnalyzer:
    """
    Screen analyzer for understanding user's visual context.
    
    This component enables the revolutionary screen understanding
    capabilities of the AI assistant.
    """
    
    def __init__(self, vision_model, ocr_engine=None):
        """
        Initialize the screen analyzer.
        
        Args:
            vision_model: Vision model for image analysis
            ocr_engine: OCR engine for text extraction
        """
        self.vision_model = vision_model
        self.ocr_engine = ocr_engine or pytesseract
        
        # Initialize element classifiers
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize UI element classifiers."""
        # In a real implementation, this would load trained models
        # for UI element classification
        pass
    
    def analyze_screen(self, image) -> Dict[str, Any]:
        """
        Analyze a screen capture.
        
        Args:
            image: Screen capture image
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "elements": [],
            "text": [],
            "recognized_apps": [],
            "active_window": None
        }
        
        try:
            # 1. Convert to RGB for processing
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Extract text using OCR
            text = self.ocr_engine.image_to_string(Image.fromarray(rgb_image))
            if text:
                results["text"] = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 3. Detect UI elements
            elements = self._detect_ui_elements(rgb_image)
            results["elements"] = elements
            
            # 4. Recognize applications
            apps = self._recognize_applications(rgb_image)
            results["recognized_apps"] = apps
            
            # 5. Determine active window
            active_window = self._determine_active_window(elements)
            if active_window:
                results["active_window"] = active_window
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing screen: {e}")
            return results
    
    def _detect_ui_elements(self, image) -> List[Dict[str, Any]]:
        """Detect UI elements in the image."""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract element image
                element_image = image[y:y+h, x:x+w]
                
                # Classify element
                element_type = self._classify_element(element_image)
                
                # Add to elements list
                elements.append({
                    "id": f"element_{i}",
                    "type": element_type,
                    "bounds": {"x": x, "y": y, "width": w, "height": h}
                })
            
            return elements
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
            return elements
    
    def _classify_element(self, element_image) -> str:
        """Classify a UI element."""
        # This is a simplified version - in a real implementation, 
        # we would use a trained model for more accurate classification
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(element_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple heuristic classification
        if aspect_ratio > 5:
            return "menu_bar"
        elif aspect_ratio > 3:
            return "toolbar"
        elif aspect_ratio < 0.5:
            return "sidebar"
        elif w > 300 and h > 200:
            if avg_brightness > 200:
                return "dialog"
            else:
                return "window"
        elif w < 100 and h < 50:
            if avg_brightness > 200:
                return "button"
            else:
                return "icon"
        elif w > 200 and h < 50:
            return "text_field"
        else:
            return "panel"
    
    def _recognize_applications(self, image) -> List[Dict[str, Any]]:
        """Recognize applications in the image."""
        apps = []
        
        try:
            # Convert to PIL image for the vision model
            pil_image = Image.fromarray(image)
            
            # Get predictions from vision model
            predictions = self.vision_model(pil_image)
            
            # Process predictions
            for prediction in predictions:
                if prediction["score"] > 0.7:  # Confidence threshold
                    apps.append({
                        "name": prediction["label"],
                        "confidence": prediction["score"]
                    })
            
            return apps
            
        except Exception as e:
            logger.error(f"Error recognizing applications: {e}")
            return apps
    
    def _determine_active_window(self, elements: List[Dict[str, Any]]) -> Optional[str]:
        """Determine the active window from detected elements."""
        if not elements:
            return None
        
        # Heuristic: The largest element near the top of the screen is likely the active window
        window_elements = [e for e in elements if e["type"] in ["window", "dialog"]]
        
        if not window_elements:
            return None
        
        # Sort by y-coordinate (top to bottom)
        top_elements = sorted(window_elements, key=lambda e: e["bounds"]["y"])[:5]
        
        # Get the largest element
        largest_element = max(top_elements, key=lambda e: e["bounds"]["width"] * e["bounds"]["height"])
        
        return largest_element["id"]
class KnowledgeManager:
    """
    Knowledge manager for storing and retrieving information.
    
    This component enables the revolutionary knowledge capabilities
    of the AI assistant, allowing it to learn and adapt over time.
    """
    
    def __init__(self, embedding_model, storage_dir: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            embedding_model: Model for generating embeddings
            storage_dir: Directory for storing knowledge
        """
        self.embedding_model = embedding_model
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base
        self.knowledge_texts = []
        self.knowledge_vectors = []
        self.knowledge_metadata = []
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        # Load existing knowledge
        await self.load_knowledge()
    
    async def load_knowledge(self):
        """Load knowledge from storage."""
        # Clear existing knowledge
        self.knowledge_texts = []
        self.knowledge_vectors = []
        self.knowledge_metadata = []
        
        # Load knowledge index if it exists
        index_path = self.storage_dir / "knowledge_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                
                # Load knowledge items
                for item in index:
                    item_path = self.storage_dir / item["file"]
                    if item_path.exists():
                        with open(item_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        
                        # Load vector
                        vector_path = self.storage_dir / item["vector_file"]
                        if vector_path.exists():
                            vector = np.load(vector_path)
                            
                            # Add to knowledge base
                            self.knowledge_texts.append(text)
                            self.knowledge_vectors.append(vector)
                            self.knowledge_metadata.append(item["metadata"])
            
            except Exception as e:
                logger.error(f"Error loading knowledge index: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_texts)} knowledge items")
    
    async def add_knowledge(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add knowledge to the knowledge base.
        
        Args:
            text: Knowledge text
            metadata: Metadata for the knowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_model(text)
            vector = np.mean(embedding[0], axis=0)
            
            # Generate unique ID
            knowledge_id = str(uuid.uuid4())
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {}
            
            metadata["id"] = knowledge_id
            metadata["timestamp"] = datetime.now().isoformat()
            
            # Save text
            text_file = f"knowledge_{knowledge_id}.txt"
            with open(self.storage_dir / text_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Save vector
            vector_file = f"vector_{knowledge_id}.npy"
            np.save(self.storage_dir / vector_file, vector)
            
            # Add to knowledge base
            self.knowledge_texts.append(text)
            self.knowledge_vectors.append(vector)
            self.knowledge_metadata.append(metadata)
            
            # Update index
            await self._update_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        if not self.knowledge_vectors:
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model(query)
            query_vector = np.mean(query_embedding[0], axis=0)
            
            # Calculate similarity with all knowledge items
            similarities = []
            for i, vector in enumerate(self.knowledge_vectors):
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k results
            results = []
            for i, similarity in similarities[:top_k]:
                if similarity > 0.5:  # Similarity threshold
                    results.append({
                        "text": self.knowledge_texts[i],
                        "similarity": float(similarity),
                        "metadata": self.knowledge_metadata[i]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def _update_index(self):
        """Update the knowledge index."""
        try:
            # Create index
            index = []
            for i, metadata in enumerate(self.knowledge_metadata):
                knowledge_id = metadata.get("id", str(uuid.uuid4()))
                index.append({
                    "id": knowledge_id,
                    "file": f"knowledge_{knowledge_id}.txt",
                    "vector_file": f"vector_{knowledge_id}.npy",
                    "metadata": metadata
                })
            
            # Save index
            with open(self.storage_dir / "knowledge_index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating knowledge index: {e}")

class TaskAutomator:
    """
    Task automator for automating user tasks.
    
    This component enables the revolutionary automation capabilities
    of the AI assistant, allowing it to perform tasks on behalf of the user.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the task automator.
        
        Args:
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus
        
        # Automation capabilities
        self.capabilities = {
            "open_application": self._open_application,
            "close_application": self._close_application,
            "click": self._click,
            "type_text": self._type_text,
            "copy_text": self._copy_text,
            "paste_text": self._paste_text,
            "save_file": self._save_file,
            "open_file": self._open_file,
            "search": self._search,
            "navigate_to": self._navigate_to
        }
    
    async def execute_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task_type: Type of task to execute
            params: Task parameters
            
        Returns:
            Task result
        """
        if task_type not in self.capabilities:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
        
        try:
            # Execute the task
            result = await self.capabilities[task_type](params)
            
            # Publish task completed event
            await self.event_bus.publish("task.completed", {
                "type": "task_completed",
                "task": task_type,
                "params": params,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error executing task {task_type}: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _open_application(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open an application."""
        app_name = params.get("name")
        if not app_name:
            raise ValueError("Application name is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to open the application
        
        return {
            "status": "opened",
            "app_name": app_name
        }
    
    async def _close_application(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Close an application."""
        app_name = params.get("name")
        if not app_name:
            raise ValueError("Application name is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to close the application
        
        return {
            "status": "closed",
            "app_name": app_name
        }
    
    async def _click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Click at a position or on an element."""
        # Check if we have coordinates
        if "x" in params and "y" in params:
            x = params.get("x")
            y = params.get("y")
            
            # In a real implementation, this would use platform-specific APIs
            # to click at the specified coordinates
            
            return {
                "status": "clicked",
                "position": {"x": x, "y": y}
            }
        
        # Check if we have an element
        elif "element_id" in params:
            element_id = params.get("element_id")
            
            # In a real implementation, this would use platform-specific APIs
            # to click on the specified element
            
            return {
                "status": "clicked",
                "element_id": element_id
            }
        
        else:
            raise ValueError("Either coordinates (x, y) or element_id is required")
    
    async def _type_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Type text."""
        text = params.get("text")
        if not text:
            raise ValueError("Text is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to type the specified text
        
        return {
            "status": "typed",
            "text": text
        }
    
    async def _copy_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Copy text to clipboard."""
        text = params.get("text")
        if not text:
            raise ValueError("Text is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to copy the specified text to the clipboard
        
        return {
            "status": "copied",
            "text": text
        }
    
    async def _paste_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Paste text from clipboard."""
        # In a real implementation, this would use platform-specific APIs
        # to paste text from the clipboard
        
        return {
            "status": "pasted"
        }
    
    async def _save_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Save a file."""
        path = params.get("path")
        content = params.get("content")
        
        if not path:
            raise ValueError("File path is required")
        
        if content is None:
            raise ValueError("File content is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to save the file
        
        return {
            "status": "saved",
            "path": path
        }
    
    async def _open_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a file."""
        path = params.get("path")
        if not path:
            raise ValueError("File path is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to open the file
        
        return {
            "status": "opened",
            "path": path
        }
    
    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a search."""
        query = params.get("query")
        if not query:
            raise ValueError("Search query is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to perform the search
        
        return {
            "status": "searched",
            "query": query
        }
    
    async def _navigate_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a URL or location."""
        url = params.get("url")
        if not url:
            raise ValueError("URL is required")
        
        # In a real implementation, this would use platform-specific APIs
        # to navigate to the URL
        
        return {
            "status": "navigated",
            "url": url
        }

class UserModelManager:
    """
    User model manager for tracking user preferences and behavior.
    
    This component enables the revolutionary personalization capabilities
    of the AI assistant, allowing it to adapt to each user's unique needs.
    """
    
    def __init__(self, storage_dir: str = "./user_models"):
        """
        Initialize the user model manager.
        
        Args:
            storage_dir: Directory for storing user models
        """
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # User models
        self.user_models = {}
    
    async def initialize(self):
        """Initialize the user model manager."""
        # Load existing user models
        await self.load_user_models()
    
    async def load_user_models(self):
        """Load user models from storage."""
        # Clear existing models
        self.user_models = {}
        
        # Load user models
        for file_path in self.storage_dir.glob("user_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    user_model = json.load(f)
                
                user_id = user_model.get("user_id")
                if user_id:
                    self.user_models[user_id] = user_model
            
            except Exception as e:
                logger.error(f"Error loading user model {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.user_models)} user models")
    
    async def get_user_model(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model
        """
        # Return existing model if available
        if user_id in self.user_models:
            return self.user_models[user_id]
        
        # Create new model
        user_model = {
            "user_id": user_id,
            "preferences": {
                "voice_enabled": True,
                "screen_capture_interval": 1.0,
                "proactive_assistance": True,
                "preferred_communication_style": "conversational"
            },
            "behavior_patterns": {},
            "interaction_history": [],
            "created_at": datetime.now().isoformat(),
            "
           "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save the model
        self.user_models[user_id] = user_model
        await self._save_user_model(user_id)
        
        return user_model
    
    async def update_user_preference(self, user_id: str, preference: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            preference: Preference name
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        # Get user model
        user_model = await self.get_user_model(user_id)
        
        # Update preference
        user_model["preferences"][preference] = value
        user_model["updated_at"] = datetime.now().isoformat()
        
        # Save the model
        return await self._save_user_model(user_id)
    
    async def record_interaction(self, user_id: str, interaction_type: str, details: Dict[str, Any]) -> bool:
        """
        Record a user interaction.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            details: Interaction details
            
        Returns:
            True if successful, False otherwise
        """
        # Get user model
        user_model = await self.get_user_model(user_id)
        
        # Add interaction to history
        interaction = {
            "type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        user_model["interaction_history"].append(interaction)
        
        # Limit history size
        if len(user_model["interaction_history"]) > 1000:
            user_model["interaction_history"] = user_model["interaction_history"][-1000:]
        
        # Update behavior patterns
        await self._update_behavior_patterns(user_model)
        
        # Save the model
        return await self._save_user_model(user_id)
    
    async def _update_behavior_patterns(self, user_model: Dict[str, Any]):
        """Update behavior patterns based on interaction history."""
        # Extract interaction history
        interactions = user_model["interaction_history"]
        
        # Skip if not enough interactions
        if len(interactions) < 10:
            return
        
        # Update patterns
        patterns = {}
        
        # 1. Preferred interaction times
        hour_counts = {}
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction["timestamp"])
            hour = timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours
        total_interactions = len(interactions)
        peak_hours = [hour for hour, count in hour_counts.items() if count > total_interactions * 0.1]
        patterns["preferred_hours"] = peak_hours
        
        # 2. Preferred interaction types
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction["type"]
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        # Find preferred types
        preferred_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        patterns["preferred_types"] = [t[0] for t in preferred_types]
        
        # 3. Response patterns
        response_times = []
        for i in range(1, len(interactions)):
            if interactions[i]["type"] == "assistant_message" and interactions[i-1]["type"] == "user_message":
                prev_time = datetime.fromisoformat(interactions[i-1]["timestamp"])
                curr_time = datetime.fromisoformat(interactions[i]["timestamp"])
                response_time = (curr_time - prev_time).total_seconds()
                response_times.append(response_time)
        
        if response_times:
            patterns["avg_response_time"] = sum(response_times) / len(response_times)
        
        # 4. Common queries
        queries = [
            interaction["details"].get("content", "")
            for interaction in interactions
            if interaction["type"] == "user_message"
        ]
        
        # Extract common words
        word_counts = {}
        for query in queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns["common_words"] = [w[0] for w in common_words]
        
        # Update user model
        user_model["behavior_patterns"] = patterns
    
    async def _save_user_model(self, user_id: str) -> bool:
        """Save a user model to storage."""
        if user_id not in self.user_models:
            return False
        
        try:
            # Get user model
            user_model = self.user_models[user_id]
            
            # Save to file
            file_path = self.storage_dir / f"user_{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_model, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user model for {user_id}: {e}")
            return False

class PredictiveEngine:
    """
    Predictive engine for anticipating user needs.
    
    This component enables the revolutionary predictive capabilities
    of the AI assistant, allowing it to anticipate user needs before
    they are explicitly expressed.
    """
    
    def __init__(self, user_model_manager: UserModelManager, knowledge_manager: KnowledgeManager):
        """
        Initialize the predictive engine.
        
        Args:
            user_model_manager: User model manager
            knowledge_manager: Knowledge manager
        """
        self.user_model_manager = user_model_manager
        self.knowledge_manager = knowledge_manager
    
    async def predict_user_needs(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict user needs based on context and user model.
        
        Args:
            user_id: User ID
            context: Current context
            
        Returns:
            List of predicted needs
        """
        # Get user model
        user_model = await self.user_model_manager.get_user_model(user_id)
        
        # Extract relevant information
        current_time = datetime.now()
        screen_context = context.get("screen_context", {})
        recent_messages = context.get("recent_messages", [])
        
        # Predictions
        predictions = []
        
        # 1. Time-based predictions
        time_predictions = await self._predict_time_based_needs(user_model, current_time)
        predictions.extend(time_predictions)
        
        # 2. Context-based predictions
        context_predictions = await self._predict_context_based_needs(user_model, screen_context)
        predictions.extend(context_predictions)
        
        # 3. Conversation-based predictions
        conversation_predictions = await self._predict_conversation_based_needs(user_model, recent_messages)
        predictions.extend(conversation_predictions)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predictions
    
    async def _predict_time_based_needs(self, user_model: Dict[str, Any], current_time: datetime) -> List[Dict[str, Any]]:
        """Predict needs based on time patterns."""
        predictions = []
        
        # Get behavior patterns
        patterns = user_model.get("behavior_patterns", {})
        preferred_hours = patterns.get("preferred_hours", [])
        
        # Check if current hour is a preferred hour
        current_hour = current_time.hour
        if current_hour in preferred_hours:
            # Predict based on common activities during this hour
            hour_interactions = [
                interaction for interaction in user_model.get("interaction_history", [])
                if datetime.fromisoformat(interaction["timestamp"]).hour == current_hour
            ]
            
            if hour_interactions:
                # Count interaction types
                type_counts = {}
                for interaction in hour_interactions:
                    interaction_type = interaction["type"]
                    type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
                
                # Find most common type
                most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                # Add prediction
                predictions.append({
                    "type": "time_based",
                    "need": f"typical_{most_common_type}",
                    "description": f"User typically performs {most_common_type} at this time",
                    "confidence": 0.7
                })
        
        return predictions
    
    async def _predict_context_based_needs(self, user_model: Dict[str, Any], screen_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict needs based on screen context."""
        predictions = []
        
        # Check for recognized applications
        recognized_apps = screen_context.get("recognized_apps", [])
        if recognized_apps:
            app_name = recognized_apps[0]["name"].lower()
            
            # Predict based on application
            if "code" in app_name or "editor" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "coding_assistance",
                    "description": "User might need help with coding",
                    "confidence": 0.8
                })
            
            elif "terminal" in app_name or "command" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "command_assistance",
                    "description": "User might need help with terminal commands",
                    "confidence": 0.8
                })
            
            elif "browser" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "information_search",
                    "description": "User might need help finding information",
                    "confidence": 0.7
                })
            
            elif "document" in app_name or "word" in app_name:
                predictions.append({
                    "type": "context_based",
                    "need": "writing_assistance",
                    "description": "User might need help with writing",
                    "confidence": 0.7
                })
        
        # Check for error messages
        error_detected = False
        for text in screen_context.get("text", []):
            error_patterns = ["error", "exception", "failed", "warning", "invalid"]
            if any(pattern in text.lower() for pattern in error_patterns):
                error_detected = True
                break
        
        if error_detected:
            predictions.append({
                "type": "context_based",
                "need": "error_resolution",
                "description": "User might need help resolving an error",
                "confidence": 0.9
            })
        
        return predictions
    
    async def _predict_conversation_based_needs(self, user_model: Dict[str, Any], recent_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict needs based on recent conversation."""
        predictions = []
        
        # Skip if no recent messages
        if not recent_messages:
            return predictions
        
        # Extract user messages
        user_messages = [
            msg["content"] for msg in recent_messages
            if msg.get("role") == "user"
        ]
        
        # Skip if no user messages
        if not user_messages:
            return predictions
        
        # Analyze last user message
        last_message = user_messages[-1].lower()
        
        # Check for question patterns
        question_patterns = ["how", "what", "why", "when", "where", "can", "could", "would", "will", "?"]
        is_question = any(pattern in last_message for pattern in question_patterns)
        
        if is_question:
            # Search knowledge base for relevant information
            knowledge_results = await self.knowledge_manager.search_knowledge(last_message)
            
            if knowledge_results:
                predictions.append({
                    "type": "conversation_based",
                    "need": "knowledge_retrieval",
                    "description": "User is asking a question that might be answered from knowledge base",
                    "confidence": 0.8,
                    "knowledge": knowledge_results[0]
                })
        
        # Check for request patterns
        request_patterns = ["can you", "could you", "please", "help me", "i need", "show me"]
        is_request = any(pattern in last_message for pattern in request_patterns)
        
        if is_request:
            predictions.append({
                "type": "conversation_based",
                "need": "task_assistance",
                "description": "User is requesting assistance with a task",
                "confidence": 0.7
            })
        
        return predictions

class ContextWindow:
    """
    Context window for tracking conversation and context.
    
    This component enables the AI assistant to maintain context
    across interactions, providing a more coherent and personalized
    experience.
    """
    
    def __init__(self, user_id: str, session_id: str, max_messages: int = 100):
        """
        Initialize the context window.
        
        Args:
            user_id: User ID
            session_id: Session ID
            max_messages: Maximum number of messages to keep
        """
        self.user_id = user_id
        self.session_id = session_id
        self.max_messages = max_messages
        
        # Messages
        self.messages = []
        
        # Context
        self.screen_context = {}
        self.system_state = {}
        self.user_preferences = {}
    
    def add_message(self, message: Dict[str, Any]):
        """
        Add a message to the context window.
        
        Args:
            message: Message to add
        """
        # Add message
        self.messages.append(message)
        
        # Trim if necessary
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages.
        
        Args:
            count: Number of messages to get
            
        Returns:
            List of recent messages
        """
        return self.messages[-count:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.
        
        Returns:
            Context summary
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "recent_messages": self.get_recent_messages(5),
            "screen_context": self.screen_context,
            "system_state": self.system_state,
            "user_preferences": self.user_preferences
        }

# Main entry point
def create_cognitive_collaboration_system(config: Dict[str, Any] = None) -> CognitiveCollaborationSystem:
    """
    Create a cognitive collaboration system.
    
    Args:
        config: Configuration
   Create a cognitive collaboration system.
    
    Args:
        config: Configuration
        
    Returns:
        Cognitive collaboration system
    """
    # Default configuration
    default_config = {
        "model_path": "./models",
        "knowledge_dir": "./knowledge",
        "user_models_dir": "./user_models",
        "log_level": "INFO"
    }
    
    # Merge configurations
    if config is None:
        config = {}
    
    merged_config = {**default_config, **config}
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, merged_config["log_level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create event bus
    event_bus = EventBus()
    
    # Create system
    system = CognitiveCollaborationSystem(
        model_path=merged_config["model_path"],
        event_bus=event_bus
    )
    
    return system

# Example usage
if __name__ == "__main__":
    # Create system
    system = create_cognitive_collaboration_system()
    
    # Initialize system
    asyncio.run(system.initialize())
    
    # Start system
    asyncio.run(system.start())
    
    try:
        # Run forever
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        # Stop system
        asyncio.run(system.stop())
"""
Event bus for communication between components.

This module provides a simple event bus implementation for asynchronous
communication between components of the AI assistant.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class EventBus:
    """
    Event bus for communication between components.
    
    This class provides a simple publish-subscribe mechanism for
    asynchronous communication between components.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.event_history = {}
        self.max_history = 100
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        # Add to history
        if event_type not in self.event_history:
            self.event_history[event_type] = []
        
        self.event_history[event_type].append(event_data)
        
        # Trim history if necessary
        if len(self.event_history[event_type]) > self.max_history:
            self.event_history[event_type] = self.event_history[event_type][-self.max_history:]
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event subscriber for {event_type}: {e}")
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        """
        Subscribe to an event.
        
        Args:
            event_type: Type of event
            callback: Callback function
            
        Returns:
            Unsubscribe function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        
        # Return unsubscribe function
        def unsubscribe():
            if event_type in self.subscribers and callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
        
        return unsubscribe
    
    def get_recent_events(self, event_type: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events of a specific type.
        
        Args:
            event_type: Type of event
            count: Number of events to get
            
        Returns:
            List of recent events
        """
        if event_type not in self.event_history:
            return []
        
        return self.event_history[event_type][-count:]
"""
AI Assistant package for Edge-Native UMaaS.

This package provides the AI assistant capabilities for the Edge-Native
Universal Multimodal Assistant as a Service (UMaaS) platform.
"""

from .cognitive_collaboration_system import CognitiveCollaborationSystem, create_cognitive_collaboration_system
from .event_bus import EventBus

__all__ = [
    'CognitiveCollaborationSystem',
    'create_cognitive_collaboration_system',
    'EventBus'
]
#!/usr/bin/env python3
"""
Command-line interface for Edge-Native UMaaS.

This module provides a command-line interface for interacting with
the Edge-Native Universal Multimodal Assistant as a Service (UMaaS) platform.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any

from edgenative_umaas.ai_assistant import create_cognitive_collaboration_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class UMaaSCLI:
    """Command-line interface for UMaaS."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.system = None
        self.session_id = None
        self.user_id = os.environ.get("USER", "default_user")
    
    async def initialize(self, config_path: str = None):
        """
        Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Create system
        self.system = create_cognitive_collaboration_system(config)
        
        # Initialize system
        await self.system.initialize()
        
        # Start system
        await self.system.start()
        
        # Subscribe to assistant messages
        self.system.event_bus.subscribe("assistant.message", self._handle_assistant_message)
        
        # Start session
        self.session_id = await self.system.start_session(self.user_id)
        
        logger.info(f"Started session {self.session_id} for user {self.user_id}")
    
    async def _handle_assistant_message(self, event_data: Dict[str, Any]):
        """Handle assistant message event."""
        message = event_data.get("message", {})
        content = message.get("content", "")
        
        # Print message
        print(f"\nAssistant: {content}\n")
    
    async def process_command(self, command: str):
        """
        Process a command.
        
        Args:
            command: Command to process
        """
        if not self.system or not self.session_id:
            print("System not initialized")
            return
        
        if command.lower() in ["exit", "quit"]:
            # End session
            await self.system.end_session(self.session_id)
            
            # Stop system
            await self.system.stop()
            
            return False
        
        # Process command
        await self.system.process_user_input(self.session_id, command)
        
        return True
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("Edge-Native UMaaS CLI")
        print("Type 'exit' or 'quit' to exit")
        print()
        
        # Welcome message
        print("Assistant: Hello! I'm your AI assistant. How can I help you today?")
        
        # Main loop
        running = True
        while running:
            try:
                # Get user input
                user_input = input("\nYou: ")
                
                # Process command
                result = await self.process_command(user_input)
                if result is False:
                    running = False
            
            except KeyboardInterrupt:
                print("\nExiting...")
                
                # End session
                await self.system.end_session(self.session_id)
                
                # Stop system
                await self.system.stop()
                
                running = False
            
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(f"\nError: {e}")

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Edge-Native UMaaS CLI")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create CLI
    cli = UMaaSCLI()
    
    # Run
    try:
        # Initialize
        asyncio.run(cli.initialize(args.config))
        
        # Run in interactive mode
        asyncio.run(cli.interactive_mode())
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
from setuptools import setup, find_packages

setup(
    name="edgenative-umaas",
    version="0.1.0",
    description="Edge-Native Universal Multimodal Assistant as a Service",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "opencv-python",
        "pillow",
        "pytesseract",
        "pyaudio",
        "whisper",
    ],
    entry_points={
        "console_scripts": [
            "umaas-cli=edgenative_umaas.cli:main",
        ],
    },
    python_requires=">=3.8",
)
"""
Utility functions for the AI assistant.

This module provides utility functions used by various components
of the AI assistant.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False

def capture_screen() -> Optional[np.ndarray]:
    """
    Capture the screen.
    
    Returns:
        Screen image as numpy array, or None if failed
    """
    try:
        # This is a simplified implementation that works on most platforms
        # In a real implementation, we would use platform-specific APIs
        # for better performance and reliability
        
        # Try to use mss for fast screen capture
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # Primary monitor
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except ImportError:
            pass
        
        # Fall back to PIL and ImageGrab
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return np.array(img)
        except ImportError:
            pass
        
        logger.warning("No screen capture method available")
        return None
        
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return None

def save_image(image: np.ndarray, directory: str, prefix: str = "image") -> Optional[str]:
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array
        directory: Directory to save to
        prefix: Filename prefix
        
    Returns:
        Path to saved image, or None if failed
    """
    try:
        # Ensure directory exists
        if not ensure_directory(directory):
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
    """
    Crop an image.
    
    Args:
        image: Image as numpy array
        x: X coordinate
        y: Y coordinate
        width: Width
        height: Height
        
    Returns:
        Cropped image, or None if failed
    """
    try:
        # Check bounds
        img_height, img_width = image.shape[:2]
        
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            logger.warning(f"Crop region ({x}, {y}, {width}, {height}) out of bounds ({img_width}, {img_height})")
            return None
        
        # Crop image
        return image[y:y+height, x:x+width]
        
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return None

def resize_image(image: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """
    Resize an image.
    
    Args:
        image: Image as numpy array
        width: Width
        height: Height
        
    Returns:
        Resized image, or None if failed
    """
    try:
        return cv2.resize(image, (width, height))
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

def extract_text_from_image(image: np.ndarray, lang: str = "eng") -> List[str]:
    """
    Extract text from an image using OCR.
    
    Args:
        image: Image as numpy array
        lang: Language code
        
    Returns:
        List of extracted text lines
    """
    try:
        # Convert to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Extract text using pytesseract
        import pytesseract
        text = pytesseract.image_to_string(pil_image, lang=lang)
        
        # Split into lines and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return lines
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return []

def measure_execution_time(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper

def format_timestamp(timestamp: float) -> str:
    """
    Format a timestamp.
    
    Args:
        timestamp: Timestamp in seconds
        
    Returns:
        Formatted timestamp
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    try:
        # Split into words
        words = text.lower().split()
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "that", "this", "these", "those", "it", "they", 
                        "he", "she", "we", "you", "i", "me", "my", "your", "our", "their"}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    try:
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def parse_command(text: str) -> Tuple[str, str]:
    """
    Parse a command.
    
    Args:
        text: Command text
        
    Returns:
        Tuple of (command, arguments)
    """
    parts = text.strip().split(maxsplit=1)
    
    if len(parts) == 0:
        return "", ""
    
    if len(parts) == 1:
        return parts[0].lower(), ""
    
    return parts[0].lower(), parts[1]

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    
    hours = minutes / 60
    return f"{hours:.1f} hours"

def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.1f} GB"
"""
Model management for the AI assistant.

This module provides functionality for loading and managing AI models
used by the assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for loading and managing AI models.
    
    This class provides functionality for loading and managing AI models
    used by the assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Loaded models
        self.models = {}
        self.tokenizers = {}
    
    async def load_model(self, model_name: str, model_type: str = "causal_lm", device: str = None) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """
        Load a model.
        
        Args:
            model_name: Model name or path
            model_type: Model type (causal_lm or seq2seq_lm)
            device: Device to load model on (cpu, cuda, or None for auto)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if model is already loaded
        if model_name in self.models and model_name in self.tokenizers:
            return self.models[model_name], self.tokenizers[model_name]
        
        try:
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} on {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_name)
            elif model_type == "seq2seq_lm":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Move model to device
            model = model.to(device)
            
            # Save model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, None
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.models:
            return False
        
        try:
            # Remove model and tokenizer
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def generate_text(self, model_name: str, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using a model.
        
        Args:
            model_name: Model name
            prompt: Prompt text
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated text
        """
        if model_name not in self.models or model_name not in self.tokenizers:
            # Try to load the model
            model, tokenizer = await self.load_model(model_name)
            if model is None or tokenizer is None:
                return ""
        else:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate text
"""
Configuration management for the AI assistant.

This module provides functionality for loading and managing configuration
for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the AI assistant.
    
    This class provides functionality for loading and managing configuration
    for the AI assistant.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        
        # Default configuration
        self.default_config = {
            "model_path": "./models",
            "knowledge_dir": "./knowledge",
            "user_models_dir": "./user_models",
            "log_level": "INFO",
            "voice": {
                "enabled": True,
                "model": "base",
                "language": "en"
            },
            "screen_capture": {
                "enabled": True,
                "interval": 1.0
            },
            "models": {
                "assistant": "gpt2",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "vision": "facebook/detr-resnet-50"
            },
            "system": {
                "max_sessions": 10,
                "session_timeout": 3600,
                "max_messages": 100
            }
        }
    
    async def load_config(self) -> Dict[str, Any]:
        """
        Load configuration.
        
        Returns:
            Configuration dictionary
        """
        # Start with default configuration
        self.config = self.default_config.copy()
        
        # Load configuration from file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                
                # Merge configurations
                self._merge_config(self.config, file_config)
                
                logger.info(f"Loaded configuration from {self.config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")
        
        return self.config
    
    async def save_config(self) -> bool:
        """
        Save configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        # Split key into parts
        parts = key.split(".")
        
        # Navigate through configuration
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split key into parts
            parts = key.split(".")
            
            # Navigate through configuration
            config = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set value
            config[parts[-1]] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value for {key}: {e}")
            return False
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Merge two configuration dictionaries.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._merge_config(target[key], value)
            else:
                # Set value
                target[key] = value
   def _cleanup_sessions(self):
        """Clean up expired sessions."""
        # Find expired sessions
        expired_session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        
        # End expired sessions
        for session_id in expired_session_ids:
            self.end_session(session_id)
        
        if expired_session_ids:
            logger.info(f"Cleaned up {len(expired_session_ids)} expired sessions")
"""
Voice processing for the AI assistant.

This module provides functionality for speech recognition and synthesis
for the AI assistant.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for the AI assistant.
    
    This class provides functionality for speech recognition and synthesis
    for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the voice processor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Speech recognition model
        self.stt_model = None
        
        # Speech synthesis model
        self.tts_model = None
    
    async def initialize(self):
        """Initialize the voice processor."""
        try:
            # Initialize speech recognition
            await self._initialize_stt()
            
            # Initialize speech synthesis
            await self._initialize_tts()
            
        except Exception as e:
            logger.error(f"Error initializing voice processor: {e}")
    
    async def _initialize_stt(self):
        """Initialize speech recognition."""
        try:
            # Try to import whisper
            import whisper
            
            # Load model
            model_name = "base"
            logger.info(f"Loading Whisper model: {model_name}")
            self.stt_model = whisper.load_model(model_name)
            
            logger.info("Speech recognition initialized")
            
        except ImportError:
            logger.warning("Whisper not available, speech recognition disabled")
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    async def _initialize_tts(self):
        """Initialize speech synthesis."""
        try:
            # Try to import TTS
            from TTS.api import TTS
            
            # Load model
            logger.info("Loading TTS model")
            self.tts_model = TTS(gpu=torch.cuda.is_available())
            
            logger.info("Speech synthesis initialized")
            
        except ImportError:
            logger.warning("TTS not available, speech synthesis disabled")
        except Exception as e:
            logger.error(f"Error initializing speech synthesis: {e}")
    
    async def recognize_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            Recognition result
        """
        if self.stt_model is None:
            return {
                "success": False,
                "error": "Speech recognition not available"
            }
        
        try:
            # Recognize speech
            result = self.stt_model.transcribe(audio_data)
            
            return {
                "success": True,
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"]
            }
            
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def synthesize_speech(self, text: str, language: str = "en") -> Tuple[Optional[np.ndarray], int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        if self.tts_model is None:
            return None, 0
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize speech
            self.tts_model.tts_to_file(
                text=text,
                file_path=temp_path,
                language=language
            )
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None, 0
    
    async def record_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], int]:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        try:
            # Try to import pyaudio
            import pyaudio
            import wave
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            # Record audio
            logger.info(f"Recording audio for {duration} seconds")
            frames = []
            for _ in range(0, int(sample_rate / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to file
            wf = wave.open(temp_path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(temp_path)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except ImportError:
            logger.warning("PyAudio not available, audio recording disabled")
            return None, 0
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None, 0
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Play audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to import pyaudio
            import pyaudio
            import wave
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to file
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Open wave file
            wf = wave.open(temp_path, "rb")
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Play audio
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Remove temporary file
            os.unlink(temp_path)
            
            return True
            
        except ImportError:
            logger.warning("PyAudio not available, audio playback disabled")
            return False
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
"""
Vision processing for the AI assistant.

This module provides functionality for computer vision tasks
for the AI assistant.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import capture_screen, extract_text_from_image

# Configure logging
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Vision processor for the AI assistant.
    
    This class provides functionality for computer vision tasks
    for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the vision processor.
        
        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Object detection model
        self.detection_model = None
        
        # Image classification model
        self.classification_model = None
    
    async def initialize(self):
        """Initialize the vision processor."""
        try:
            # Initialize object detection
            await self._initialize_detection()
            
            # Initialize image classification
            await self._initialize_classification()
            
        except Exception as e:
            logger.error(f"Error initializing vision processor: {e}")
    
    async def _initialize_detection(self):
        """Initialize object detection."""
        try:
            # Try to import transformers
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            # Load model
            model_name = "facebook/detr-resnet-50"
            logger.info(f"Loading object detection model: {model_name}")
            
            self.detection_processor = DetrImageProcessor.from_pretrained(model_name)
            self.detection_model = DetrForObjectDetection.from_pretrained(model_name)
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detection_model = self.detection_model.to(device)
            
            logger.info("Object detection initialized")
            
        except ImportError:
            logger.warning("Transformers not available, object detection disabled")
        except Exception as e:
            logger.error(f"Error initializing object detection: {e}")
    
    async def _initialize_classification(self):
        """Initialize image classification."""
        try:
            # Try to import transformers
            from transformers import ViTForImageClassification, ViTImageProcessor
            
            # Load model
            model_name = "google/vit-base-patch16-224"
            logger.info(f"Loading image classification model: {model_name}")
            
            self.classification_processor = ViTImageProcessor.from_pretrained(model_name)
            self.classification_model = ViTForImageClassification.from_pretrained(model_name)
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classification_model = self.classification_model.to(device)
            
            logger.info("Image classification initialized")
            
        except ImportError:
            logger.warning("Transformers not available, image classification disabled")
        except Exception as e:
            logger.error(f"Error initializing image classification: {e}")
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Analysis result
        """
        # Capture screen
        screen = capture_screen()
        if screen is None:
            return {
                "success": False,
                "error": "Failed to capture screen"
            }
        
        # Analyze screen
        result = await self.analyze_image(screen)
        
        # Add screen capture
        result["screen"] = screen
        
        return result
    
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Analysis result
        """
        result = {
            "success": True,
            "text": [],
            "objects": [],
            "classification": []
        }
        
        try:
            # Extract text
            text = extract_text_from_image(image)
            result["text"] = text
            
            # Detect objects
            if self.detection_model is not None:
                objects = await self._detect_objects(image)
                result["objects"] = objects
            
            # Classify image
            if self.classification_model is not None:
                classification = await self._classify_image(image)
                result["classification"] = classification
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected objects
        """
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare image for model
            inputs = self.detection_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.detection_model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.tensor([pil_image.size[::-1]])
            target_sizes = target_sizes.to(self.detection_model.device)
            results = self.detection_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )[0]
            
            # Extract results
            objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                objects.append({
                    "label": self.detection_model
                   "label": self.detection_model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": {
                        "x": box[0].item(),
                        "y": box[1].item(),
                        "width": box[2].item() - box[0].item(),
                        "height": box[3].item() - box[1].item()
                    }
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    async def _classify_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of classifications
        """
        try:
            # Convert to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare image for model
            inputs = self.classification_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.classification_model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Extract results
            classifications = []
            for prob, idx in zip(top5_prob, top5_indices):
                classifications.append({
                    "label": self.classification_model.config.id2label[idx.item()],
                    "score": prob.item()
                })
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return []
    
    async def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Extract results
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
            
            return face_list
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    async def recognize_app_windows(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize application windows in a screen image.
        
        Args:
            image: Screen image as numpy array
            
        Returns:
            List of recognized application windows
        """
        try:
            # This is a simplified implementation that uses basic image processing
            # In a real implementation, we would use platform-specific APIs
            # to get accurate window information
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            windows = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 100 or h < 100:
                    continue
                
                # Extract window region
                window_region = image[y:y+h, x:x+w]
                
                # Extract text from window title bar
                title_bar_region = window_region[:30, :]
                title_text = extract_text_from_image(title_bar_region)
                
                # Add window
                windows.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "title": title_text[0] if title_text else "Unknown"
                })
            
            return windows
            
        except Exception as e:
            logger.error(f"Error recognizing app windows: {e}")
            return []
    
    async def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected text regions
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            text_regions = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 10:
                    continue
                
                # Extract text region
                text_region = image[y:y+h, x:x+w]
                
                # Extract text
                text = extract_text_from_image(text_region)
                
                # Add text region
                if text:
                    text_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "text": text[0]
                    })
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    async def recognize_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize UI elements in a screen image.
        
        Args:
            image: Screen image as numpy array
            
        Returns:
            List of recognized UI elements
        """
        try:
            # This is a simplified implementation that uses basic image processing
            # In a real implementation, we would use more sophisticated techniques
            # or platform-specific APIs
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            ui_elements = []
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 10:
                    continue
                
                # Extract UI element region
                element_region = image[y:y+h, x:x+w]
                
                # Determine element type
                element_type = "unknown"
                
                # Check if it's a button
                if 20 <= w <= 200 and 20 <= h <= 50:
                    element_type = "button"
                
                # Check if it's a text field
                elif 100 <= w <= 400 and 20 <= h <= 40:
                    element_type = "text_field"
                
                # Check if it's a checkbox
                elif 10 <= w <= 30 and 10 <= h <= 30:
                    element_type = "checkbox"
                
                # Extract text
                text = extract_text_from_image(element_region)
                
                # Add UI element
                ui_elements.append({
                    "type": element_type,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "text": text[0] if text else ""
                })
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error recognizing UI elements: {e}")
            return []
"""
Knowledge management for the AI assistant.

This module provides functionality for managing knowledge
for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge manager for the AI assistant.
    
    This class provides functionality for managing knowledge
    for the AI assistant.
    """
    
    def __init__(self, knowledge_dir: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            knowledge_dir: Path to knowledge directory
        """
        self.knowledge_dir = Path(knowledge_dir)
        
        # Create knowledge directory if it doesn't exist
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base
        self.knowledge_base = {}
        
        # Embeddings
        self.embeddings = {}
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        try:
            # Load knowledge base
            await self.load_knowledge_base()
            
        except Exception as e:
            logger.error(f"Error initializing knowledge manager: {e}")
    
    async def load_knowledge_base(self) -> bool:
        """
        Load knowledge base from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear knowledge base
            self.knowledge_base = {}
            self.embeddings = {}
            
            # Load knowledge files
            for file_path in self.knowledge_dir.glob("*.json"):
                try:
                    # Load knowledge file
                    with open(file_path, "r", encoding="utf-8") as f:
                        knowledge = json.load(f)
                    
                    # Add to knowledge base
                    knowledge_id = file_path.stem
                    self.knowledge_base[knowledge_id] = knowledge
                    
                    # Load embedding if available
                    embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
                    if embedding_path.exists():
                        self.embeddings[knowledge_id] = np.load(embedding_path)
                    
                except Exception as e:
                    logger.error(f"Error loading knowledge file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    async def save_knowledge_base(self) -> bool:
        """
        Save knowledge base to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save knowledge files
            for knowledge_id, knowledge in self.knowledge_base.items():
                try:
                    # Save knowledge file
                    file_path = self.knowledge_dir / f"{knowledge_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(knowledge, f, indent=2)
                    
                    # Save embedding if available
                    if knowledge_id in self.embeddings:
                        embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
                        np.save(embedding_path, self.embeddings[knowledge_id])
                    
                except Exception as e:
                    logger.error(f"Error saving knowledge file {file_path}: {e}")
            
            logger.info(f"Saved {len(self.knowledge_base)} knowledge items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    async def add_knowledge(self, knowledge: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            embedding: Knowledge embedding
            
        Returns:
            Knowledge ID
        """
        try:
            # Generate knowledge ID
            import uuid
            knowledge_id = str(uuid.uuid4())
            
            # Add to knowledge base
            self.knowledge_base[knowledge_id] = knowledge
            
            # Add embedding if available
            if embedding is not None:
                self.embeddings[knowledge_id] = embedding
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        return self.knowledge_base.get(knowledge_id)
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            embedding: Updated knowledge embedding
            
        Returns:
            True if successful, False otherwise
        """
        if knowledge_id not in self.knowledge_base:
            return False
        
        try:
            # Update knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Update embedding if available
            if embedding is not None:
                self.embeddings[knowledge_id] = embedding
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge {knowledge_id}: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        
       Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        if knowledge_id not in self.knowledge_base:
            return False
        
        try:
            # Remove from knowledge base
            del self.knowledge_base[knowledge_id]
            
            # Remove embedding if available
            if knowledge_id in self.embeddings:
                del self.embeddings[knowledge_id]
            
            # Remove knowledge files
            file_path = self.knowledge_dir / f"{knowledge_id}.json"
            if file_path.exists():
                os.remove(file_path)
            
            embedding_path = self.knowledge_dir / f"{knowledge_id}.npy"
            if embedding_path.exists():
                os.remove(embedding_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge {knowledge_id}: {e}")
            return False
    
    async def search_knowledge(self, query: str, embedding: Optional[np.ndarray] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            embedding: Query embedding
            max_results: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        try:
            results = []
            
            # Search by embedding if available
            if embedding is not None and self.embeddings:
                # Calculate similarity scores
                scores = {}
                for knowledge_id, knowledge_embedding in self.embeddings.items():
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, knowledge_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(knowledge_embedding)
                    )
                    scores[knowledge_id] = similarity
                
                # Sort by similarity
                sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
                
                # Add top results
                for knowledge_id in sorted_ids[:max_results]:
                    knowledge = self.knowledge_base[knowledge_id]
                    results.append({
                        "id": knowledge_id,
                        "knowledge": knowledge,
                        "score": float(scores[knowledge_id])
                    })
            
            # Search by text if no results or no embedding
            if not results:
                # Convert query to lowercase
                query_lower = query.lower()
                
                # Search knowledge base
                for knowledge_id, knowledge in self.knowledge_base.items():
                    # Check if query matches knowledge
                    score = 0.0
                    
                    # Check title
                    if "title" in knowledge and query_lower in knowledge["title"].lower():
                        score += 0.8
                    
                    # Check content
                    if "content" in knowledge and query_lower in knowledge["content"].lower():
                        score += 0.5
                    
                    # Check tags
                    if "tags" in knowledge:
                        for tag in knowledge["tags"]:
                            if query_lower in tag.lower():
                                score += 0.3
                    
                    # Add to results if score is positive
                    if score > 0:
                        results.append({
                            "id": knowledge_id,
                            "knowledge": knowledge,
                            "score": score
                        })
                
                # Sort by score
                results = sorted(results, key=lambda r: r["score"], reverse=True)
                
                # Limit results
                results = results[:max_results]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge from the knowledge base.
        
        Returns:
            List of all knowledge items
        """
        return [
            {
                "id": knowledge_id,
                "knowledge": knowledge
            }
            for knowledge_id, knowledge in self.knowledge_base.items()
        ]
    
    async def import_knowledge(self, file_path: str) -> int:
        """
        Import knowledge from a file.
        
        Args:
            file_path: Path to knowledge file
            
        Returns:
            Number of imported knowledge items
        """
        try:
            # Check file extension
            if file_path.endswith(".json"):
                # Load JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check if it's a list or a single item
                if isinstance(data, list):
                    # Import multiple knowledge items
                    count = 0
                    for item in data:
                        await self.add_knowledge(item)
                        count += 1
                    
                    return count
                else:
                    # Import single knowledge item
                    await self.add_knowledge(data)
                    return 1
            
            elif file_path.endswith(".txt"):
                # Load text file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create knowledge item
                knowledge = {
                    "title": os.path.basename(file_path),
                    "content": content,
                    "source": file_path,
                    "created_at": datetime.now().isoformat()
                }
                
                # Add knowledge
                await self.add_knowledge(knowledge)
                
                return 1
            
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return 0
            
        except Exception as e:
            logger.error(f"Error importing knowledge from {file_path}: {e}")
            return 0
    
    async def export_knowledge(self, file_path: str) -> bool:
        """
        Export knowledge to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all knowledge
            knowledge_items = [
                knowledge for knowledge_id, knowledge in self.knowledge_base.items()
            ]
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(knowledge_items, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge to {file_path}: {e}")
            return False
"""
User modeling for the AI assistant.

This module provides functionality for modeling user preferences
and behavior for the AI assistant.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class UserModel:
    """
    User model for the AI assistant.
    
    This class represents a model of a user's preferences and behavior.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the user model.
        
        Args:
            user_id: User ID
        """
        self.user_id = user_id
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        
        # User preferences
        self.preferences = {}
        
        # User interaction history
        self.interactions = []
        
        # User topics of interest
        self.topics = {}
    
    def update_preference(self, key: str, value: Any):
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.last_updated = datetime.now().isoformat()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        return self.preferences.get(key, default)
    
    def add_interaction(self, interaction_type: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a user interaction.
        
        Args:
            interaction_type: Type of interaction
            content: Interaction content
            metadata: Interaction metadata
        """
        # Create interaction
        interaction = {
            "type": interaction_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to interactions
        self.interactions.append(interaction)
        
        # Update last updated
        self.last_updated = datetime.now().isoformat()
        
        # Update topics of interest
        if interaction_type == "query":
            self._update_topics(content)
    
    def get_interactions(self, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        # Filter by type if specified
        if interaction_type:
            filtered = [i for i in self.interactions if i["type"] == interaction_type]
        else:
            filtered = self.interactions
        
        # Limit count if specified
        if count is not None:
            filtered = filtered[-count:]
        
        return filtered
    
    def get_top_topics(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top topics of interest.
        
        Args:
            count: Number of topics to get
            
        Returns:
            List of topics
        """
        # Sort topics by score
        sorted_topics = sorted(
            self.topics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to list of dictionaries
        return [
            {"topic": topic, "score": score}
            for topic, score in sorted_topics[:count]
        ]
    
    def _update_topics(self, content: str):
        """
        Update topics of interest based on content.
        
        Args:
            content: Content to analyze
        """
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Update topic scores
        for keyword in keywords:
            if keyword in self.topics:
                self.topics[keyword] += 1
            else:
                self.topics[keyword] = 1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Split into words
        words = text.lower().split()
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "like", 
                        "from", "of", "that", "this", "these", "those", "it", "they", 
                        "he", "she", "we", "you", "i", "me", "my", "your", "our", "their"}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:10]]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the user model to a dictionary.
        
        Returns:
            User model dictionary
        """
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "preferences": self.preferences,
            "interactions": self.interactions,
            "topics": self.topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserModel':
        """
        Create a user model from a dictionary.
        
        Args:
            data: User model dictionary
            
        Returns:
            User model
        """
        user_model = cls(data["user_id"])
        user_model.created_at = data["created_at"]
        user_model.last_updated = data["last_updated"]
        user_model.preferences = data["preferences"]
        user_model.interactions = data["interactions"]
        user_model.topics = data["topics"]
        
        return user_model

class UserModelManager:
    """
    User model manager for the AI assistant.
    
    This class manages user models for the AI assistant.
    """
    
    def __init__(self, user_models_dir: str = "./user_models"):
        """
        Initialize the user model manager.
        
        Args:
            user_models_dir: Path to user models directory
        """
        self.user_models_dir = Path(user_models_dir)
        
        # Create user models directory if it doesn't exist
        self.user_models_dir.mkdir(parents=True, exist_ok=True)
        
        # User models
        self.user_models = {}
    
    async def initialize(self):
        """Initialize the user model manager."""
        try:
            # Load user models
            await self.load_user_models()
            
        except Exception as e:
            logger.error(f"Error initializing user model manager: {e}")
    
    async def load_user_models(self) -> bool:
        """
        Load user models from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear user models
            self.user_models = {}
            
            # Load user model files
            for file_path in self.user_models_dir.glob("*.json"):
                try:
                    # Load user model file
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Create user model
                    user_model = UserModel.from_dict(data)
                    
                    # Add to user models
                    self.user_models[user_model.user_id] = user_model
                    
                except Exception as e:
                    logger.error(f"Error loading user model file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.user_models)} user models")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading user models: {e}")
            return False
    
    async def save_user_model(self, user_id: str) -> bool:
        """
        Save a user model to disk.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.user_models:
            return False
        
        try:
            # Get user model
            user_model = self.user_models[user_id]
            
            # Save user model file
            file_path = self.user_models_dir / f"{user_i
           # Save user model file
            file_path = self.user_models_dir / f"{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_model.to_dict(), f, indent=2)
            
            logger.info(f"Saved user model for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user model for user {user_id}: {e}")
            return False
    
    async def save_all_user_models(self) -> bool:
        """
        Save all user models to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save each user model
            for user_id in self.user_models:
                await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving all user models: {e}")
            return False
    
    async def get_user_model(self, user_id: str) -> UserModel:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model
        """
        # Check if user model exists
        if user_id not in self.user_models:
            # Create new user model
            self.user_models[user_id] = UserModel(user_id)
        
        return self.user_models[user_id]
    
    async def delete_user_model(self, user_id: str) -> bool:
        """
        Delete a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id not in self.user_models:
            return False
        
        try:
            # Remove from user models
            del self.user_models[user_id]
            
            # Remove user model file
            file_path = self.user_models_dir / f"{user_id}.json"
            if file_path.exists():
                os.remove(file_path)
            
            logger.info(f"Deleted user model for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user model for user {user_id}: {e}")
            return False
    
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List of user IDs
        """
        return list(self.user_models.keys())
    
    async def add_user_interaction(self, user_id: str, interaction_type: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a user interaction.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            content: Interaction content
            metadata: Interaction metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Add interaction
            user_model.add_interaction(interaction_type, content, metadata)
            
            # Save user model
            await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding interaction for user {user_id}: {e}")
            return False
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Update preference
            user_model.update_preference(key, value)
            
            # Save user model
            await self.save_user_model(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference for user {user_id}: {e}")
            return False
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get preference
            return user_model.get_preference(key, default)
            
        except Exception as e:
            logger.error(f"Error getting preference for user {user_id}: {e}")
            return default
    
    async def get_user_interactions(self, user_id: str, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get interactions
            return user_model.get_interactions(interaction_type, count)
            
        except Exception as e:
            logger.error(f"Error getting interactions for user {user_id}: {e}")
            return []
    
    async def get_user_top_topics(self, user_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get top topics of interest for a user.
        
        Args:
            user_id: User ID
            count: Number of topics to get
            
        Returns:
            List of topics
        """
        try:
            # Get user model
            user_model = await self.get_user_model(user_id)
            
            # Get top topics
            return user_model.get_top_topics(count)
            
        except Exception as e:
            logger.error(f"Error getting top topics for user {user_id}: {e}")
            return []
"""
Utility functions for the AI assistant.

This module provides utility functions for the AI assistant.
"""

import logging
import os
import platform
import subprocess
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

# Configure logging
logger = logging.getLogger(__name__)

def capture_screen() -> Optional[np.ndarray]:
    """
    Capture the screen.
    
    Returns:
        Screen image as numpy array, or None if failed
    """
    try:
        # Try to import mss
        import mss
        
        # Capture screen
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]
            
            # Capture screenshot
            sct_img = sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(sct_img)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
            
    except ImportError:
        logger.warning("mss not available, trying alternative method")
        
        # Try alternative method based on platform
        system = platform.system()
        
        if system == "Windows":
            # Use PIL on Windows
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab()
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Error capturing screen on Windows: {e}")
                return None
                
        elif system == "Darwin":  # macOS
            try:
                # Use screencapture on macOS
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Capture screen to temporary file
                subprocess.run(["screencapture", "-x", temp_path], check=True)
                
                # Read image
                img = cv2.imread(temp_path)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                return img
                
            except Exception as e:
                logger.error(f"Error capturing screen on macOS: {e}")
                return None
                
        elif system == "Linux":
            try:
                # Use scrot on Linux
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Capture screen to temporary file
                subprocess.run(["scrot", temp_path], check=True)
                
                # Read image
                img = cv2.imread(temp_path)
                
                # Remove temporary file
                os.unlink(temp_path)
                
                return img
                
            except Exception as e:
                logger.error(f"Error capturing screen on Linux: {e}")
                return None
        
        else:
            logger.error(f"Unsupported platform: {system}")
            return None
    
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        return None

def extract_text_from_image(image: np.ndarray) -> List[str]:
    """
    Extract text from an image.
    
    Args:
        image: Image as numpy array
        
    Returns:
        List of extracted text
    """
    try:
        # Check if pytesseract is available
        if not pytesseract.get_tesseract_version():
            logger.warning("Tesseract not available, text extraction disabled")
            return []
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Split into lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        return lines
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return []

def execute_command(command: str) -> dict:
    """
    Execute a shell command.
    
    Args:
        command: Command to execute
        
    Returns:
        Dictionary with command result
    """
    try:
        # Execute command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Get output
        stdout, stderr = process.communicate()
        
        # Get return code
        return_code = process.returncode
        
        return {
            "success": return_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code
        }
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    try:
        # Get system information
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        # Get memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory_total"] = memory.total
            info["memory_available"] = memory.available
            info["memory_percent"] = memory.percent
            
            # Get disk information
            disk = psutil.disk_usage("/")
            info["disk_total"] = disk.total
            info["disk_free"] = disk.free
            info["disk_percent"] = disk.percent
            
            # Get CPU information
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
        except ImportError:
            logger.warning("psutil not available, limited system information")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system information: {e}")
        
        return {
            "error": str(e)
        }

def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"
    
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    
    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"
"""
AI assistant main module.

This module provides the main AI assistant functionality.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import ConfigManager
from .knowledge import KnowledgeManager
from .models import ModelManager
from .session import SessionManager
from .user_model import UserModelManager
from .vision import VisionProcessor
from .voice import VoiceProcessor

# Configure logging
logger = logging.getLogger(__name__)

class AIAssistant:
    """
    AI assistant.
    
    This class provides the main AI assistant functionality.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the AI assistant.
        
        Args:
            config_path: Path to configuration file
        """
        # Configuration
        self.config_manager = ConfigManager(config_path)
        self.config = {}
        
        # Components
        self.model_manager = None
        self.session_manager = None
        self.knowledge_manager = None
        self.user_model_manager = None
        self.vision_processor = None
        self.voice_processor = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize the AI assistant."""
        try:
            # Load configuration
            self.config = await self.config_manager.load_config()
            
            # Configure logging
            log_level = self.config.get("log_level", "INFO")
            logging.basicConfig(
                level=getattr(logging, log_level),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            
            # Initialize components
            await self._initialize_components()
            
            self.initialized = True
            logger.info("AI assistant initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI assistant: {e}")
    
    async def _initialize_components(self):
        """Initialize AI assistant components."""
        # Initialize model manager
        model_path
       # Initialize model manager
        model_path = self.config.get("model_path", "./models")
        self.model_manager = ModelManager(model_path)
        await self.model_manager.initialize()
        
        # Initialize session manager
        session_timeout = self.config.get("session_timeout", 300)
        self.session_manager = SessionManager(session_timeout)
        await self.session_manager.initialize()
        
        # Initialize knowledge manager
        knowledge_dir = self.config.get("knowledge_dir", "./knowledge")
        self.knowledge_manager = KnowledgeManager(knowledge_dir)
        await self.knowledge_manager.initialize()
        
        # Initialize user model manager
        user_models_dir = self.config.get("user_models_dir", "./user_models")
        self.user_model_manager = UserModelManager(user_models_dir)
        await self.user_model_manager.initialize()
        
        # Initialize vision processor
        self.vision_processor = VisionProcessor(model_path)
        await self.vision_processor.initialize()
        
        # Initialize voice processor
        self.voice_processor = VoiceProcessor(model_path)
        await self.voice_processor.initialize()
    
    async def process_query(self, query: str, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Start time
            start_time = time.time()
            
            # Get or create session
            if session_id is None:
                session_id = self.session_manager.create_session()
            else:
                # Update session
                self.session_manager.update_session(session_id)
            
            # Get session
            session = self.session_manager.get_session(session_id)
            if session is None:
                return {
                    "success": False,
                    "error": "Invalid session ID"
                }
            
            # Add query to session history
            session.add_message("user", query)
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "query",
                    query,
                    {"session_id": session_id}
                )
            
            # Process query
            response = await self._generate_response(query, session, user_id)
            
            # Add response to session history
            session.add_message("assistant", response["text"])
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "response",
                    response["text"],
                    {"session_id": session_id}
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return response
            return {
                "success": True,
                "session_id": session_id,
                "text": response["text"],
                "processing_time": processing_time,
                "actions": response.get("actions", []),
                "context": response.get("context", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_response(self, query: str, session, user_id: str = None) -> Dict[str, Any]:
        """
        Generate a response to a user query.
        
        Args:
            query: User query
            session: Session object
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        try:
            # Get conversation history
            history = session.get_messages()
            
            # Get user preferences if user ID is provided
            user_preferences = {}
            if user_id:
                user_model = await self.user_model_manager.get_user_model(user_id)
                user_preferences = user_model.preferences
            
            # Search knowledge base
            knowledge_results = await self.knowledge_manager.search_knowledge(query)
            
            # Prepare context
            context = {
                "knowledge": knowledge_results,
                "user_preferences": user_preferences,
                "session_data": session.data
            }
            
            # Generate response
            response = await self.model_manager.generate_response(
                query,
                history,
                context
            )
            
            # Extract actions
            actions = self._extract_actions(response)
            
            # Execute actions
            action_results = await self._execute_actions(actions, session, user_id)
            
            # Update response if needed
            if action_results:
                # Generate updated response
                updated_response = await self.model_manager.generate_response(
                    query,
                    history,
                    {
                        **context,
                        "action_results": action_results
                    }
                )
                
                response = updated_response
            
            return {
                "text": response,
                "actions": actions,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            return {
                "text": "I'm sorry, I encountered an error while processing your request."
            }
    
    def _extract_actions(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract actions from response.
        
        Args:
            response: Response text
            
        Returns:
            List of actions
        """
        # This is a simplified implementation
        # In a real implementation, we would use a more sophisticated approach
        # to extract actions from the response
        
        actions = []
        
        # Check for action markers
        if "[ACTION:" in response:
            # Extract action blocks
            action_blocks = response.split("[ACTION:")[1:]
            
            for block in action_blocks:
                # Extract action type and parameters
                action_end = block.find("]")
                if action_end != -1:
                    action_type = block[:action_end].strip()
                    
                    # Create action
                    action = {
                        "type": action_type,
                        "parameters": {}
                    }
                    
                    # Extract parameters
                    param_start = block.find("(")
                    param_end = block.find(")")
                    
                    if param_start != -1 and param_end != -1:
                        param_str = block[param_start+1:param_end]
                        
                        # Parse parameters
                        params = param_str.split(",")
                        for param in params:
                            if "=" in param:
                                key, value = param.split("=", 1)
                                action["parameters"][key.strip()] = value.strip()
                    
                    # Add action
                    actions.append(action)
        
        return actions
    
    async def _execute_actions(self, actions: List[Dict[str, Any]], session, user_id: str = None) -> Dict[str, Any]:
        """
        Execute actions.
        
        Args:
            actions: List of actions
            session: Session object
            user_id: User ID (optional)
            
        Returns:
            Action results
        """
        results = {}
        
        for action in actions:
            action_type = action["type"]
            parameters = action["parameters"]
            
            try:
                if action_type == "search_knowledge":
                    # Search knowledge base
                    query = parameters.get("query", "")
                    results["search_knowledge"] = await self.knowledge_manager.search_knowledge(query)
                
                elif action_type == "capture_screen":
                    # Capture and analyze screen
                    results["capture_screen"] = await self.vision_processor.capture_and_analyze_screen()
                
                elif action_type == "record_audio":
                    # Record audio
                    duration = float(parameters.get("duration", 5.0))
                    audio_data, sample_rate = await self.voice_processor.record_audio(duration)
                    
                    # Recognize speech
                    if audio_data is not None:
                        speech_result = await self.voice_processor.recognize_speech(audio_data, sample_rate)
                        results["record_audio"] = speech_result
                
                elif action_type == "synthesize_speech":
                    # Synthesize speech
                    text = parameters.get("text", "")
                    language = parameters.get("language", "en")
                    
                    audio_data, sample_rate = await self.voice_processor.synthesize_speech(text, language)
                    
                    # Play audio
                    if audio_data is not None:
                        await self.voice_processor.play_audio(audio_data, sample_rate)
                        
                        results["synthesize_speech"] = {
                            "success": True,
                            "text": text
                        }
                
                elif action_type == "update_user_preference":
                    # Update user preference
                    if user_id:
                        key = parameters.get("key", "")
                        value = parameters.get("value", "")
                        
                        if key:
                            success = await self.user_model_manager.update_user_preference(user_id, key, value)
                            results["update_user_preference"] = {
                                "success": success,
                                "key": key,
                                "value": value
                            }
                
                elif action_type == "store_session_data":
                    # Store session data
                    key = parameters.get("key", "")
                    value = parameters.get("value", "")
                    
                    if key:
                        session.data[key] = value
                        results["store_session_data"] = {
                            "success": True,
                            "key": key,
                            "value": value
                        }
                
                elif action_type == "execute_command":
                    # Execute command
                    command = parameters.get("command", "")
                    
                    if command:
                        from .utils import execute_command
                        result = execute_command(command)
                        results["execute_command"] = result
                
            except Exception as e:
                logger.error(f"Error executing action {action_type}: {e}")
                results[action_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def process_voice_query(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a voice query.
        
        Args:
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Record audio
            audio_data, sample_rate = await self.voice_processor.record_audio()
            
            if audio_data is None:
                return {
                    "success": False,
                    "error": "Failed to record audio"
                }
            
            # Recognize speech
            speech_result = await self.voice_processor.recognize_speech(audio_data, sample_rate)
            
            if not speech_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to recognize speech"
                }
            
            # Get recognized text
            query = speech_result["text"]
            
            # Process query
            response = await self.process_query(query, session_id, user_id)
            
            # Synthesize speech
            if response["success"]:
                audio_data, sample_rate = await self.voice_processor.synthesize_speech(response["text"])
                
                if audio_data is not None:
                    # Play audio
                    await self.voice_processor.play_audio(audio_data, sample_rate)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_visual_query(self, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process a visual query.
        
        Args:
            session_id: Session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        try:
            # Capture and analyze screen
            screen_result = await self.vision_processor.capture_and_analyze_screen()
            
            if not screen_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to capture and analyze screen"
                }
            
            # Get or create session
            if session_id is None:
                session_id = self.session_manager.create_session()
            else:
                # Update session
                self.session_manager.update_session(session_id)
            
            # Get session
            session = self.session_manager.get_session(session_id)
            if session is None:
                return {
                    "success": False,
                    "error": "Invalid session ID"
                }
            
            # Store screen analysis in session data
            session.data["screen_analysis"] = screen_result
            
            # Generate response
            response = await self._generate_response(
                "What do you see on my screen?",
                session,
                user_id
            )
            
            # Add response to session history
            session.add_message("assistant", response["text"])
            
            # Add user interaction if user ID is provided
            if user_id:
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "visual_query",
                    "What do you see on my screen?",
                    {"session_id": session_id}
                )
                
                await self.user_model_manager.add_user_interaction(
                    user_id,
                    "response",
                    response["text"],
                    {"session_id": session_id}
                )
            
            return {
                "success": True,
                "session_id": session_id,
                "text": response["text"],
                "screen_analysis": screen_result,
                "actions": response.get("actions", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing visual query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return self.session_manager.end_session(session_id)
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information, or None if session doesn't exist
        """
       """
        if not self.initialized:
            return None
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        
        return {
            "session_id": session_id,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "message_count": len(session.messages),
            "data": session.data
        }
    
    async def get_session_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get session messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of messages, or None if session doesn't exist
        """
        if not self.initialized:
            return None
        
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        
        return session.get_messages()
    
    async def get_active_sessions(self) -> List[str]:
        """
        Get active session IDs.
        
        Returns:
            List of active session IDs
        """
        if not self.initialized:
            return []
        
        return self.session_manager.get_active_sessions()
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            
        Returns:
            Knowledge ID
        """
        if not self.initialized:
            return ""
        
        return await self.knowledge_manager.add_knowledge(knowledge)
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        if not self.initialized:
            return None
        
        return await self.knowledge_manager.get_knowledge(knowledge_id)
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.knowledge_manager.update_knowledge(knowledge_id, knowledge)
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.knowledge_manager.delete_knowledge(knowledge_id)
    
    async def search_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching knowledge items
        """
        if not self.initialized:
            return []
        
        return await self.knowledge_manager.search_knowledge(query, max_results=max_results)
    
    async def get_user_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            User model, or None if not found
        """
        if not self.initialized:
            return None
        
        user_model = await self.user_model_manager.get_user_model(user_id)
        return user_model.to_dict()
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.user_model_manager.update_user_preference(user_id, key, value)
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value
        """
        if not self.initialized:
            return default
        
        return await self.user_model_manager.get_user_preference(user_id, key, default)
    
    async def get_user_interactions(self, user_id: str, interaction_type: str = None, count: int = None) -> List[Dict[str, Any]]:
        """
        Get user interactions.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction to filter by
            count: Number of interactions to get
            
        Returns:
            List of interactions
        """
        if not self.initialized:
            return []
        
        return await self.user_model_manager.get_user_interactions(user_id, interaction_type, count)
    
    async def delete_user_model(self, user_id: str) -> bool:
        """
        Delete a user model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.user_model_manager.delete_user_model(user_id)
    
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List of user IDs
        """
        if not self.initialized:
            return []
        
        return await self.user_model_manager.get_all_user_ids()
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Screen analysis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.vision_processor.capture_and_analyze_screen()
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image analysis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.vision_processor.analyze_image(image_path)
    
    async def record_audio(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Record audio.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Audio recording result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        audio_data, sample_rate = await self.voice_processor.record_audio(duration)
        
        if audio_data is None:
            return {
                "success": False,
                "error": "Failed to record audio"
            }
        
        return {
            "success": True,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    
    async def recognize_speech(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Speech recognition result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        return await self.voice_processor.recognize_speech(audio_data, sample_rate)
    
    async def synthesize_speech(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Synthesize speech.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Speech synthesis result
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI assistant not initialized"
            }
        
        audio_data, sample_rate = await self.voice_processor.synthesize_speech(text, language)
        
        if audio_data is None:
            return {
                "success": False,
                "error": "Failed to synthesize speech"
            }
        
        return {
            "success": True,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    
    async def play_audio(self, audio_data: bytes, sample_rate: int) -> bool:
        """
        Play audio.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            return False
        
        return await self.voice_processor.play_audio(audio_data, sample_rate)
    
    async def shutdown(self):
        """Shut down the AI assistant."""
        try:
            # Save user models
            if self.user_model_manager:
                await self.user_model_manager.save_all_user_models()
            
            # Save knowledge base
            if self.knowledge_manager:
                await self.knowledge_manager.save_knowledge_base()
            
            # Save configuration
            if self.config_manager:
                await self.config_manager.save_config(self.config)
            
            logger.info("AI assistant shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down AI assistant: {e}")
"""
AI assistant main entry point.

This module provides the main entry point for the AI assistant.
"""

import argparse
import asyncio
import logging
import sys

from .assistant import AIAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI assistant")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    # Create AI assistant
    assistant = AIAssistant(args.config)
    
    try:
        # Initialize AI assistant
        await assistant.initialize()
        
        # Start interactive mode
        await interactive_mode(assistant)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error: {e}")
    
    finally:
        # Shut down AI assistant
        await assistant.shutdown()

async def interactive_mode(assistant: AIAssistant):
    """
    Interactive mode.
    
    Args:
        assistant: AI assistant
    """
    print("AI assistant interactive mode")
    print("Type 'exit' to exit")
    print("Type 'help' for help")
    print()
    
    session_id = None
    user_id = "interactive_user"
    
    while True:
        try:
            # Get user input
            user_input = input("> ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Check for help command
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Check for special commands
            if user_input.startswith("/"):
                await handle_command(assistant, user_input, session_id, user_id)
                continue
            
            # Process query
            response = await assistant.process_query(user_input, session_id, user_id)
            
            # Update session ID
            if response["success"]:
                session_id = response["session_id"]
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            logger.error(f"Error: {e}")

async def handle_command(assistant: AIAssistant, command: str, session_id: str, user_id: str):
    """
    Handle a special command.
    
    Args:
        assistant: AI assistant
        command: Command string
        session_id: Session ID
        user_id: User ID
    """
    # Parse command
    parts = command[1:].split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]
    
    try:
        if cmd == "voice":
            # Process voice query
            print("Listening...")
            response = await assistant.process_voice_query(session_id, user_id)
            
            if response["success"]:
                session_id = response["session_id"]
                print(f"You said: {response.get('query', '')}")
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        elif cmd == "visual":
            # Process visual query
            print("Capturing screen...")
            response = await assistant.process_visual_query(session_id, user_id)
            
            if response["success"]:
                session_id = response["session_id"]
                print(f"Assistant: {response['text']}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
        
        elif cmd == "session":
            # Session commands
            if not args:
                # Get session info
                if session_id:
                    info = await assistant.get_session_info(session_id)
                    print(f"Session ID: {session_id}")
                    print(f"Created at: {info['created_at']}")
                    print(f"Last updated: {info['last_updated']}")
                    print(f"Message count: {info['message_count']}")
                else:
                    print("No active session")
            
            elif args[0] == "new":
                # Create new session
                session_id = assistant.session_manager.create_session()
                print(f"Created new session: {session_id}")
           elif args[0] == "list":
                # List active sessions
                sessions = await assistant.get_active_sessions()
                print(f"Active sessions ({len(sessions)}):")
                for s in sessions:
                    info = await assistant.get_session_info(s)
                    print(f"  {s} - {info['message_count']} messages, last updated: {info['last_updated']}")
            
            elif args[0] == "end":
                # End session
                if session_id:
                    success = await assistant.end_session(session_id)
                    if success:
                        print(f"Ended session: {session_id}")
                        session_id = None
                    else:
                        print(f"Failed to end session: {session_id}")
                else:
                    print("No active session")
            
            elif args[0] == "messages":
                # Get session messages
                if session_id:
                    messages = await assistant.get_session_messages(session_id)
                    print(f"Session messages ({len(messages)}):")
                    for msg in messages:
                        print(f"  {msg['role']}: {msg['content']}")
                else:
                    print("No active session")
            
            else:
                print(f"Unknown session command: {args[0]}")
        
        elif cmd == "knowledge":
            # Knowledge commands
            if not args:
                print("Usage: /knowledge [add|get|update|delete|search|list]")
            
            elif args[0] == "add":
                # Add knowledge
                if len(args) < 3:
                    print("Usage: /knowledge add <title> <content>")
                else:
                    title = args[1]
                    content = " ".join(args[2:])
                    
                    knowledge = {
                        "title": title,
                        "content": content,
                        "source": "interactive",
                        "tags": []
                    }
                    
                    knowledge_id = await assistant.add_knowledge(knowledge)
                    print(f"Added knowledge: {knowledge_id}")
            
            elif args[0] == "get":
                # Get knowledge
                if len(args) < 2:
                    print("Usage: /knowledge get <id>")
                else:
                    knowledge_id = args[1]
                    knowledge = await assistant.get_knowledge(knowledge_id)
                    
                    if knowledge:
                        print(f"Knowledge ID: {knowledge_id}")
                        print(f"Title: {knowledge['title']}")
                        print(f"Content: {knowledge['content']}")
                        print(f"Source: {knowledge.get('source', 'unknown')}")
                        print(f"Tags: {', '.join(knowledge.get('tags', []))}")
                    else:
                        print(f"Knowledge not found: {knowledge_id}")
            
            elif args[0] == "update":
                # Update knowledge
                if len(args) < 4:
                    print("Usage: /knowledge update <id> <title> <content>")
                else:
                    knowledge_id = args[1]
                    title = args[2]
                    content = " ".join(args[3:])
                    
                    # Get existing knowledge
                    knowledge = await assistant.get_knowledge(knowledge_id)
                    
                    if knowledge:
                        # Update knowledge
                        knowledge["title"] = title
                        knowledge["content"] = content
                        
                        success = await assistant.update_knowledge(knowledge_id, knowledge)
                        
                        if success:
                            print(f"Updated knowledge: {knowledge_id}")
                        else:
                            print(f"Failed to update knowledge: {knowledge_id}")
                    else:
                        print(f"Knowledge not found: {knowledge_id}")
            
            elif args[0] == "delete":
                # Delete knowledge
                if len(args) < 2:
                    print("Usage: /knowledge delete <id>")
                else:
                    knowledge_id = args[1]
                    success = await assistant.delete_knowledge(knowledge_id)
                    
                    if success:
                        print(f"Deleted knowledge: {knowledge_id}")
                    else:
                        print(f"Failed to delete knowledge: {knowledge_id}")
            
            elif args[0] == "search":
                # Search knowledge
                if len(args) < 2:
                    print("Usage: /knowledge search <query>")
                else:
                    query = " ".join(args[1:])
                    results = await assistant.search_knowledge(query)
                    
                    print(f"Search results ({len(results)}):")
                    for result in results:
                        print(f"  ID: {result['id']}")
                        print(f"  Title: {result['knowledge']['title']}")
                        print(f"  Score: {result['score']}")
                        print()
            
            elif args[0] == "list":
                # List all knowledge
                knowledge_items = await assistant.knowledge_manager.get_all_knowledge()
                
                print(f"Knowledge items ({len(knowledge_items)}):")
                for item in knowledge_items:
                    print(f"  ID: {item['id']}")
                    print(f"  Title: {item['knowledge']['title']}")
                    print()
            
            else:
                print(f"Unknown knowledge command: {args[0]}")
        
        elif cmd == "user":
            # User commands
            if not args:
                # Get user info
                user_model = await assistant.get_user_model(user_id)
                
                if user_model:
                    print(f"User ID: {user_model['user_id']}")
                    print(f"Created at: {user_model['created_at']}")
                    print(f"Last updated: {user_model['last_updated']}")
                    print(f"Preferences: {user_model['preferences']}")
                    print(f"Interactions: {len(user_model['interactions'])}")
                    
                    # Get top topics
                    topics = sorted(user_model['topics'].items(), key=lambda x: x[1], reverse=True)[:5]
                    if topics:
                        print("Top topics:")
                        for topic, score in topics:
                            print(f"  {topic}: {score}")
                else:
                    print(f"User not found: {user_id}")
            
            elif args[0] == "preference":
                # User preference commands
                if len(args) < 2:
                    print("Usage: /user preference [get|set] <key> [value]")
                
                elif args[1] == "get":
                    # Get preference
                    if len(args) < 3:
                        print("Usage: /user preference get <key>")
                    else:
                        key = args[2]
                        value = await assistant.get_user_preference(user_id, key)
                        print(f"Preference {key}: {value}")
                
                elif args[1] == "set":
                    # Set preference
                    if len(args) < 4:
                        print("Usage: /user preference set <key> <value>")
                    else:
                        key = args[2]
                        value = " ".join(args[3:])
                        
                        success = await assistant.update_user_preference(user_id, key, value)
                        
                        if success:
                            print(f"Set preference {key} = {value}")
                        else:
                            print(f"Failed to set preference {key}")
                
                else:
                    print(f"Unknown preference command: {args[1]}")
            
            elif args[0] == "interactions":
                # Get user interactions
                count = int(args[1]) if len(args) > 1 else 5
                interactions = await assistant.get_user_interactions(user_id, count=count)
                
                print(f"User interactions ({len(interactions)}):")
                for interaction in interactions:
                    print(f"  {interaction['type']} at {interaction['timestamp']}")
                    print(f"    {interaction['content']}")
                    print()
            
            else:
                print(f"Unknown user command: {args[0]}")
        
        elif cmd == "config":
            # Configuration commands
            if not args:
                # Show configuration
                print("Configuration:")
                for key, value in assistant.config.items():
                    print(f"  {key}: {value}")
            
            elif len(args) >= 2:
                # Set configuration
                key = args[0]
                value = " ".join(args[1:])
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        value = float(value)
                except:
                    pass
                
                # Update configuration
                assistant.config[key] = value
                
                # Save configuration
                await assistant.config_manager.save_config(assistant.config)
                
                print(f"Set {key} = {value}")
            
            else:
                print("Usage: /config [key value]")
        
        elif cmd == "help":
            # Show help
            print_help()
        
        else:
            print(f"Unknown command: {cmd}")
    
    except Exception as e:
        logger.error(f"Error handling command: {e}")
        print(f"Error: {e}")

def print_help():
    """Print help information."""
    print("AI assistant commands:")
    print("  /voice - Process voice query")
    print("  /visual - Process visual query")
    print("  /session - Session commands")
    print("    /session - Show current session info")
    print("    /session new - Create new session")
    print("    /session list - List active sessions")
    print("    /session end - End current session")
    print("    /session messages - Show session messages")
    print("  /knowledge - Knowledge commands")
    print("    /knowledge add <title> <content> - Add knowledge")
    print("    /knowledge get <id> - Get knowledge")
    print("    /knowledge update <id> <title> <content> - Update knowledge")
    print("    /knowledge delete <id> - Delete knowledge")
    print("    /knowledge search <query> - Search knowledge")
    print("    /knowledge list - List all knowledge")
    print("  /user - User commands")
    print("    /user - Show user info")
    print("    /user preference get <key> - Get user preference")
    print("    /user preference set <key> <value> - Set user preference")
    print("    /user interactions [count] - Show user interactions")
    print("  /config - Configuration commands")
    print("    /config - Show configuration")
    print("    /config <key> <value> - Set configuration")
    print("  /help - Show help")
    print("  exit - Exit")

if __name__ == "__main__":
    asyncio.run(main())
"""
Vision processing for the AI assistant.

This module provides vision processing functionality for the AI assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import capture_screen, extract_text_from_image

# Configure logging
logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Vision processor for the AI assistant.
    
    This class provides vision processing functionality for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the vision processor.
        
        Args:
            model_path: Path to vision models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.object_detector = None
        self.face_detector = None
        self.text_detector = None
    
    async def initialize(self):
        """Initialize the vision processor."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            await self._initialize_models()
            
            logger.info("Vision processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vision processor: {e}")
    
    async def _initialize_models(self):
        """Initialize vision models."""
        try:
            # Initialize object detector
            self._initialize_object_detector()
            
            # Initialize face detector
            self._initialize_face_detector()
            
            # Initialize text detector
            self._initialize_text_detector()
            
        except Exception as e:
            logger.error(f"Error initializing vision models: {e}")
    
    def _initialize_object_detector(self):
        """Initialize object detector."""
        try:
            # Check if OpenCV DNN module is available
            if hasattr(cv2, "dnn"):
                # Load YOLO model
                yolo_cfg = self.model_path / "yolov3.cfg"
                yolo_weights = self.model_path / "yolov3.weights"
                
                if yolo_cfg.exists() and yolo_weights.exists():
                    self.object_detector = cv2.dnn.readNetFromDarknet(
                        str(yolo_cfg),
                        str(yolo_weights)
                    )
                    
                    # Load COCO class names
                    coco_names = self.model_path / "coco.names"
                    if coco_names.exists():
                        with open(coco_names, "r") as f:
                            self.object_classes = f.read().strip().split("\n")
                    else:
                        # Default COCO class names
                        self.object_classes = [
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                            "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                        ]
                    
                    logger.info("Object detector initialized")
                else:
                    logger.warning("YOLO model files not found, object detection disabled")
            else:
                logger.warning("OpenCV DNN module not available, object detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
    
    def _initialize_face_detector(self):
        """Initialize face detector."""
        try:
            # Load Haar cascade for face detection
           # Load Haar cascade for face detection
            face_cascade_path = self.model_path / "haarcascade_frontalface_default.xml"
            
            if face_cascade_path.exists():
                self.face_detector = cv2.CascadeClassifier(str(face_cascade_path))
                logger.info("Face detector initialized")
            else:
                # Try to use OpenCV's built-in cascades
                builtin_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(builtin_cascade):
                    self.face_detector = cv2.CascadeClassifier(builtin_cascade)
                    logger.info("Face detector initialized using built-in cascade")
                else:
                    logger.warning("Face cascade file not found, face detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
    
    def _initialize_text_detector(self):
        """Initialize text detector."""
        try:
            # Check if OpenCV text detection is available
            if hasattr(cv2, "text") and hasattr(cv2.text, "TextDetectorCNN_create"):
                # Load EAST text detector
                east_model_path = self.model_path / "frozen_east_text_detection.pb"
                
                if east_model_path.exists():
                    self.text_detector = cv2.dnn.readNet(str(east_model_path))
                    logger.info("Text detector initialized")
                else:
                    logger.warning("EAST model file not found, advanced text detection disabled")
            else:
                logger.warning("OpenCV text detection not available, advanced text detection disabled")
                
        except Exception as e:
            logger.error(f"Error initializing text detector: {e}")
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture and analyze the screen.
        
        Returns:
            Screen analysis result
        """
        try:
            # Capture screen
            screen = capture_screen()
            
            if screen is None:
                return {
                    "success": False,
                    "error": "Failed to capture screen"
                }
            
            # Analyze screen
            analysis = await self.analyze_image_data(screen)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error capturing and analyzing screen: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image analysis result
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to read image"
                }
            
            # Analyze image
            analysis = await self.analyze_image_data(image)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_image_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image data.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            Image analysis result
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Extract text
            text = extract_text_from_image(image)
            
            # Detect objects
            objects = self._detect_objects(image) if self.object_detector else []
            
            # Detect faces
            faces = self._detect_faces(image) if self.face_detector else []
            
            # Detect colors
            colors = self._detect_dominant_colors(image)
            
            # Basic image properties
            properties = {
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height > 0 else 0,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "is_color": len(image.shape) > 2 and image.shape[2] > 1
            }
            
            return {
                "properties": properties,
                "text": text,
                "objects": objects,
                "faces": faces,
                "colors": colors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image data: {e}")
            
            return {
                "properties": {
                    "width": 0,
                    "height": 0,
                    "aspect_ratio": 0,
                    "channels": 0,
                    "is_color": False
                },
                "text": [],
                "objects": [],
                "faces": [],
                "colors": []
            }
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            List of detected objects
        """
        try:
            if self.object_detector is None:
                return []
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image,
                1/255.0,
                (416, 416),
                swapRB=True,
                crop=False
            )
            
            # Set input
            self.object_detector.setInput(blob)
            
            # Get output layer names
            layer_names = self.object_detector.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.object_detector.getUnconnectedOutLayers()]
            
            # Forward pass
            outputs = self.object_detector.forward(output_layers)
            
            # Process outputs
            class_ids = []
            confidences = []
            boxes = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Prepare results
            objects = []
            
            for i in indices:
                if isinstance(i, tuple):
                    i = i[0]  # For OpenCV 3
                
                box = boxes[i]
                x, y, w, h = box
                
                # Get class name
                class_id = class_ids[i]
                class_name = self.object_classes[class_id] if class_id < len(self.object_classes) else f"unknown_{class_id}"
                
                # Add object
                objects.append({
                    "class": class_name,
                    "confidence": confidences[i],
                    "box": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "center": {
                        "x": x + w // 2,
                        "y": y + h // 2
                    },
                    "area": w * h,
                    "relative_area": (w * h) / (width * height)
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image data as numpy array
            
        Returns:
            List of detected faces
        """
        try:
            if self.face_detector is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rect = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Prepare results
            faces = []
            
            for (x, y, w, h) in faces_rect:
                # Add face
                faces.append({
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "center": {
                        "x": int(x + w // 2),
                        "y": int(y + h // 2)
                    },
                    "area": int(w * h),
                    "relative_area": float((w * h) / (width * height))
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _detect_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Dict[str, Any]]:
        """
        Detect dominant colors in an image.
        
        Args:
            image: Image data as numpy array
            num_colors: Number of dominant colors to detect
            
        Returns:
            List of dominant colors
        """
        try:
            # Reshape image
            pixels = image.reshape(-1, 3)
            
            # Convert to float32
            pixels = np.float32(pixels)
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Apply k-means clustering
            _, labels, centers = cv2.kmeans(
                pixels,
                num_colors,
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert centers to uint8
            centers = np.uint8(centers)
            
            # Count labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            
            # Prepare results
            colors = []
            
            for i in sorted_indices:
                # Get color
                color = centers[i].tolist()
                
                # Calculate percentage
                percentage = counts[i] / len(labels)
                
                # Add color
                colors.append({
                    "color": {
                        "b": color[0],
                        "g": color[1],
                        "r": color[2],
                        "hex": f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                    },
                    "percentage": float(percentage)
                })
            
            return colors
            
        except Exception as e:
            logger.error(f"Error detecting dominant colors: {e}")
            return []
"""
Voice processing for the AI assistant.

This module provides voice processing functionality for the AI assistant.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processor for the AI assistant.
    
    This class provides voice processing functionality for the AI assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the voice processor.
        
        Args:
            model_path: Path to voice models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.speech_recognizer = None
        self.speech_synthesizer = None
        
        # Audio device
        self.audio_device = None
    
    async def initialize(self):
        """Initialize the voice processor."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            await self._initialize_models()
            
            # Initialize audio device
            await self._initialize_audio_device()
            
            logger.info("Voice processor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing voice processor: {e}")
    
    async def _initialize_models(self):
        """Initialize voice models."""
        try:
            # Initialize speech recognizer
            await self._initialize_speech_recognizer()
            
            # Initialize speech synthesizer
            await self._initialize_speech_synthesizer()
            
        except Exception as e:
            logger.error(f"Error initializing voice models: {e}")
    
    async def _initialize_speech_recognizer(self):
        """Initialize speech recognizer."""
        try:
            # Try to import speech recognition library
            import speech_recognition as sr
            
            # Create recognizer
            self.speech_recognizer = sr.Recognizer()
            
            logger.info("Speech recognizer initialized")
            
        except ImportError:
            logger.warning("speech_recognition not available, speech recognition disabled")
            
        except Exception as e:
            logger.error(f"Error initializing speech recognizer: {e}")
    
    async def _initialize_speech_synthesizer(self):
        """Initialize speech synthesizer."""
        try:
            # Try to import pyttsx3
            import pyttsx3
            
            # Create synthesizer
            self.speech_synthesizer = pyttsx3.init()
            
            logger.info("Speech synthesizer initialized")
            
        except ImportError:
            logger.warning("pyttsx3 not available, speech synthesis disabled")
            
            # Try to import gTTS as fallback
            try:
                from gtts import gTTS
                
                # Use gTTS
                self.speech_synthesizer = "gtts"
                
                logger.info("Speech synthesizer initialized using gTTS")
                
            except ImportError:
                logger.warning("gtts not available, speech synthesis disabled")
            
        except Exception as e:
            logger.error(f"Error initializing speech synthesizer: {e}")
    
    async def _initialize_audio_device(self):
        """Initialize audio device."""
        try:
            #

"""
Session management for the AI assistant.

This module provides session management functionality for the AI assistant.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class Session:
    """
    Session class for the AI assistant.
    
    This class represents a session with the AI assistant.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize a session.
        
        Args:
            session_id: Session ID
        """
        self.session_id = session_id
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.messages = []
        self.data = {}
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the session.
        
        Args:
            role: Message role (user or assistant)
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        self.last_updated = time.time()
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get session messages.
        
        Returns:
            List of messages
        """
        return self.messages
    
    def clear_messages(self):
        """Clear session messages."""
        self.messages = []
        self.last_updated = time.time()
    
    def is_expired(self, timeout: float) -> bool:
        """
        Check if the session is expired.
        
        Args:
            timeout: Session timeout in seconds
            
        Returns:
            True if the session is expired, False otherwise
        """
        return time.time() - self.last_updated > timeout

class SessionManager:
    """
    Session manager for the AI assistant.
    
    This class manages sessions for the AI assistant.
    """
    
    def __init__(self, session_timeout: float = 300.0):
        """
        Initialize the session manager.
        
        Args:
            session_timeout: Session timeout in seconds
        """
        self.sessions = {}
        self.session_timeout = session_timeout
    
    async def initialize(self):
        """Initialize the session manager."""
        try:
            logger.info("Session manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing session manager: {e}")
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        self.sessions[session_id] = Session(session_id)
        
        logger.info(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session, or None if not found
        """
        # Check if session exists
        if session_id not in self.sessions:
            return None
        
        # Get session
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.session_timeout):
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
            
            return None
        
        return session
    
    def update_session(self, session_id: str) -> bool:
        """
        Update a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.sessions:
            return False
        
        # Get session
        session = self.sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.session_timeout):
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
            
            return False
        
        # Update session
        session.last_updated = time.time()
        
        return True
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        if session_id not in self.sessions:
            return False
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Ended session: {session_id}")
        
        return True
    
    def get_active_sessions(self) -> List[str]:
        """
        Get active session IDs.
        
        Returns:
            List of active session IDs
        """
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        # Return active session IDs
        return list(self.sessions.keys())
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        # Get current time
        current_time = time.time()
        
        # Find expired sessions
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if current_time - session.last_updated > self.session_timeout
        ]
        
        # Remove expired sessions
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
            logger.info(f"Session expired: {session_id}")
"""
Model management for the AI assistant.

This module provides model management functionality for the AI assistant.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager for the AI assistant.
    
    This class manages AI models for the assistant.
    """
    
    def __init__(self, model_path: str = "./models"):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to models
        """
        self.model_path = Path(model_path)
        
        # Models
        self.models = {}
        
        # Default model
        self.default_model = None
    
    async def initialize(self):
        """Initialize the model manager."""
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Load models
            await self._load_models()
            
            logger.info("Model manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model manager: {e}")
    
    async def _load_models(self):
        """Load AI models."""
        try:
            # Try to import transformers
            try:
                from transformers import pipeline
                
                # Load text generation model
                try:
                    logger.info("Loading text generation model...")
                    
                    # Check if model exists locally
                    local_model_path = self.model_path / "text-generation"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["text-generation"] = pipeline(
                            "text-generation",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["text-generation"] = pipeline(
                            "text-generation",
                            model="gpt2"
                        )
                    
                    logger.info("Text generation model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading text generation model: {e}")
                
                # Load text classification model
                try:
                    logger.info("Loading text classification model...")
                    
                    # Check if model exists locally
                    local_model_path = self.model_path / "text-classification"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["text-classification"] = pipeline(
                            "text-classification",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["text-classification"] = pipeline(
                            "text-classification"
                        )
                    
                    logger.info("Text classification model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading text classification model: {e}")
                
                # Load question answering model
                try:
                    logger.info("Loading question answering model...")
                    
                    # Check if model
                   # Check if model exists locally
                    local_model_path = self.model_path / "question-answering"
                    
                    if local_model_path.exists():
                        # Load local model
                        self.models["question-answering"] = pipeline(
                            "question-answering",
                            model=str(local_model_path)
                        )
                    else:
                        # Load default model
                        self.models["question-answering"] = pipeline(
                            "question-answering"
                        )
                    
                    logger.info("Question answering model loaded")
                    
                except Exception as e:
                    logger.error(f"Error loading question answering model: {e}")
                
                # Set default model
                self.default_model = "text-generation"
                
            except ImportError:
                logger.warning("transformers not available, using fallback models")
                
                # Load fallback models
                self._load_fallback_models()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
            # Load fallback models
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback AI models."""
        try:
            # Simple rule-based model
            self.models["rule-based"] = {
                "type": "rule-based",
                "rules": [
                    {
                        "pattern": r"hello|hi|hey",
                        "response": "Hello! How can I help you today?"
                    },
                    {
                        "pattern": r"how are you",
                        "response": "I'm functioning well, thank you for asking. How can I assist you?"
                    },
                    {
                        "pattern": r"bye|goodbye",
                        "response": "Goodbye! Have a great day."
                    },
                    {
                        "pattern": r"thank you|thanks",
                        "response": "You're welcome! Is there anything else I can help with?"
                    },
                    {
                        "pattern": r"help",
                        "response": "I'm here to help. What do you need assistance with?"
                    }
                ]
            }
            
            # Set default model
            self.default_model = "rule-based"
            
            logger.info("Fallback models loaded")
            
        except Exception as e:
            logger.error(f"Error loading fallback models: {e}")
    
    async def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        try:
            # Check if text generation model is available
            if "text-generation" in self.models:
                # Generate text
                result = self.models["text-generation"](
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1
                )
                
                # Extract generated text
                generated_text = result[0]["generated_text"]
                
                return generated_text
            
            # Check if rule-based model is available
            elif "rule-based" in self.models:
                # Use rule-based model
                import re
                
                # Check rules
                for rule in self.models["rule-based"]["rules"]:
                    if re.search(rule["pattern"], prompt.lower()):
                        return rule["response"]
                
                # Default response
                return "I'm not sure how to respond to that."
            
            else:
                return "Text generation model not available."
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"
    
    async def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result
        """
        try:
            # Check if text classification model is available
            if "text-classification" in self.models:
                # Classify text
                result = self.models["text-classification"](text)
                
                # Extract classification
                if isinstance(result, list):
                    result = result[0]
                
                return {
                    "label": result["label"],
                    "score": result["score"]
                }
            
            else:
                return {
                    "label": "unknown",
                    "score": 0.0
                }
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            
            return {
                "label": "error",
                "score": 0.0,
                "error": str(e)
            }
    
    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question based on context.
        
        Args:
            question: Question to answer
            context: Context for the question
            
        Returns:
            Answer result
        """
        try:
            # Check if question answering model is available
            if "question-answering" in self.models:
                # Answer question
                result = self.models["question-answering"](
                    question=question,
                    context=context
                )
                
                return {
                    "answer": result["answer"],
                    "score": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                }
            
            else:
                return {
                    "answer": "I don't know the answer to that question.",
                    "score": 0.0,
                    "start": 0,
                    "end": 0
                }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            
            return {
                "answer": f"Error answering question: {e}",
                "score": 0.0,
                "start": 0,
                "end": 0
            }
    
    async def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_name: Model name, or None for default model
            
        Returns:
            Model information
        """
        try:
            # Get model name
            if model_name is None:
                model_name = self.default_model
            
            # Check if model exists
            if model_name not in self.models:
                return {
                    "name": model_name,
                    "available": False,
                    "error": "Model not found"
                }
            
            # Get model
            model = self.models[model_name]
            
            # Get model info
            if model_name == "rule-based":
                return {
                    "name": model_name,
                    "available": True,
                    "type": "rule-based",
                    "rules": len(model["rules"])
                }
            else:
                return {
                    "name": model_name,
                    "available": True,
                    "type": "transformer"
                }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            
            return {
                "name": model_name,
                "available": False,
                "error": str(e)
            }
    
    async def get_available_models(self) -> List[str]:
        """
        Get available models.
        
        Returns:
            List of available model names
        """
        return list(self.models.keys())
"""
Knowledge management for the AI assistant.

This module provides knowledge management functionality for the AI assistant.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge manager for the AI assistant.
    
    This class manages knowledge for the AI assistant.
    """
    
    def __init__(self, knowledge_path: str = "./knowledge"):
        """
        Initialize the knowledge manager.
        
        Args:
            knowledge_path: Path to knowledge base
        """
        self.knowledge_path = Path(knowledge_path)
        
        # Knowledge base
        self.knowledge_base = {}
        
        # Vector store for search
        self.vector_store = None
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        try:
            # Create knowledge directory if it doesn't exist
            self.knowledge_path.mkdir(parents=True, exist_ok=True)
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            logger.info("Knowledge manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge manager: {e}")
    
    async def _load_knowledge_base(self):
        """Load knowledge base."""
        try:
            # Knowledge base file
            knowledge_file = self.knowledge_path / "knowledge_base.json"
            
            # Check if file exists
            if knowledge_file.exists():
                # Load knowledge base
                with open(knowledge_file, "r") as f:
                    self.knowledge_base = json.load(f)
                
                logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} items")
            else:
                # Create empty knowledge base
                self.knowledge_base = {}
                
                logger.info("Created empty knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            
            # Create empty knowledge base
            self.knowledge_base = {}
    
    async def _initialize_vector_store(self):
        """Initialize vector store for search."""
        try:
            # Try to import sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                
                # Load model
                self.vector_store = SentenceTransformer("all-MiniLM-L6-v2")
                
                logger.info("Vector store initialized")
                
            except ImportError:
                logger.warning("sentence-transformers not available, using fallback search")
                
                # Use fallback search
                self.vector_store = None
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            
            # Use fallback search
            self.vector_store = None
    
    async def save_knowledge_base(self):
        """Save knowledge base."""
        try:
            # Knowledge base file
            knowledge_file = self.knowledge_path / "knowledge_base.json"
            
            # Save knowledge base
            with open(knowledge_file, "w") as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            logger.info(f"Saved knowledge base with {len(self.knowledge_base)} items")
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add
            
        Returns:
            Knowledge ID
        """
        try:
            # Generate knowledge ID
            knowledge_id = str(uuid.uuid4())
            
            # Add knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Added knowledge: {knowledge_id}")
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Knowledge, or None if not found
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return None
            
            # Get knowledge
            return self.knowledge_base[knowledge_id]
            
        except Exception as e:
            logger.error(f"Error getting knowledge: {e}")
            return None
    
    async def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            knowledge: Updated knowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return False
            
            # Update knowledge
            self.knowledge_base[knowledge_id] = knowledge
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Updated knowledge: {knowledge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the knowledge base.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if knowledge exists
            if knowledge_id not in self.knowledge_base:
                return False
            
            # Delete knowledge
            del self.knowledge_base[knowledge_id]
            
            # Save knowledge base
            await self.save_knowledge_base()
            
            logger.info(f"Deleted knowledge: {knowledge_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge: {e}")
            return False
   async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Check if knowledge base is empty
            if not self.knowledge_base:
                return []
            
            # Check if vector store is available
            if self.vector_store is not None:
                # Use vector store for search
                return await self._vector_search(query, limit)
            else:
                # Use fallback search
                return await self._fallback_search(query, limit)
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search knowledge base using vector store.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Encode query
            query_embedding = self.vector_store.encode(query)
            
            # Encode knowledge
            knowledge_texts = []
            knowledge_ids = []
            
            for knowledge_id, knowledge in self.knowledge_base.items():
                # Get text
                text = f"{knowledge.get('title', '')} {knowledge.get('content', '')}"
                
                # Add to list
                knowledge_texts.append(text)
                knowledge_ids.append(knowledge_id)
            
            # Encode knowledge
            knowledge_embeddings = self.vector_store.encode(knowledge_texts)
            
            # Calculate similarity
            from numpy import dot
            from numpy.linalg import norm
            
            # Calculate cosine similarity
            similarities = [
                dot(query_embedding, embedding) / (norm(query_embedding) * norm(embedding))
                for embedding in knowledge_embeddings
            ]
            
            # Sort by similarity
            results = sorted(
                [
                    {
                        "id": knowledge_id,
                        "knowledge": self.knowledge_base[knowledge_id],
                        "score": float(similarity)
                    }
                    for knowledge_id, similarity in zip(knowledge_ids, similarities)
                ],
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            
            # Fallback to basic search
            return await self._fallback_search(query, limit)
    
    async def _fallback_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search knowledge base using fallback search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Simple keyword search
            query_terms = query.lower().split()
            
            # Calculate scores
            results = []
            
            for knowledge_id, knowledge in self.knowledge_base.items():
                # Get text
                title = knowledge.get("title", "").lower()
                content = knowledge.get("content", "").lower()
                
                # Calculate score
                score = 0.0
                
                for term in query_terms:
                    # Check title
                    if term in title:
                        score += 2.0
                    
                    # Check content
                    if term in content:
                        score += 1.0
                
                # Normalize score
                if len(query_terms) > 0:
                    score /= len(query_terms)
                
                # Add to results if score > 0
                if score > 0:
                    results.append({
                        "id": knowledge_id,
                        "knowledge": knowledge,
                        "score": score
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing fallback search: {e}")
            return []
    
    async def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge.
        
        Returns:
            List of all knowledge items
        """
        try:
            # Convert knowledge base to list
            return [
                {
                    "id": knowledge_id,
                    "knowledge": knowledge
                }
                for knowledge_id, knowledge in self.knowledge_base.items()
            ]
            
        except Exception as e:
            logger.error(f"Error getting all knowledge: {e}")
            return []
"""
Configuration management for the AI assistant.

This module provides configuration management functionality for the AI assistant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the AI assistant.
    
    This class manages configuration for the AI assistant.
    """
    
    def __init__(self, config_path: str = "./config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration
        """
        self.config_path = Path(config_path)
        
        # Default configuration
        self.default_config = {
            # General
            "name": "AI Assistant",
            "version": "1.0.0",
            
            # Session
            "session_timeout": 300.0,  # 5 minutes
            
            # Models
            "model_path": "./models",
            "default_model": "text-generation",
            
            # Knowledge
            "knowledge_path": "./knowledge",
            
            # Voice
            "voice_enabled": True,
            "voice_language": "en",
            
            # Vision
            "vision_enabled": True,
            
            # User
            "user_path": "./users",
            
            # Logging
            "log_level": "INFO",
            "log_file": "./logs/assistant.log",
            
            # Advanced
            "debug_mode": False,
            "max_tokens": 100,
            "temperature": 0.7
        }
    
    async def initialize(self):
        """Initialize the configuration manager."""
        try:
            # Create configuration directory if it doesn't exist
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing configuration manager: {e}")
    
    async def load_config(self) -> Dict[str, Any]:
        """
        Load configuration.
        
        Returns:
            Configuration
        """
        try:
            # Configuration file
            config_file = self.config_path / "config.json"
            
            # Check if file exists
            if config_file.exists():
                # Load configuration
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                logger.info("Loaded configuration")
                
                # Merge with default configuration
                merged_config = self.default_config.copy()
                merged_config.update(config)
                
                return merged_config
            else:
                # Create default configuration
                config = self.default_config.copy()
                
                # Save configuration
                await self.save_config(config)
                
                logger.info("Created default configuration")
                
                return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
            # Return default configuration
            return self.default_config.copy()
    
    async def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration.
        
        Args:
            config: Configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Configuration file
            config_file = self.config_path / "config.json"
            
            # Save configuration
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info("Saved configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    async def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        try:
            # Load configuration
            config = await self.load_config()
            
            # Get value
            return config.get(key, default)
            
        except Exception as e:
            logger.error(f"Error getting configuration value: {e}")
            
            # Return default value
            return default
    
    async def set_config_value(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load configuration
            config = await self.load_config()
            
            # Set value
            config[key] = value
            
            # Save configuration
            return await self.save_config(config)
            
        except Exception as e:
            logger.error(f"Error setting configuration value: {e}")
            return False
    
    async def reset_config(self) -> bool:
        """
        Reset configuration to default.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save default configuration
            return await self.save_config(self.default_config.copy())
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False



            
        Returns:
            Configuration value
        """
        try:
            # Load configuration
            config = await self.load_config()
            
            # Get value
            return config.get(key, default)
            
        except Exception as e:
            logger.error(f"Error getting configuration value: {e}")
            
            # Return default value
            return default
    
    async def set_config_value(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load configuration
            config = await self.load_config()
            
            # Set value
            config[key] = value
            
            # Save configuration
            return await self.save_config(config)
            
        except Exception as e:
            logger.error(f"Error setting configuration value: {e}")
            return False
    
    async def reset_config(self) -> bool:
        """
        Reset configuration to default.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save default configuration
            return await self.save_config(self.default_config.copy())
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False



