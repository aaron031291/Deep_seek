"""
Remote access functionality for the AI assistant.

This module provides remote access capabilities for the AI assistant,
allowing it to be controlled from remote devices and execute commands
on the local system.
"""

import asyncio
import json
import logging
import os
import ssl
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol

# Configure logging
logger = logging.getLogger(__name__)

class RemoteAccessManager:
    """
    Remote access manager for the AI assistant.
    
    This class manages remote access to the AI assistant, allowing it to be
    controlled from remote devices and execute commands on the local system.
    """
    
    def __init__(self, assistant, config: Dict[str, Any]):
        """
        Initialize the remote access manager.
        
        Args:
            assistant: AI assistant instance
            config: Configuration
        """
        self.assistant = assistant
        self.config = config
        
        # Server
        self.server = None
        self.server_task = None
        
        # Clients
        self.clients = {}
        self.active_client = None
        
        # Device registry
        self.devices = {}
        self.whitelisted_devices = set()
        
        # Command whitelist
        self.command_whitelist = set()
        
        # Connection status
        self.is_connected = False
        
        # Secure connection
        self.ssl_context = None
    
    async def initialize(self):
        """Initialize the remote access manager."""
        try:
            # Load device registry
            await self._load_device_registry()
            
            # Load command whitelist
            await self._load_command_whitelist()
            
            # Initialize SSL context
            await self._initialize_ssl_context()
            
            # Start server if enabled
            if self.config.get("remote_access_enabled", False):
                await self.start_server()
            
            logger.info("Remote access manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing remote access manager: {e}")
    
    async def _load_device_registry(self):
        """Load device registry."""
        try:
            # Device registry file
            registry_path = Path(self.config.get("device_registry_path", "./devices"))
            registry_file = registry_path / "device_registry.json"
            
            # Create directory if it doesn't exist
            registry_path.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            if registry_file.exists():
                # Load device registry
                with open(registry_file, "r") as f:
                    registry_data = json.load(f)
                
                # Set devices
                self.devices = registry_data.get("devices", {})
                
                # Set whitelisted devices
                self.whitelisted_devices = set(registry_data.get("whitelisted_devices", []))
                
                logger.info(f"Loaded device registry with {len(self.devices)} devices")
            else:
                # Create empty device registry
                self.devices = {}
                self.whitelisted_devices = set()
                
                # Save device registry
                await self._save_device_registry()
                
                logger.info("Created empty device registry")
            
        except Exception as e:
            logger.error(f"Error loading device registry: {e}")
            
            # Create empty device registry
            self.devices = {}
            self.whitelisted_devices = set()
    
    async def _save_device_registry(self):
        """Save device registry."""
        try:
            # Device registry file
            registry_path = Path(self.config.get("device_registry_path", "./devices"))
            registry_file = registry_path / "device_registry.json"
            
            # Create directory if it doesn't exist
            registry_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare registry data
            registry_data = {
                "devices": self.devices,
                "whitelisted_devices": list(self.whitelisted_devices)
            }
            
            # Save device registry
            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Saved device registry with {len(self.devices)} devices")
            
        except Exception as e:
            logger.error(f"Error saving device registry: {e}")
    
    async def _load_command_whitelist(self):
        """Load command whitelist."""
        try:
            # Command whitelist file
            whitelist_path = Path(self.config.get("command_whitelist_path", "./commands"))
            whitelist_file = whitelist_path / "command_whitelist.json"
            
            # Create directory if it doesn't exist
            whitelist_path.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            if whitelist_file.exists():
                # Load command whitelist
                with open(whitelist_file, "r") as f:
                    whitelist_data = json.load(f)
                
                # Set command whitelist
                self.command_whitelist = set(whitelist_data.get("commands", []))
                
                logger.info(f"Loaded command whitelist with {len(self.command_whitelist)} commands")
            else:
                # Create default command whitelist
                self.command_whitelist = {
                    "search", "open", "close", "navigate", "read", "write", "copy", "paste",
                    "screenshot", "record", "play", "pause", "stop", "volume", "brightness",
                    "notification", "reminder", "alarm", "calendar", "email", "message",
                    "call", "answer", "hangup", "mute", "unmute", "lock", "unlock"
                }
                
                # Save command whitelist
                await self._save_command_whitelist()
                
                logger.info(f"Created default command whitelist with {len(self.command_whitelist)} commands")
            
        except Exception as e:
            logger.error(f"Error loading command whitelist: {e}")
            
            # Create default command whitelist
            self.command_whitelist = {
                "search", "open", "close", "navigate", "read", "write", "copy", "paste",
                "screenshot", "record", "play", "pause", "stop", "volume", "brightness",
                "notification", "reminder", "alarm", "calendar", "email", "message",
                "call", "answer", "hangup", "mute", "unmute", "lock", "unlock"
            }
    
    async def _save_command_whitelist(self):
        """Save command whitelist."""
        try:
            # Command whitelist file
            whitelist_path = Path(self.config.get("command_whitelist_path", "./commands"))
            whitelist_file = whitelist_path / "command_whitelist.json"
            
            # Create directory if it doesn't exist
            whitelist_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare whitelist data
            whitelist_data = {
                "commands": list(self.command_whitelist)
            }
            
            # Save command whitelist
            with open(whitelist_file, "w") as f:
                json.dump(whitelist_data, f, indent=2)
            
            logger.info(f"Saved command whitelist with {len(self.command_whitelist)} commands")
            
        except Exception as e:
            logger.error(f"Error saving command whitelist: {e}")
    
    async def _initialize_ssl_context(self):
        """Initialize SSL context for secure connections."""
        try:
            # Check if SSL is enabled
            if self.config.get("remote_access_ssl_enabled", True):
                # SSL certificate and key paths
                cert_path = Path(self.config.get("ssl_cert_path", "./certs/server.crt"))
                key_path = Path(self.config.get("ssl_key_path", "./certs/server.key"))
                
                # Create directory if it doesn't exist
                cert_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if certificate and key exist
                if cert_path.exists() and key_path.exists():
                    # Create SSL context
                    self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    self.ssl_context.load_cert_chain(cert_path, key_path)
                    
                    logger.info("SSL context initialized")
                else:
                    logger.warning("SSL certificate or key not found, using insecure connection")
                    self.ssl_context = None
            else:
                logger.warning("SSL disabled, using insecure connection")
                self.ssl_context = None
            
        except Exception as e:
            logger.error(f"Error initializing SSL context: {e}")
            self.ssl_context = None
    
    async def start_server(self):
        """Start the remote access server."""
        try:
            # Check if server is already running
            if self.server is not None:
                logger.warning("Server is already running")
                return
            
            # Get server configuration
            host = self.config.get("remote_access_host", "0.0.0.0")
            port = self.config.get("remote_access_port", 8765)
            
            # Start server
            self.server = await websockets.serve(
                self._handle_client,
                host,
                port,
                ssl=self.ssl_context
            )
            
            # Start server task
            self.server_task = asyncio.create_task(self.server.wait_closed())
            
            logger.info(f"Remote access server started on {host}:{port}")
            
        except Exception as e:
            logger.error(f"Error starting remote access server: {e}")
            self.server = None
    
    async def stop_server(self):
        """Stop the remote access server."""
        try:
            # Check if server is running
            if self.server is None:
                logger.warning("Server is not running")
                return
            
            # Close server
            self.server.close()
            
            # Wait for server to close
            await self.server.wait_closed()
            
            # Clear server
            self.server = None
            self.server_task = None
            
            logger.info("Remote access server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping remote access server: {e}")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        try:
            # Generate client ID
            client_id = str(uuid.uuid4())
            
            # Add client
            self.clients[client_id] = {
                "websocket": websocket,
                "device_id": None,
                "authenticated": False,
                "connected_at": time.time(),
                "last_activity": time.time()
            }
            
            logger.info(f"Client connected: {client_id}")
            
            # Handle client messages
            try:
                async for message in websocket:
                    try:
                        # Parse message
                        data = json.loads(message)
                        
                        # Update last activity
                        self.clients[client_id]["last_activity"] = time.time()
                        
                        # Handle message
                        await self._handle_client_message(client_id, data)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid message from client {client_id}: {message}")
                        
                        # Send error response
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "Invalid message format"
                        }))
                    
                    except Exception as e:
                        logger.error(f"Error handling message from client {client_id}: {e}")
                        
                        # Send error response
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))
            
            finally:
                # Remove client
                if client_id in self.clients:
                    del self.clients[client_id]
                
                # Clear active client if this was the active client
                if self.active_client == client_id:
                    self.active_client = None
                
                logger.info(f"Client disconnected: {client_id}")
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
    
    async def _handle_client_message(self, client_id: str, data: Dict[str, Any]):
        """
        Handle client message.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Get message type
            message_type = data.get("type")
            
            if message_type is None:
                logger.warning(f"Missing message type from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing message type"
                }))
                
                return
            
            # Handle message based on type
            if message_type == "register":
                # Register device
                await self._handle_register(client_id, data)
                
            elif message_type == "authenticate":
                # Authenticate device
                await self._handle_authenticate(client_id, data)
                
            elif message_type == "command":
                # Execute command
                await self._handle_command(client_id, data)
                
            elif message_type == "query":
                # Process query
                await self._handle_query(client_id, data)
                
            elif message_type == "voice":
                # Process voice input
                await self._handle_voice(client_id, data)
                
            elif message_type == "visual":
                # Process visual input
                await self._handle_visual(client_id, data)
                
            elif message_type == "switch":
                # Switch active device
                await self._handle_switch(client_id, data)
                
            elif message_type == "ping":
                # Ping
                await self._handle_ping(client_id, data)
                
            else:
                logger.warning(f"Unknown message type from client {client_id}: {message_type}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                }))
            
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
   async def _handle_register(self, client_id: str, data: Dict[str, Any]):
        """
        Handle device registration.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Get device information
            device_info = data.get("device", {})
            
            if not device_info:
                logger.warning(f"Missing device information from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing device information"
                }))
                
                return
            
            # Get or generate device ID
            device_id = device_info.get("id")
            
            if device_id is None:
                # Generate device ID
                device_id = str(uuid.uuid4())
                
                # Add device ID to device info
                device_info["id"] = device_id
            
            # Check if device exists
            if device_id in self.devices:
                # Update device information
                self.devices[device_id].update(device_info)
                
                logger.info(f"Updated device: {device_id}")
            else:
                # Add device
                self.devices[device_id] = device_info
                
                logger.info(f"Registered new device: {device_id}")
            
            # Update client
            client["device_id"] = device_id
            
            # Save device registry
            await self._save_device_registry()
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "register_response",
                "device_id": device_id,
                "whitelisted": device_id in self.whitelisted_devices
            }))
            
        except Exception as e:
            logger.error(f"Error handling device registration: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _handle_authenticate(self, client_id: str, data: Dict[str, Any]):
        """
        Handle device authentication.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Get device ID
            device_id = client.get("device_id")
            
            if device_id is None:
                logger.warning(f"Device not registered for client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Device not registered"
                }))
                
                return
            
            # Get authentication token
            token = data.get("token")
            
            if token is None:
                logger.warning(f"Missing authentication token from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing authentication token"
                }))
                
                return
            
            # Validate token
            # In a real implementation, this would validate the token against a secure store
            # For this example, we'll use a simple check against the config
            valid_token = self.config.get("remote_access_token")
            
            if valid_token is None or token != valid_token:
                logger.warning(f"Invalid authentication token from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Invalid authentication token"
                }))
                
                return
            
            # Authenticate client
            client["authenticated"] = True
            
            # Add device to whitelist if not already whitelisted
            if device_id not in self.whitelisted_devices:
                self.whitelisted_devices.add(device_id)
                
                # Save device registry
                await self._save_device_registry()
                
                logger.info(f"Added device to whitelist: {device_id}")
            
            # Set as active client if no active client
            if self.active_client is None:
                self.active_client = client_id
                
                logger.info(f"Set active client: {client_id}")
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "authenticate_response",
                "authenticated": True,
                "active": self.active_client == client_id
            }))
            
            logger.info(f"Authenticated client: {client_id}")
            
        except Exception as e:
            logger.error(f"Error handling device authentication: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _handle_command(self, client_id: str, data: Dict[str, Any]):
        """
        Handle command execution.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Check if client is authenticated
            if not client.get("authenticated", False):
                logger.warning(f"Unauthenticated command from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Not authenticated"
                }))
                
                return
            
            # Get command
            command = data.get("command")
            
            if command is None:
                logger.warning(f"Missing command from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing command"
                }))
                
                return
            
            # Get command type
            command_type = command.get("type")
            
            if command_type is None:
                logger.warning(f"Missing command type from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing command type"
                }))
                
                return
            
            # Check if command is whitelisted
            if command_type not in self.command_whitelist:
                logger.warning(f"Non-whitelisted command from client {client_id}: {command_type}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": f"Command not whitelisted: {command_type}"
                }))
                
                return
            
            # Execute command
            result = await self._execute_command(command)
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "command_response",
                "command_id": command.get("id"),
                "result": result
            }))
            
            logger.info(f"Executed command from client {client_id}: {command_type}")
            
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a command.
        
        Args:
            command: Command to execute
            
        Returns:
            Command result
        """
        try:
            # Get command type
            command_type = command.get("type")
            
            # Execute command based on type
            if command_type == "search":
                # Search for information
                query = command.get("query", "")
                
                # Use assistant to search
                result = await self.assistant.process_query(query)
                
                return {
                    "success": True,
                    "data": result
                }
                
            elif command_type == "open":
                # Open application or URL
                target = command.get("target", "")
                
                # TODO: Implement application/URL opening
                
                return {
                    "success": True,
                    "message": f"Opened {target}"
                }
                
            elif command_type == "screenshot":
                # Take screenshot
                result = await self.assistant.vision_processor.capture_and_analyze_screen()
                
                return {
                    "success": result.get("success", False),
                    "data": result.get("analysis", {})
                }
                
            elif command_type == "record":
                # Record audio
                duration = command.get("duration", 5.0)
                
                # Record audio
                audio_data, sample_rate = await self.assistant.voice_processor.record_audio(duration)
                
                if audio_data is None:
                    return {
                        "success": False,
                        "error": "Failed to record audio"
                    }
                
                # Recognize speech
                result = await self.assistant.voice_processor.recognize_speech(audio_data, sample_rate)
                
                return {
                    "success": result.get("success", False),
                    "text": result.get("text", ""),
                    "engine": result.get("engine", "")
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported command type: {command_type}"
                }
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_query(self, client_id: str, data: Dict[str, Any]):
        """
        Handle query processing.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Check if client is authenticated
            if not client.get("authenticated", False):
                logger.warning(f"Unauthenticated query from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Not authenticated"
                }))
                
                return
            
            # Get query
            query = data.get("query")
            
            if query is None:
                logger.warning(f"Missing query from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing query"
                }))
                
                return
            
            # Process query
            result = await self.assistant.process_query(query)
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "query_response",
                "query_id": data.get("id"),
                "result": result
            }))
            
            logger.info(f"Processed query from client {client_id}: {query}")
            
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _handle_voice(self, client_id: str, data: Dict[str, Any]):
        """
        Handle voice input processing.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Check if client is authenticated
            if not client.get("authenticated", False):
                logger.warning(f"Unauthenticated voice input from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Not authenticated"
                }))
                
                return
            
            # Get audio data
            audio_data = data.get("audio")
            
            if audio_data is None:
                logger.warning(f"Missing audio data from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing audio data"
                }))
                
                return
            
            # Get sample rate
            sample_rate = data.get("sample_rate", 16000)
            
            # Decode audio data
            import base64
            decoded_audio = base64.b64decode(audio_data)
            
            # Recognize speech
            result = await self.assistant.voice_processor.recognize_speech(decoded_audio, sample_rate)
            
            if not result.get("success", False):
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "voice_response",
                    "voice_id": data.get("id"),
                    "success": False,
                    "error": result.get("error", "Failed to recognize speech")
                }))
                
                return
            
            # Get recognized text
            text = result.get("text", "")
            
            # Process query
            query_result = await self.assistant.process_query(text)
            
            # Generate response
            response_text = query_result.get("response", "")
            
            # Synthesize speech
            audio_data, audio_sample_rate = await self.assistant.voice_processor.synthesize_speech(response_text)
            
            if audio_data is None:
                # Send text-only response
                await client["websocket"].send(json.dumps({
                    "type": "voice_response",
                    "voice_id": data.get("id"),
                    "success": True,
                    "text": text,
                    "response": response_text
                }))
                
                return
            
            # Encode audio data
            encoded_audio = base64.b64encode(audio_data).decode("utf-8")
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "voice_response",
                "voice_id": data.get("id"),
                "success": True
               "success": True,
                "text": text,
                "response": response_text,
                "audio": encoded_audio,
                "sample_rate": audio_sample_rate
            }))
            
            logger.info(f"Processed voice input from client {client_id}")
            
        except Exception as e:
            logger.error(f"Error handling voice input: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _handle_visual(self, client_id: str, data: Dict[str, Any]):
        """
        Handle visual input processing.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Check if client is authenticated
            if not client.get("authenticated", False):
                logger.warning(f"Unauthenticated visual input from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Not authenticated"
                }))
                
                return
            
            # Get image data
            image_data = data.get("image")
            
            if image_data is None:
                logger.warning(f"Missing image data from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Missing image data"
                }))
                
                return
            
            # Decode image data
            import base64
            import io
            from PIL import Image
            
            decoded_image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded_image))
            
            # Analyze image
            result = await self.assistant.vision_processor.analyze_image(image)
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "visual_response",
                "visual_id": data.get("id"),
                "result": result
            }))
            
            logger.info(f"Processed visual input from client {client_id}")
            
        except Exception as e:
            logger.error(f"Error handling visual input: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _handle_switch(self, client_id: str, data: Dict[str, Any]):
        """
        Handle device switching.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Check if client is authenticated
            if not client.get("authenticated", False):
                logger.warning(f"Unauthenticated switch request from client {client_id}")
                
                # Send error response
                await client["websocket"].send(json.dumps({
                    "type": "error",
                    "error": "Not authenticated"
                }))
                
                return
            
            # Set as active client
            self.active_client = client_id
            
            logger.info(f"Switched active client to: {client_id}")
            
            # Send response
            await client["websocket"].send(json.dumps({
                "type": "switch_response",
                "active": True
            }))
            
            # Notify other clients
            for other_id, other_client in self.clients.items():
                if other_id != client_id and other_client.get("authenticated", False):
                    try:
                        await other_client["websocket"].send(json.dumps({
                            "type": "switch_notification",
                            "active": False
                        }))
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Error handling switch request: {e}")
            
            # Send error response
            try:
                client = self.clients.get(client_id)
                
                if client is not None:
                    await client["websocket"].send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
            except:
                pass
    
    async def _handle_ping(self, client_id: str, data: Dict[str, Any]):
        """
        Handle ping.
        
        Args:
            client_id: Client ID
            data: Message data
        """
        try:
            # Get client
            client = self.clients.get(client_id)
            
            if client is None:
                logger.warning(f"Client not found: {client_id}")
                return
            
            # Send pong response
            await client["websocket"].send(json.dumps({
                "type": "pong",
                "timestamp": time.time()
            }))
            
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
    
    async def connect_to_server(self, server_url: str, device_info: Dict[str, Any], token: str) -> bool:
        """
        Connect to a remote server.
        
        Args:
            server_url: Server URL
            device_info: Device information
            token: Authentication token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already connected
            if self.is_connected:
                logger.warning("Already connected to a server")
                return False
            
            # Connect to server
            websocket = await websockets.connect(server_url)
            
            # Register device
            await websocket.send(json.dumps({
                "type": "register",
                "device": device_info
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "register_response":
                logger.error(f"Unexpected response: {data}")
                await websocket.close()
                return False
            
            # Get device ID
            device_id = data.get("device_id")
            
            if device_id is None:
                logger.error("Missing device ID in response")
                await websocket.close()
                return False
            
            # Update device info
            device_info["id"] = device_id
            
            # Authenticate
            await websocket.send(json.dumps({
                "type": "authenticate",
                "token": token
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "authenticate_response":
                logger.error(f"Unexpected response: {data}")
                await websocket.close()
                return False
            
            # Check if authenticated
            if not data.get("authenticated", False):
                logger.error("Authentication failed")
                await websocket.close()
                return False
            
            # Set connection status
            self.is_connected = True
            
            # Start message handler
            asyncio.create_task(self._handle_server_messages(websocket))
            
            logger.info(f"Connected to server: {server_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            return False
    
    async def _handle_server_messages(self, websocket: WebSocketClientProtocol):
        """
        Handle server messages.
        
        Args:
            websocket: WebSocket connection
        """
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Get message type
                    message_type = data.get("type")
                    
                    if message_type is None:
                        logger.warning(f"Missing message type from server: {data}")
                        continue
                    
                    # Handle message based on type
                    if message_type == "command":
                        # Execute command
                        await self._handle_server_command(data, websocket)
                        
                    elif message_type == "query":
                        # Process query
                        await self._handle_server_query(data, websocket)
                        
                    elif message_type == "voice":
                        # Process voice input
                        await self._handle_server_voice(data, websocket)
                        
                    elif message_type == "visual":
                        # Process visual input
                        await self._handle_server_visual(data, websocket)
                        
                    elif message_type == "switch_notification":
                        # Handle switch notification
                        await self._handle_server_switch(data)
                        
                    elif message_type == "pong":
                        # Handle pong
                        pass
                        
                    elif message_type == "error":
                        # Handle error
                        logger.error(f"Error from server: {data.get('error')}")
                        
                    else:
                        logger.warning(f"Unknown message type from server: {message_type}")
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid message from server: {message}")
                    
                except Exception as e:
                    logger.error(f"Error handling server message: {e}")
            
        except Exception as e:
            logger.error(f"Error in server message handler: {e}")
            
        finally:
            # Set connection status
            self.is_connected = False
            
            logger.info("Disconnected from server")
    
    async def _handle_server_command(self, data: Dict[str, Any], websocket: WebSocketClientProtocol):
        """
        Handle server command.
        
        Args:
            data: Message data
            websocket: WebSocket connection
        """
        try:
            # Get command
            command = data.get("command")
            
            if command is None:
                logger.warning("Missing command from server")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Missing command"
                }))
                
                return
            
            # Get command type
            command_type = command.get("type")
            
            if command_type is None:
                logger.warning("Missing command type from server")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Missing command type"
                }))
                
                return
            
            # Check if command is whitelisted
            if command_type not in self.command_whitelist:
                logger.warning(f"Non-whitelisted command from server: {command_type}")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Command not whitelisted: {command_type}"
                }))
                
                return
            
            # Execute command
            result = await self._execute_command(command)
            
            # Send response
            await websocket.send(json.dumps({
                "type": "command_response",
                "command_id": command.get("id"),
                "result": result
            }))
            
            logger.info(f"Executed command from server: {command_type}")
            
        except Exception as e:
            logger.error(f"Error handling server command: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
   async def _handle_server_query(self, data: Dict[str, Any], websocket: WebSocketClientProtocol):
        """
        Handle server query.
        
        Args:
            data: Message data
            websocket: WebSocket connection
        """
        try:
            # Get query
            query = data.get("query")
            
            if query is None:
                logger.warning("Missing query from server")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Missing query"
                }))
                
                return
            
            # Process query
            result = await self.assistant.process_query(query)
            
            # Send response
            await websocket.send(json.dumps({
                "type": "query_response",
                "query_id": data.get("id"),
                "result": result
            }))
            
            logger.info(f"Processed query from server: {query}")
            
        except Exception as e:
            logger.error(f"Error handling server query: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
    
    async def _handle_server_voice(self, data: Dict[str, Any], websocket: WebSocketClientProtocol):
        """
        Handle server voice input.
        
        Args:
            data: Message data
            websocket: WebSocket connection
        """
        try:
            # Get audio data
            audio_data = data.get("audio")
            
            if audio_data is None:
                logger.warning("Missing audio data from server")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Missing audio data"
                }))
                
                return
            
            # Get sample rate
            sample_rate = data.get("sample_rate", 16000)
            
            # Decode audio data
            import base64
            decoded_audio = base64.b64decode(audio_data)
            
            # Recognize speech
            result = await self.assistant.voice_processor.recognize_speech(decoded_audio, sample_rate)
            
            if not result.get("success", False):
                # Send error response
                await websocket.send(json.dumps({
                    "type": "voice_response",
                    "voice_id": data.get("id"),
                    "success": False,
                    "error": result.get("error", "Failed to recognize speech")
                }))
                
                return
            
            # Get recognized text
            text = result.get("text", "")
            
            # Process query
            query_result = await self.assistant.process_query(text)
            
            # Generate response
            response_text = query_result.get("response", "")
            
            # Synthesize speech
            audio_data, audio_sample_rate = await self.assistant.voice_processor.synthesize_speech(response_text)
            
            if audio_data is None:
                # Send text-only response
                await websocket.send(json.dumps({
                    "type": "voice_response",
                    "voice_id": data.get("id"),
                    "success": True,
                    "text": text,
                    "response": response_text
                   "response": response_text
                }))
                
                return
            
            # Encode audio data
            encoded_audio = base64.b64encode(audio_data).decode("utf-8")
            
            # Send response
            await websocket.send(json.dumps({
                "type": "voice_response",
                "voice_id": data.get("id"),
                "success": True,
                "text": text,
                "response": response_text,
                "audio": encoded_audio,
                "sample_rate": audio_sample_rate
            }))
            
            logger.info("Processed voice input from server")
            
        except Exception as e:
            logger.error(f"Error handling server voice input: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
    
    async def _handle_server_visual(self, data: Dict[str, Any], websocket: WebSocketClientProtocol):
        """
        Handle server visual input.
        
        Args:
            data: Message data
            websocket: WebSocket connection
        """
        try:
            # Get image data
            image_data = data.get("image")
            
            if image_data is None:
                logger.warning("Missing image data from server")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Missing image data"
                }))
                
                return
            
            # Decode image data
            import base64
            import io
            from PIL import Image
            
            decoded_image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded_image))
            
            # Analyze image
            result = await self.assistant.vision_processor.analyze_image(image)
            
            # Send response
            await websocket.send(json.dumps({
                "type": "visual_response",
                "visual_id": data.get("id"),
                "result": result
            }))
            
            logger.info("Processed visual input from server")
            
        except Exception as e:
            logger.error(f"Error handling server visual input: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
    
    async def _handle_server_switch(self, data: Dict[str, Any]):
        """
        Handle server switch notification.
        
        Args:
            data: Message data
        """
        try:
            # Get active status
            active = data.get("active", False)
            
            # Update active status
            if active:
                logger.info("This device is now active")
            else:
                logger.info("This device is no longer active")
            
        except Exception as e:
            logger.error(f"Error handling server switch notification: {e}")
    
    async def send_ping(self, websocket: WebSocketClientProtocol) -> bool:
        """
        Send ping to server.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Send ping
            await websocket.send(json.dumps({
                "type": "ping",
                "timestamp": time.time()
            }))
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending ping: {e}")
            return False
    
    async def send_command(self, websocket: WebSocketClientProtocol, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send command to server.
        
        Args:
            websocket: WebSocket connection
            command: Command to send
            
        Returns:
            Command result
        """
        try:
            # Generate command ID
            command_id = str(uuid.uuid4())
            
            # Send command
            await websocket.send(json.dumps({
                "type": "command",
                "command": {
                    "id": command_id,
                    **command
                }
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "command_response" or data.get("command_id") != command_id:
                logger.error(f"Unexpected response: {data}")
                
                return {
                    "success": False,
                    "error": "Unexpected response"
                }
            
            # Return result
            return data.get("result", {})
            
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_query(self, websocket: WebSocketClientProtocol, query: str) -> Dict[str, Any]:
        """
        Send query to server.
        
        Args:
            websocket: WebSocket connection
            query: Query to send
            
        Returns:
            Query result
        """
        try:
            # Generate query ID
            query_id = str(uuid.uuid4())
            
            # Send query
            await websocket.send(json.dumps({
                "type": "query",
                "id": query_id,
                "query": query
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "query_response" or data.get("query_id") != query_id:
                logger.error(f"Unexpected response: {data}")
                
                return {
                    "success": False,
                    "error": "Unexpected response"
                }
            
            # Return result
            return data.get("result", {})
            
        except Exception as e:
            logger.error(f"Error sending query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
"""
Mobile client for the AI assistant.

This module provides a mobile client for the AI assistant, allowing it to be
controlled from a mobile device.
"""

import asyncio
import base64
import json
import logging
import os
import ssl
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import websockets
from websockets.client import WebSocketClientProtocol

# Configure logging
logger = logging.getLogger(__name__)

class MobileClient:
    """
    Mobile client for the AI assistant.
    
    This class provides a mobile client for the AI assistant, allowing it to be
    controlled from a mobile device.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mobile client.
        
        Args:
            config: Configuration
self.config = config
        
        # Connection
        self.websocket = None
        self.is_connected = False
        self.is_active = False
        
        # Device information
        self.device_id = None
        self.device_info = {
            "type": "mobile",
            "name": config.get("device_name", "Mobile Device"),
            "platform": config.get("device_platform", "unknown"),
            "version": config.get("device_version", "1.0.0")
        }
        
        # Authentication token
        self.token = config.get("remote_access_token", "")
        
        # Server URL
        self.server_url = config.get("remote_access_server", "")
        
        # SSL context
        self.ssl_context = None
        
        # Message handlers
        self.message_handlers = {}
        
        # Command queue
        self.command_queue = asyncio.Queue()
        
        # Response futures
        self.response_futures = {}
    
    async def initialize(self):
        """Initialize the mobile client."""
        try:
            # Initialize SSL context
            await self._initialize_ssl_context()
            
            # Load device ID
            await self._load_device_id()
            
            # Register message handlers
            self._register_message_handlers()
            
            logger.info("Mobile client initialized")
            
        except Exception as e:
            logger.error(f"Error initializing mobile client: {e}")
    
    async def _initialize_ssl_context(self):
        """Initialize SSL context for secure connections."""
        try:
            # Check if SSL is enabled
            if self.config.get("remote_access_ssl_enabled", True):
                # SSL certificate path
                cert_path = Path(self.config.get("ssl_cert_path", "./certs/server.crt"))
                
                # Check if certificate exists
                if cert_path.exists():
                    # Create SSL context
                    self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                    self.ssl_context.load_verify_locations(cert_path)
                    
                    logger.info("SSL context initialized")
                else:
                    logger.warning("SSL certificate not found, using insecure connection")
                    self.ssl_context = None
            else:
                logger.warning("SSL disabled, using insecure connection")
                self.ssl_context = None
            
        except Exception as e:
            logger.error(f"Error initializing SSL context: {e}")
            self.ssl_context = None
    
    async def _load_device_id(self):
        """Load device ID."""
        try:
            # Device ID file
            device_path = Path(self.config.get("device_id_path", "./device"))
            device_file = device_path / "device_id.json"
            
            # Create directory if it doesn't exist
            device_path.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            if device_file.exists():
                # Load device ID
                with open(device_file, "r") as f:
                    device_data = json.load(f)
                
                # Set device ID
                self.device_id = device_data.get("device_id")
                
                if self.device_id:
                    # Update device info
                    self.device_info["id"] = self.device_id
                    
                    logger.info(f"Loaded device ID: {self.device_id}")
                else:
                    # Generate device ID
                    self.device_id = str(uuid.uuid4())
                    
                    # Update device info
                    self.device_info["id"] = self.device_id
                    
                    # Save device ID

   async def _handle_voice_response(self, data: Dict[str, Any]):
        """
        Handle voice response.
        
        Args:
            data: Message data
        """
        try:
            # Get voice ID
            voice_id = data.get("voice_id")
            
            if voice_id is None:
                logger.error("Missing voice ID in voice response")
                return
            
            # Resolve future
            future = self.response_futures.get(f"voice:{voice_id}")
            
            if future is not None:
                future.set_result(data)
                del self.response_futures[f"voice:{voice_id}"]
            
        except Exception as e:
            logger.error(f"Error handling voice response: {e}")
            
            # Resolve future with error
            voice_id = data.get("voice_id")
            
            if voice_id is not None:
                future = self.response_futures.get(f"voice:{voice_id}")
                
                if future is not None:
                    future.set_exception(e)
                    del self.response_futures[f"voice:{voice_id}"]
    
    async def _handle_visual_response(self, data: Dict[str, Any]):
        """
        Handle visual response.
        
        Args:
            data: Message data
        """
        try:
            # Get visual ID
            visual_id = data.get("visual_id")
            
            if visual_id is None:
                logger.error("Missing visual ID in visual response")
                return
            
            # Resolve future
            future = self.response_futures.get(f"visual:{visual_id}")
            
            if future is not None:
                future.set_result(data)
                del self.response_futures[f"visual:{visual_id}"]
            
        except Exception as e:
            logger.error(f"Error handling visual response: {e}")
            
            # Resolve future with error
            visual_id = data.get("visual_id")
            
            if visual_id is not None:
                future = self.response_futures.get(f"visual:{visual_id}")
                
                if future is not None:
                    future.set_exception(e)
                    del self.response_futures[f"visual:{visual_id}"]
    
    async def _handle_switch_response(self, data: Dict[str, Any]):
        """
        Handle switch response.
        
        Args:
            data: Message data
        """
        try:
            # Check if active
            active = data.get("active", False)
            
            if active:
                self.is_active = True
                logger.info("This device is now active")
            else:
                self.is_active = False
                logger.info("Failed to set this device as active")
            
            # Resolve future
            future = self.response_futures.get("switch")
            
            if future is not None:
                future.set_result(data)
                del self.response_futures["switch"]
            
        except Exception as e:
            logger.error(f"Error handling switch response: {e}")
            
            # Resolve future with error
            future = self.response_futures.get("switch")
            
            if future is not None:
                future.set_exception(e)
                del self.response_futures["switch"]
    
    async def _handle_switch_notification(self, data: Dict[str, Any]):
        """
        Handle switch notification.
        
        Args:
            data: Message data
        """
        try:
            # Check if active
            active = data.get("active", False)
            
            if active:
                self.is_active = True
                logger.info("This device is now active")
            else:
                self.is_active = False
                logger.info("This device is no longer active")
            
        except Exception as e:
            logger.error(f"Error
           logger.error(f"Error handling switch notification: {e}")
    
    async def _handle_pong(self, data: Dict[str, Any]):
        """
        Handle pong.
        
        Args:
            data: Message data
        """
        try:
            # Get timestamp
            timestamp = data.get("timestamp", 0)
            
            # Calculate latency
            latency = time.time() - timestamp
            
            logger.debug(f"Ping latency: {latency:.3f}s")
            
            # Resolve future
            future = self.response_futures.get("ping")
            
            if future is not None:
                future.set_result(data)
                del self.response_futures["ping"]
            
        except Exception as e:
            logger.error(f"Error handling pong: {e}")
            
            # Resolve future with error
            future = self.response_futures.get("ping")
            
            if future is not None:
                future.set_exception(e)
                del self.response_futures["ping"]
    
    async def _handle_error(self, data: Dict[str, Any]):
        """
        Handle error.
        
        Args:
            data: Message data
        """
        try:
            # Get error
            error = data.get("error", "Unknown error")
            
            logger.error(f"Error from server: {error}")
            
        except Exception as e:
            logger.error(f"Error handling error: {e}")
    
    async def register_device(self) -> Dict[str, Any]:
        """
        Register device with the server.
        
        Returns:
            Registration result
        """
        try:
            # Check if connected
            if not self.is_connected or self.websocket is None:
                logger.error("Not connected to server")
                
                return {
                    "success": False,
                    "error": "Not connected to server"
                }
            
            # Create future
            future = asyncio.Future()
            self.response_futures["register"] = future
            
            # Send register command
            await self.command_queue.put({
                "type": "register",
                "device": self.device_info
            })
            
            # Wait for response
            try:
                response = await asyncio.wait_for(future, timeout=10.0)
                
                return {
                    "success": True,
                    "device_id": response.get("device_id"),
                    "whitelisted": response.get("whitelisted", False)
                }
                
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for register response")
                
                # Remove future
                if "register" in self.response_futures:
                    del self.response_futures["register"]
                
                return {
                    "success": False,
                    "error": "Timeout waiting for response"
                }
            
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with the server.
        
        Returns:
            Authentication result
        """
        try:
            # Check if connected
            if not self.is_connected or self.websocket is None:
                logger.error("Not connected to server")
                
                return {
                    "success": False,
                    "error": "Not connected to server"
                }
            
            # Check token
            if not self.token:
                logger.error("Authentication token not specified")
                
                return {
                    "success": False,
                    "error": "Authentication token not specified"
                }
            
            # Create future
            future = asyncio.Future()
            self.response_futures["authenticate"] = future
            
            # Send authenticate command
            await self.command_queue.put({
                "type": "authenticate",
                "token": self.token
            })
            
            # Wait for response
            try:
                response = await asyncio.wait_for(future, timeout=10.0)
                
                # Update active status
                self.is_active = response.get("active", False)
                
                return {
                    "success": response.get("authenticated", False),
                    "active": self.is_active
                }
                
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for authenticate response")
                
                # Remove future
                if "authenticate" in self.response_futures:
                    del self.response_futures["authenticate"]
                
                return {
                    "success": False,
                    "error": "Timeout waiting for response"
                }
            
        except Exception as e:
            logger.error(f"Error authenticating: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send command to the server.
        
        Args:
            command: Command to send
            
        Returns:
            Command result
        """
        try:
            # Check if connected
            if not self.is_connected or self.websocket is None:
                logger.error("Not connected to server")
                
                return {
                    "success": False,
                    "error": "Not connected to server"
                }
            
            # Generate command ID
            command_id = str(uuid.uuid4())
            
            # Create future
            future = asyncio.Future()
            self.response_futures[f"command:{command_id}"] = future
            
            # Send command
            await self.command_queue.put({
                "type": "command",
                "command": {
                    "id": command_id,
                    **command
                }
            })
            
            # Wait for response
            try:
                response = await asyncio.wait_for(future, timeout=30.0)
                
                return response.get("result", {})
                
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for command response")
                
                # Remove future
                if f"command:{command_id}" in self.response_futures:
                    del self.response_futures[f"command:{command_id}"]
                
                return {
                    "success": False,
                    "error": "Timeout waiting for response"
                }
            
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_query(self, query: str) -> Dict[str, Any]:
        """
        Send query to the server.
        
        Args:
            query: Query to send
            
        Returns:
            Query result
        """
        try:
            # Check if connected
            if not self.is_connected or self.websocket is None:
                logger.error("Not connected to server")
                
                return {
                    "success": False,
                    "error": "Not connected to server"
                }
            
            # Generate query ID
            query_id = str(uuid.uuid4())
            
            # Create future
            future = asyncio.Future()
            self.response_futures[f"query:{query_id}"] = future
            
            # Send query
            await self.command_queue.put({
                "type": "query",
                "id": query_id,
                "query": query
            })
            
            # Wait for response
            try:
                response = await asyncio.wait_for(future, timeout=30.0)
                
                return response.get("result", {})
                
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for query response")
                
                # Remove future
                if f"query:{query_id}" in self.response_futures:
                    del self.response_futures[f"query:{query_id}"]
                
                return {
                    "success": False,
                    "error": "Timeout waiting for response"
                }
            
        except Exception as e:
            logger.error(f"Error sending query: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_voice(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Send voice input to the server.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Voice processing result
        """
        try:
            # Check if connected
            if not self.is_connected or self.websocket is None:
                logger.error("Not connected to server")
                
                return {
                    "success": False,
                    "error": "Not connected to server"
                }
            
            # Generate voice ID
            voice_id = str(uuid.uuid4())
            
            # Create future
            future = asyncio.Future()
            self.response_futures[f"voice:{voice_id}"] = future
            
            # Encode audio data
            encoded_audio = base64.b64encode(audio_data).decode("utf-8")
            
            # Send voice input
            await self.command_queue.put({
                "type": "voice",
                "id": voice_id,
                "audio": encoded_audio,
                "sample_rate": sample_rate
            })
            
            # Wait for response
            try:
                response = await asyncio.wait_for(future, timeout=30.0)
                
                return response
                
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for voice response")
                
                # Remove future
                if f"voice:{voice_id}" in self.response_futures:
                    del self.response_futures[f"voice:{voice_id}"]
                
                return {
                    "success": False,
                    "error": "Timeout waiting for response"
                }
            
        except Exception as e:
            logger.error(f"Error sending voice input: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }

"""
Example of using the mobile client.

This script demonstrates how to use the mobile client to interact with the AI assistant.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgenative_umaas.ai_assistant.mobile_client import MobileClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def main():
    """Run the example."""
    try:
        # Load configuration
        config_path = Path("config/config.json")
        
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            return
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create mobile client
        client = MobileClient(config)
        
        # Initialize client
        await client.initialize()
        
        # Connect to server
        print("Connecting to server...")
        connected = await client.connect()
        
        if not connected:
            print("Failed to connect to server")
            return
        
        print("Connected to server")
        
        # Register device
        print("Registering device...")
        result = await client.register_device()
        
        if not result.get("success", False):
            print(f"Failed to register device: {result.get('error')}")
            await client.disconnect()
            return
        
        print(f"Device registered with ID: {result.get('device_id')}")
        
        # Authenticate
        print("Authenticating...")
        result = await client.authenticate()
        
        if not result.get("success", False):
            print(f"Failed to authenticate: {result.get('error')}")
            await client.disconnect()
            return
        
        print("Authentication successful")
        
        # Check if active
        if result.get("active", False):
            print("This device is active")
        else:
            print("This device is not active")
            
            # Switch to active
            print("Switching to active...")
            active = await client.switch_active()
            
            if active:
                print("This device is now active")
            else:
                print("Failed to set this device as active")
        
        # Ping server
        print("Pinging server...")
        latency = await client.ping()
        
        if latency >= 0:
            print(f"Ping latency: {latency:.3f}s")
        else:
            print("Failed to ping server")
        
        # Send query
        print("Sending query...")
        result = await client.send_query("What's the weather like today?")
        
        print(f"Query result: {result}")
        
        # Send search command
        print("Searching...")
        result = await client.search("What is the capital of France?")
        
        print(f"Search result: {result}")
        
        # Take screenshot
        print("Taking screenshot...")
        result = await client.screenshot()
        
        print(f"Screenshot result: {result}")
        
        # Disconnect
        print("Disconnecting...")
        await client.disconnect()
        
        print("Disconnected from server")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Remote server for the AI assistant.

This module provides a remote server for the AI assistant, allowing it to be
controlled from remote devices.
"""

import asyncio
import base64
import json
import logging
import os
import ssl
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logger = logging.getLogger(__name__)

class RemoteServer:
    """
    Remote server for the AI assistant.
    
    This class provides a remote server for the AI assistant, allowing it to be
    controlled from remote devices.
    """
    
    def __init__(self, config: Dict[str, Any], assistant=None):
        """
        Initialize the remote server.
        
        Args:
            config: Configuration
            assistant: AI assistant instance
        """
        self.config = config
        self.assistant = assistant
        
        # Server
        self.server = None
        self.is_running = False
        
        # Devices
        self.devices = {}
        self.active_device = None
        
        # Whitelist
        self.device_whitelist = set(config.get("remote_access_whitelist", []))
        
        # Authentication tokens
        self.auth_tokens = set(config.get("remote_access_tokens", []))
        
        # SSL context
        self.ssl_context = None
        
        # Command handlers
        self.command_handlers = {}
    
    async def initialize(self):
        """Initialize the remote server."""
        try:
            # Initialize SSL context
            await self._initialize_ssl_context()
            
            # Register command handlers
            self._register_command_handlers()
            
            # Load devices
            await self._load_devices()
            
            logger.info("Remote server initialized")
            
        except Exception as e:
            logger.error(f"Error initializing remote server: {e}")
    
    async def _initialize_ssl_context(self):
        """Initialize SSL context for secure connections."""
        try:
            # Check if SSL is enabled
            if self.config.get("remote_access_ssl_enabled", True):
                # SSL certificate paths
                cert_path = Path(self.config.get("ssl_cert_path", "./certs/server.crt"))
                key_path = Path(self.config.get("ssl_key_path", "./certs/server.key"))
                
                # Check if certificates exist
                if cert_path.exists() and key_path.exists():
                    # Create SSL context
                    self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                    self.ssl_context.load_cert_chain(cert_path, key_path)
                    
                    logger.info("SSL context initialized")
                else:
                    logger.warning("SSL certificates not found, using insecure connection")
                    self.ssl_context = None
            else:
                logger.warning("SSL disabled, using insecure connection")
                self.ssl_context = None
            
        except Exception as e:
            logger.error(f"Error initializing SSL context: {e}")
            self.ssl_context = None
    
    def _register_command_handlers(self):
        """Register command handlers."""
        try:
            # Register handlers
            self.command_handlers = {
                "search": self._handle_search_command,
                "open": self._handle_open_command,
                "screenshot": self._handle_screenshot_command,
                "record": self._handle_record_command
            }
            
            logger.info("Registered command handlers")
            
        except Exception as e:
            logger.error(f"Error registering command handlers: {e}")
    
    async def _load_devices(self):
        """Load registered devices."""
        try:
            # Devices file
            devices_path = Path(self.config.get("devices_path", "./devices"))
            devices_file = devices_path / "devices.json"
            
            # Create directory if it doesn't exist
            devices_path.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            if devices_file.exists():
                # Load devices
                with open(devices_file, "r") as f:
                    devices_data = json.load(f)
                
                # Set devices
                self.devices = devices_data.get("devices", {})
                
                # Set active device
                self.active_device = devices_data.get("active_device")
                
                logger.info(f"Loaded {len(self.devices)} devices")
            else:
                # Initialize empty devices
                self.devices = {}
                self.active_device = None
                
                # Save devices
                await self._save_devices()
                
                logger.info("Initialized empty devices list")
            
        except Exception as e:
            logger.error(f"Error loading devices: {e}")
            
            # Initialize empty devices
            self.devices = {}
            self.active_device = None
    
    async def _save_devices(self):
        """Save registered devices."""
        try:
            # Devices file
            devices_path = Path(self.config.get("devices_path", "./devices"))
            devices_file = devices_path / "devices.json"
            
            # Create directory if it doesn't exist
            devices_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare devices data
            devices_data = {
                "devices": self.devices,
                "active_device": self.active_device
            }
            
            # Save devices
            with open(devices_file, "w") as f:
                json.dump(devices_data, f, indent=2)
            
            logger.info(f"Saved {len(self.devices)} devices")
            
        except Exception as e:
            logger.error(f"Error saving devices: {e}")
    
    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Start the remote server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            # Check if already running
            if self.is_running:
                logger.warning("Server already running")
                return
            
            # Start server
            self.server = await websockets.serve(
                self._handle_connection,
                host,
                port,
                ssl=self.ssl_context
            )
            
            # Set running status
            self.is_running = True
            
            logger.info(f"Remote server started on {host}:{port}")
            
        except Exception as e:
            logger.error(f"Error starting remote server: {e}")
            
            # Reset server
            self.server = None
            self.is_running = False
    
    async def stop(self):
        """Stop the remote server."""
        try:
            # Check if running
            if not self.is_running or self.server is None:
                logger.warning("Server not running")
                return
            
            # Stop server
            self.server.close()
            await self.server.wait_closed()
            
            # Reset server
            self.server = None
            self.is_running = False
            
            logger.info("Remote server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping remote server: {e}")
            
            # Reset server
            self.server = None
            self.is_running = False
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        try:
            # Client information
            client_info = {
                "websocket": websocket,
                "device_id": None,
                "device_info": None,
                "authenticated": False,
                "connected_at": time.time()
            }
            
            # Handle messages
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Get message type
                    message_type = data.get("type")
                    
                    if message_type is None:
                        logger.warning(f"Missing message type from client: {data}")
                        continue
                    
                    # Handle message based on type
                    if message_type == "register":
                        # Register device
                        await self._handle_register(websocket, data, client_info)
                        
                    elif message_type == "authenticate":
                        # Authenticate device
                        await self._handle_authenticate(websocket, data, client_info)
                        
                    elif message_type == "command":
                        # Execute command
                        await self._handle_command(websocket, data, client_info)
                        
                    elif message_type == "query":
                        # Process query
                        await self._handle_query(websocket, data, client_info)
                        
                    elif message_type == "voice":
                        # Process voice input
                        await self._handle_voice(websocket, data, client_info)
                        
                    elif message_type == "visual":
                        # Process visual input
                        await self._handle_visual(websocket, data, client_info)
                        
                    elif message_type == "switch":
                        # Handle switch request
                        await self._handle_switch(websocket, data, client_info)
                        
                    elif message_type == "ping":
                        # Handle ping
                        await self._handle_ping(websocket, data, client_info)
                        
                    else:
                        logger.warning(f"Unknown message type from client: {message_type}")
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid message from client: {message}")
                    
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
            
        except Exception as e:
            logger.error(f"Error in client connection handler: {e}")
            
        finally:
            # Clean up client
            if client_info.get("device_id") is not None:
                # Update device status
                device_id = client_info.get("device_id")
                
                if device_id in self.devices:
                    self.devices[device_id]["connected"] = False
                    self.devices[device_id]["last_seen"] = time.time()
                    
                    # Save devices
                    await self._save_devices()
                    
                    logger.info(f"Device disconnected: {device_id}")
   async def _handle_register(self, websocket: WebSocketServerProtocol, data: Dict[str, Any], client_info: Dict[str, Any]):
        """
        Handle device registration.
        
        Args:
            websocket: WebSocket connection
            data: Message data
            client_info: Client information
        """
        try:
            # Get device information
            device_info = data.get("device")
            
            if device_info is None:
                logger.warning("Missing device information in register request")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Missing device information"
                }))
                
                return
            
            # Get device ID
            device_id = device_info.get("id")
            
            if device_id is None:
                # Generate device ID
                device_id = str(uuid.uuid4())
                
                logger.info(f"Generated new device ID: {device_id}")
            
            # Check if device exists
            if device_id in self.devices:
                # Update device information
                self.devices[device_id].update({
                    "info": device_info,
                    "connected": True,
                    "last_seen": time.time()
                })
                
                logger.info(f"Updated existing device: {device_id}")
            else:
                # Create new device
                self.devices[device_id] = {
                    "id": device_id,
                    "info": device_info,
                    "whitelisted": device_id in self.device_whitelist,
                    "connected": True,
                    "authenticated": False,
                    "last_seen": time.time(),
                    "created_at": time.time()
                }
                
                logger.info(f"Registered new device: {device_id}")
            
            # Update client information
            client_info["device_id"] = device_id
            client_info["device_info"] = device_info
            
            # Check if device is whitelisted
            whitelisted = self.devices[device_id]["whitelisted"]
            
            # Save devices
            await self._save_devices()
            
            # Send response
            await websocket.send(json.dumps({
                "type": "register_response",
                "device_id": device_id,
                "whitelisted": whitelisted
            }))
            
        except Exception as e:
            logger.error(f"Error handling register request: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
    
    async def _handle_authenticate(self, websocket: WebSocketServerProtocol, data: Dict[str, Any], client_info: Dict[str, Any]):
        """
        Handle device authentication.
        
        Args:
            websocket: WebSocket connection
            data: Message data
            client_info: Client information
        """
        try:
            # Check if device is registered
            device_id = client_info.get("device_id")
            
            if device_id is None or device_id not in self.devices:
                logger.warning("Device not registered")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "Device not registered"
                }))
                
                return
            
            # Get authentication token
            token = data.get("token")
            
            if token is None:
                logger.warning("Missing authentication token")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "authenticate_response",
                    "authenticated": False,
                    "active": False
                }))
                
                return
            
            # Check token
            authenticated = token in self.auth_tokens
            
            if authenticated:
                # Update device authentication
                self.devices[device_id]["authenticated"] = True
                
                # Update client information
                client_info["authenticated"] = True
                
                logger.info(f"Device authenticated: {device_id}")
            else:
                logger.warning(f"Invalid authentication token for device: {device_id}")
            
            # Check if device is active
            active = self.active_device == device_id
            
            # If no active device and this device is authenticated, make it active
            if self.active_device is None and authenticated:
                self.active_device = device_id
                active = True
                
                logger.info(f"Set active device: {device_id}")
            
            # Save devices
            await self._save_devices()
            
            # Send response
            await websocket.send(json.dumps({
                "type": "authenticate_response",
                "authenticated": authenticated,
                "active": active
            }))
            
        except Exception as e:
            logger.error(f"Error handling authenticate request: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
    
    async def _handle_command(self, websocket: WebSocketServerProtocol, data: Dict[str, Any], client_info: Dict[str, Any]):
        """
        Handle command execution.
        
        Args:
            websocket: WebSocket connection
            data: Message data
            client_info: Client information
        """
        try:
            # Check if device is authenticated
            if not client_info.get("authenticated", False):
                logger.warning("Unauthenticated device attempting to execute command")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "command_response",
                    "command_id": data.get("command", {}).get("id"),
                    "result": {
                        "success": False,
                        "error": "Device not authenticated"
                    }
                }))
                
                return
            
            # Get command
            command = data.get("command")
            
            if command is None:
                logger.warning("Missing command in command request")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "command_response",
                    "command_id": None,
                    "result": {
                        "success": False,
                        "error": "Missing command"
                    }
                }))
                
                return
            
            # Get command ID
            command_id = command.get("id")
            
            if command_id is None:
                logger.warning("Missing command ID in command request")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "command_response",
                    "command_id": None,
                    "result": {
                        "success": False,
                        "error": "Missing command ID"
                    }
                }))
                
                return
            
            # Get command type
            command_type = command.get("type")
            
            if command_type is None:
                logger.warning("Missing command type in command request")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "command_response",
                    "command_id": command_id,
                    "result": {
                        "success": False,
                        "error": "Missing command type"
                    }
                }))
                
                return
            
            # Get command handler
            handler = self.command_handlers.get(command_type)
            
            if handler is None:
                logger.warning(f"No handler for command type: {command_type}")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "command_response",
                    "command_id": command_id,
                    "result": {
                        "success": False,
                        "error": f"Unknown command type: {command_type}"
                    }
                }))
                
                return
            
            # Execute command
            result = await handler(command)
            
            # Send response
            await websocket.send(json.dumps({
                "type": "command_response",
                "command_id": command_id,
                "result": result
            }))
            
            logger.info(f"Executed command: {command_type}")
            
        except Exception as e:
            logger.error(f"Error handling command request: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "command_response",
                    "command_id": data.get("command", {}).get("id"),
                    "result": {
                        "success": False,
                        "error": str(e)
                    }
                }))
            except:
                pass
    
    async def _handle_query(self, websocket: WebSocketServerProtocol, data: Dict[str, Any], client_info: Dict[str, Any]):
        """
        Handle query processing.
        
        Args:
            websocket: WebSocket connection
            data: Message data
            client_info: Client information
        """
        try:
            # Check if device is authenticated
            if not client_info.get("authenticated", False):
                logger.warning("Unauthenticated device attempting to process query")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "query_response",
                    "query_id": data.get("id"),
                    "result": {
                        "success": False,
                        "error": "Device not authenticated"
                    }
                }))
                
                return
            
            # Get query ID
            query_id = data.get("id")
            
            if query_id is None:
                logger.warning("Missing query ID in query request")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "query_response",
                    "query_id": None,
                    "result": {
                        "success": False,
                        "error": "Missing query ID"
                    }
                }))
                
                return
            
            # Get query
            query = data.get("query")
            
            if query is None:
                logger.warning("Missing query in query request")
                
                # Send error response
                await websocket.send(json.dumps({
                    "type": "query_response",
                    "query_id": query_id,
                    "result": {
                        "success": False,
                        "error": "Missing query"
                    }
                }))
                
                return
            
            # Process query
            if self.assistant is not None:
                # Process query with assistant
                response = await self.assistant.process_text(query)
                
                # Send response
                await websocket.send(json.dumps({
                    "type": "query_response",
                    "query_id": query_id,
                    "result": {
                        "success": True,
                        "response": response
                    }
                }))
            else:
                # No assistant available
                await websocket.send(json.dumps({
                    "type": "query_response",
                    "query_id": query_id,
                    "result": {
                        "success": False,
                        "error": "Assistant not available"
                    }
                }))
            
            logger.info(f"Processed query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error handling query request: {e}")
            
            # Send error response
            try:
                await websocket.send(json.dumps({
                    "type": "query_response",
                    "query_id": data.get("id"),
                    "result": {
                        "success": False,
                        "error": str(e)
                    }
                }))
            except:
                pass
   async def _handle_search_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle search command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Get query
            query = command.get("query")
            
            if query is None:
                return {
                    "success": False,
                    "error": "Missing query"
                }
            
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has search capability
            if not hasattr(self.assistant, 'search') or not callable(getattr(self.assistant, 'search')):
                # Use process_text as fallback
                result = await self.assistant.process_text(f"Search for: {query}")
                
                return {
                    "success": True,
                    "result": result
                }
            
            # Perform search
            result = await self.assistant.search(query)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling search command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_open_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle open command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Get target
            target = command.get("target")
            
            if target is None:
                return {
                    "success": False,
                    "error": "Missing target"
                }
            
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has open capability
            if not hasattr(self.assistant, 'open_application') or not callable(getattr(self.assistant, 'open_application')):
                return {
                    "success": False,
                    "error": "Open capability not available"
                }
            
            # Open target
            result = await self.assistant.open_application(target)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling open command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_screenshot_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle screenshot command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has screenshot capability
            if not hasattr(self.assistant, 'take_screenshot') or not callable(getattr(self.assistant, 'take_screenshot')):
                return {
                    "success": False,
                    "error": "Screenshot capability not available"
                }
            
            # Take screenshot
            screenshot = await self.assistant.take_screenshot()
            
            # Convert to base64
            import io
            import base64
            
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            screenshot_b64 = base64.b64encode(buffer.
   async def _handle_search_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle search command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Get query
            query = command.get("query")
            
            if query is None:
                return {
                    "success": False,
                    "error": "Missing query"
                }
            
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has search capability
            if not hasattr(self.assistant, 'search') or not callable(getattr(self.assistant, 'search')):
                # Use process_text as fallback
                result = await self.assistant.process_text(f"Search for: {query}")
                
                return {
                    "success": True,
                    "result": result
                }
            
            # Perform search
            result = await self.assistant.search(query)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling search command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_open_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle open command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Get target
            target = command.get("target")
            
            if target is None:
                return {
                    "success": False,
                    "error": "Missing target"
                }
            
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has open capability
            if not hasattr(self.assistant, 'open_application') or not callable(getattr(self.assistant, 'open_application')):
                return {
                    "success": False,
                    "error": "Open capability not available"
                }
            
            # Open target
            result = await self.assistant.open_application(target)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling open command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_screenshot_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle screenshot command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has screenshot capability
            if not hasattr(self.assistant, 'take_screenshot') or not callable(getattr(self.assistant, 'take_screenshot')):
                return {
                    "success": False,
                    "error": "Screenshot capability not available"
                }
            
            # Take screenshot
            screenshot = await self.assistant.take_screenshot()
            
            # Convert to base64
            import io
            import base64
            
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            screenshot_b64 = base64.b64encode(buffer.
           screenshot_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return {
                "success": True,
                "image": screenshot_b64,
                "format": "png"
            }
            
        except Exception as e:
            logger.error(f"Error handling screenshot command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_record_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle record command.
        
        Args:
            command: Command data
            
        Returns:
            Command result
        """
        try:
            # Get duration
            duration = command.get("duration", 5.0)
            
            # Check if assistant is available
            if self.assistant is None:
                return {
                    "success": False,
                    "error": "Assistant not available"
                }
            
            # Check if assistant has recording capability
            if not hasattr(self.assistant, 'record_audio') or not callable(getattr(self.assistant, 'record_audio')):
                return {
                    "success": False,
                    "error": "Recording capability not available"
                }
            
            # Record audio
            audio_data, sample_rate = await self.assistant.record_audio(duration)
            
            # Convert to base64
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            
            return {
                "success": True,
                "audio": audio_b64,
                "sample_rate": sample_rate,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error handling record command: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
"""
Example of using the remote server.

This script demonstrates how to use the remote server to provide AI assistant
functionality to remote devices.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgenative_umaas.ai_assistant.remote_server import RemoteServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Mock AI Assistant for demonstration
class MockAssistant:
    """Mock AI Assistant for demonstration."""
    
    async def process_text(self, text):
        """Process text input."""
        return f"You said: {text}"
    
    async def speech_to_text(self, audio_data, sample_rate):
        """Convert speech to text."""
        # In a real implementation, this would use a speech recognition model
        return "This is a mock transcription"
    
    async def text_to_speech(self, text):
        """Convert text to speech."""
        # In a real implementation, this would use a text-to-speech model
        # For now, return an empty audio buffer with a sample rate
        return b"", 16000
    
    async def search(self, query):
        """Search for information."""
        return f"Search results for: {query}"
    
    async def open_application(self, target):
        """Open application or URL."""
        return f"Opened: {target}"
    
    async def take_screenshot(self):
        """Take screenshot."""
        # Create a simple image for demonstration
        from PIL import Image, ImageDraw
        
        # Create a blank image
        image = Image.new("RGB", (800, 600), (255, 255, 255))
        
        # Draw something on the image
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 100, 700, 500), fill=(200, 200, 200), outline=(0, 0, 0))
        draw.text((400, 300), "Mock Screenshot", fill=(0, 0, 0))
        
        return image
    
    async def record_audio(self, duration):
        """Record audio."""
        # In a real implementation, this would record audio from the microphone
        # For now, return an empty audio buffer with a sample rate
        return b"", 16000

async def main():
    """Run the example."""
    try:
        # Load configuration
        config_path = Path("config/config.json")
        
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            
            # Create a default configuration
            config = {
                "remote_access_enabled": True,
                "remote_access_host": "0.0.0.0",
                "remote_access_port": 8765,
                "remote_access_ssl_enabled": False,
                "remote_access_whitelist": [],
                "remote_access_tokens": ["test_token"],
                "devices_path": "./devices",
                "ssl_cert_path": "./certs/server.crt",
                "ssl_key_path": "./certs/server.key"
            }
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"Created default configuration: {config_path}")
        else:
            # Load configuration
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Create mock assistant
        assistant = MockAssistant()
        
        # Create remote server
        server = RemoteServer(config, assistant)
        
        # Initialize server
        await server.initialize()
        
        # Get host and port from configuration
        host = config.get("remote_access_host", "0.0.0.0")
        port = config.get("remote_access_port", 8765)
        
        # Start server
        await server.start(host, port)
        
        print(f"Remote server started on {host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        print("Stopping server...")
        
        # Stop server
        await server.stop()
        
        print("Server stopped")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Remote access integration example.

This script demonstrates how to integrate the remote server with a real AI assistant.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgenative_umaas.ai_assistant.assistant import AIAssistant
from edgenative_umaas.ai_assistant.remote_server import RemoteServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def main():
    """Run the example."""
    try:
        # Load configuration
        config_path = Path("config/config.json")
        
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            return
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create AI assistant
        assistant = AIAssistant(config)
        
        # Initialize assistant
        await assistant.initialize()
        
        # Create remote server
        server = RemoteServer(config, assistant)
        
        # Initialize server
        await server.initialize()
        
        # Get host and port from configuration
        host = config.get("remote_access_host", "0.0.0.0")
        port = config.get("remote_access_port", 8765)
        
        # Check if remote access is enabled
        if not config.get("remote_access_enabled", False):
            print("Remote access is disabled in configuration")
            return
        
        # Start server
        await server.start(host, port)
        
        print(f"Remote server started on {host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        print("Stopping server...")
        
        # Stop server
        await server.stop()
        
        # Stop assistant
        await assistant.shutdown()
        
        print("Server stopped")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Remote client example.

This script demonstrates how to use the mobile client to connect to a remote server
and perform various operations.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgenative_umaas.ai_assistant.mobile_client import MobileClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def main():
    """Run the example."""
    try:
        # Create configuration
        config = {
            "server_url": "ws://localhost:8765",
            "token": "test_token",
            "device_id_path": "./client_device",
            "device_info": {
                "name": "Test Client",
                "type": "desktop",
                "os": "Linux",
                "version": "1.0.0"
            }
        }
        
        # Create mobile client
        client = MobileClient(config)
        
        # Initialize client
        await client.initialize()
        
        # Connect to server
        print("Connecting to server...")
        connected = await client.connect()
        
        if not connected:
            print("Failed to connect to server")
            return
        
        print("Connected to server")
        
        # Register device
        print("Registering device...")
        result = await client.register_device()
        
        if not result.get("success", False):
            print(f"Failed to register device: {result.get('error')}")
            await client.disconnect()
            return
        
        print(f"Device registered with ID: {result.get('device_id')}")
        print(f"Device whitelisted: {result.get('whitelisted')}")
        
        # Authenticate
        print("Authenticating...")
        result = await client.authenticate()
        
        if not result.get("success", False):
            print(f"Failed to authenticate: {result.get('error')}")
            await client.disconnect()
            return
        
        print("Authentication successful")
        
        # Check if active
        if result.get("active", False):
            print("This device is active")
        else:
            print("This device is not active")
            
            # Switch to active
            print("Switching to active...")
            active = await client.switch_active()
            
            if active:
                print("This device is now active")
            else:
                print("Failed to set this device as active")
        
        # Ping server
        print("Pinging server...")
        latency = await client.ping()
        
        if latency >= 0:
            print(f"Ping latency: {latency:.3f}s")
        else:
            print("Failed to ping server")
        
        # Send query
        print("Sending query...")
        result = await client.send_query("Hello, how are you?")
        
        print(f"Query result: {result}")
        
        # Send search command
        print("Searching...")
        result = await client.search("What is the capital of France?")
        
        print(f"Search result: {result}")
        
        # Take screenshot
        print("Taking screenshot...")
        result = await client.screenshot()
        
        if result.get("success", False) and "image" in result:
            # Save screenshot
            image_data = base64.b64decode(result["image"])
            
            with open("screenshot.png", "wb") as f:
                f.write(image_data)
                
            print(f"Screenshot saved to: screenshot.png")
        else:
            print(f"Failed to take screenshot: {result.get('error')}")
        
        # Record audio
        print("Recording audio...")
        result = await client.record(duration=3.0)
        
        if result.get("success", False) and "audio" in result:
            # Save audio
            audio_data = base64.b64decode(result["audio"])
            
            with open("recording.wav", "wb") as f:
                f.write(audio_data)
                
            print(f"Audio saved to: recording.wav")
        else:
            print(f"Failed to record audio: {result.get('error')}")
        
        # Disconnect
        print("Disconnecting...")
        await client.disconnect()
        
        print("Disconnected from server")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Mobile app example using Kivy.

This script demonstrates how to create a simple mobile app interface using Kivy
that integrates with the mobile client to connect to a remote AI assistant.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import Kivy
os.environ['KIVY_NO_ARGS'] = '1'
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.image import Image as CoreImage
from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

# Import mobile client
from edgenative_umaas.ai_assistant.mobile_client import MobileClient

# Define the UI layout
KV = '''
<ChatMessage>:
    orientation: 'vertical'
    size_hint_y: None
    height: self.minimum_height
    padding: 10
    spacing: 5
    
    Label:
        text: root.message
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]
        halign: 'left'
        color: (0.1, 0.1, 0.1, 1) if root.is_user else (0.2, 0.6, 0.8, 1)

<AssistantApp>:
    orientation: 'vertical'
    padding: 10
    spacing: 10
    
    BoxLayout:
        size_hint_y: None
        height: 50
        spacing: 10
        
        Label:
            text: 'AI Assistant'
            font_size: 20
            size_hint_x: 0.7
        
        BoxLayout:
            size_hint_x: 0.3
            spacing: 5
            
            Label:
                text: 'Connected' if root.is_connected else 'Disconnected'
                color: (0, 1, 0, 1) if root.is_connected else (1, 0, 0, 1)
                size_hint_x: 0.6
            
            Button:
                text: 'Connect' if not root.is_connected else 'Disconnect'
                on_release: root.toggle_connection()
                size_hint_x: 0.4
    
    ScrollView:
        id: chat_scroll
        size_hint_y: 0.8
        
        BoxLayout:
            id: chat_container
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            spacing: 10
            padding: 10
    
    BoxLayout:
        size_hint_y: None
        height: 50
        spacing: 10
        
        TextInput:
            id: message_input
            hint_text: 'Type a message...'
            size_hint_x: 0.7
            multiline: False
            on_text_validate: root.send_message()
        
        Button:
            text: 'Send'
            size_hint_x: 0.15
            on_release: root.send_message()
        
        Button:
            text: 'Voice'
            size_hint_x: 0.15
            on_release: root.record_voice()
    
    BoxLayout:
        size_hint_y: None
        height: 50
        spacing: 10
        
        Button:
            text: 'Camera'
            size_hint_x: 0.25
            on_release: root.take_photo()
        
        Button:
            text: 'Screenshot'
            size_hint_x: 0.25
            on_release: root.take_screenshot()
        
        Button:
            text: 'Search'
            size_hint_x: 0.25
            on_release: root.search()
        
        Button:
            text: 'Switch'
            size_hint_x: 0.25
            on_release: root.switch_active()
'''

class ChatMessage(BoxLayout):
    """Chat message widget."""
    
    message = StringProperty('')
    is_user = BooleanProperty(False)

class AssistantApp(BoxLayout):
    """Main application widget."""
    
    is_connected = BooleanProperty(False)
    client = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        """Initialize the application."""
        super(AssistantApp, self).__init__(**kwargs)
        
        # Create event loop
        self.loop = asyncio.new_event_loop()
        
        # Create client
        self.create_client()
        
        # Start background thread for asyncio loop
        threading.Thread(target=self.run_async_loop, daemon=True).start()
    
    def create_client(self):
        """Create the mobile client."""
        # Create configuration
        config = {
            "server_url": "ws://localhost:8765",
            "token": "test_token",
            "device_id_path": "./mobile_device",
            "device_info": {
                "name": "Mobile App",
                "type": "mobile",
                "os": "Android",
                "version": "1.0.0"
            }
        }
        
        # Create mobile client
        self.client = MobileClient(config)
    
    def run_async_loop(self):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_coroutine(self, coroutine):
        """Run a coroutine in the background asyncio loop."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)
    
    def toggle_connection(self):
        """Toggle the connection state."""
        if self.is_connected:
            # Disconnect
            future = self.run_coroutine(self.disconnect())
            future.add_done_callback(lambda f: self.update_connection_state(False))
        else:
            # Connect
            future = self.run_coroutine(self.connect())
            future.add_done_callback(lambda f: self.update_connection_state(f.result()))
    
    def update_connection_state(self, connected):
        """Update the connection state."""
        def update(dt):
            self.is_connected = connected
        Clock.schedule_once(update, 0)
    
    async def connect(self):
        """Connect to the server."""
        try:
            # Initialize client
            await self.client.initialize()
            
            # Connect to server
            connected = await self.client.connect()
            
            if connected:
                # Register device
                result = await self.client.register_device()
                
                if not result.get("success", False):
                    self.add_message(f"Failed to register device: {result.get('error')}", False)
                    await self.client.disconnect()
                    return False
                
                # Authenticate
                result = await self.client.authenticate()
                
                if not result.get("success", False):
                    self.add_message(f"Failed to authenticate: {result.get('error')}", False)
                    await self.client.disconnect()
                    return False
                
                # Add welcome message
                self.add_message("Connected to AI Assistant", False)
                
                return True
            else:
                self.add_message("Failed to connect to server", False)
                return False
            
        except Exception as e:
            self.add_message(f"Error connecting: {e}", False)
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        try:
            await self.client.disconnect()
            self.add_message("Disconnected from AI Assistant", False)
        except Exception as e:
            self.add_message(f"Error disconnecting: {e}", False)
    
    def add_message(self, message, is_user=True):
        """Add a message to the chat."""
        def add(dt):
            chat_message = ChatMessage(message=message, is_user=is_user)
            self.ids.chat_container.add_widget(chat_message)
            self.ids.chat_scroll.scroll_to(chat_message)
        Clock.schedule_once(add, 0)
    
    def send_message(self):
        """Send a message to the assistant."""
        # Get message text
        message = self.ids.message_input.text.strip()
        
        if not message:
            return
        
        # Clear input
        self.ids.message_input.text = ""
        
        # Add message to chat
        self.add_message(message, True)
        
        # Check if connected
        if not self.is_connected:
            self.add_message("Not connected to server", False)
            return
        
        # Send query
        future = self.run_coroutine(self.client.send_query(message))
        future.add_done_callback(self.handle_query_response)
    
    def handle_query_response(self, future):
        """Handle query response."""
        try:
            result = future.result()
            
            if result.get("success", False):
                response = result.get("response", "No response")
                self.add_message(response, False)
            else:
                error = result.get("error", "Unknown error")
                self.add_message(f"Error: {error}", False)
                
        except Exception as e:
            self.add_message(f"Error processing response: {e}", False)
    
    def record_voice(self):
        """Record voice input."""
        # Check if connected
        if not self.is_connected:
            self.add_message("Not connected to server", False)
            return
        
        self.add_message("Recording voice...", True)
        
        # Record audio
        future = self.run_coroutine(self.client.record(duration=5.0))
        future.add_done_callback(self.handle_voice_recording)
    
    def handle_voice_recording(self, future):
        """Handle voice recording."""
        try:
            result = future.result()
            
            if result.get("success", False) and "audio" in result:
                # Get audio data
                audio_data = base64.b64decode(result["audio"])
                sample_rate = result.get("sample_rate", 16000)
                
                # Save audio to temporary file
                temp_file = "temp_recording.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio_data)
                
                # Send voice input
                future = self.run_coroutine(self.client.send_voice(audio_data, sample_rate))
                future.add_done_callback(self.handle_voice_response)
            else:
                error = result.get("error", "Unknown error")
                self.add_message(f"Error recording voice: {error}", False)
                
        except Exception as e:
            self.add_message(f"Error processing recording: {e}", False)
    
    def handle_voice_response(self, future):
        """Handle voice response."""
        try:
            result = future.result()
            
            if result.get("success", False):
                # Get recognized text
                text = result.get("text", "")
                
                if text:
                    self.add_message(f"You said: {text}", True)
                
                # Get response
                response = result.get("response", "")
                
                if response:
                    self.add_message(response, False)
                
                # Play audio response if available
                if "audio" in result:
                    audio_data = base64.b64decode(result["audio"])
                    
                    # Save audio to temporary file
                    temp_file = "temp_response.wav"
                    with open(temp_file, "wb") as f:
                        f.write(audio_data)
                    
                    # Play audio
                    sound = SoundLoader.load(temp_file)
                    if sound:
                        sound.play()
            else:
                error = result.get("error", "Unknown error")
                self.add_message(f"Error processing voice: {error}", False)
                
        except Exception as e:
            self.add_message(f"Error processing voice response: {e}", False)
    
    def take_photo(self):
        """Take a photo and send it for analysis."""
        # Check if connected
        if not self.is_connected:
            self.add_message("Not connected to server", False)
            return
        
        self.add_message("Taking photo...", True)
        
        # In a real app, this would use the camera
        # For this example, we'll create a simple image
        try:
            from PIL import Image as PILImage, ImageDraw
            
            # Create a simple image
            image = PILImage.new("RGB", (640, 480), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((100, 100, 540, 380), fill=(200, 200, 200), outline=(0, 0, 0))
            draw.text((320, 240), "Test Photo", fill=(0, 0, 0))
            
            # Convert to bytes
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
            
            # Send visual input
            future = self.run_coroutine(self.client.send_visual(image_data))
            future.add_done_callback(self.handle_visual_response)
            
        except Exception as e:
            self.add_message(f"Error taking photo: {e}", False)
    
    def handle_visual_response(self, future):
        """Handle visual response."""
        try:
            result = future.result()
            
            if result.get("success", False):
                # Get analysis
                analysis = result.get("result", {})
                
                # Display analysis
                self.add_message(f"Visual analysis: {json.dumps(analysis, indent=2)}", False)
            else:
                error = result.get("error", "Unknown error")
                self.add_message(f"Error analyzing image: {error}", False)
                
        except Exception as e:
            self.add_message(f"Error processing visual response: {e}", False)
    
    def take_screenshot(self):
        """Take a screenshot of the remote device."""
        # Check if connected
        if not self.is_connected:
            self.add_message("Not connected to server", False)
            return
        
        self.add_message("Taking screenshot...", True)
        
        # Send screenshot command
        future = self.run_coroutine(self.client.screenshot())
        future.add_done_callback(self.handle_screenshot_response)
    
    def handle_screenshot_response(self, future):
        """Handle screenshot response."""
        try:
            result = future.result()
            
            if result.get("success", False) and "image" in result:
                # Get image data
                image_data = base64.b
               # Get image data
                image_data = base64.b64decode(result["image"])
                
                # Display image
                def display_image(dt):
                    # Create image widget
                    buf = BytesIO(image_data)
                    cim = CoreImage(buf, ext='png')
                    img = Image(texture=cim.texture)
                    
                    # Add to chat
                    self.ids.chat_container.add_widget(img)
                    self.ids.chat_scroll.scroll_to(img)
                
                Clock.schedule_once(display_image, 0)
                
                self.add_message("Screenshot from remote device", False)
            else:
                error = result.get("error", "Unknown error")
                self.add_message(f"Error taking screenshot: {error}", False)
                
        except Exception as e:
            self.add_message(f"Error processing screenshot: {e}", False)
    
    def search(self):
        """Perform a search."""
        # Check if connected
        if not self.is_connected:
            self.add_message("Not connected to server", False)
            return
        
        # Get message text
        message = self.ids.message_input.text.strip()
        
        if not message:
            self.add_message("Please enter a search query", False)
            return
        
        # Clear input
        self.ids.message_input.text = ""
        
        # Add message to chat
        self.add_message(f"Searching for: {message}", True)
        
        # Send search command
        future = self.run_coroutine(self.client.search(message))
        future.add_done_callback(self.handle_search_response)
    
    def handle_search_response(self, future):
        """Handle search response."""
        try:
            result = future.result()
            
            if result.get("success", False):
                # Get search results
                search_results = result.get("result", "No results found")
                
                # Display results
                self.add_message(f"Search results: {search_results}", False)
            else:
                error = result.get("error", "Unknown error")
                self.add_message(f"Error searching: {error}", False)
                
        except Exception as e:
            self.add_message(f"Error processing search response: {e}", False)
    
    def switch_active(self):
        """Switch to active device."""
        # Check if connected
        if not self.is_connected:
            self.add_message("Not connected to server", False)
            return
        
        self.add_message("Switching to active device...", True)
        
        # Send switch command
        future = self.run_coroutine(self.client.switch_active())
        future.add_done_callback(self.handle_switch_response)
    
    def handle_switch_response(self, future):
        """Handle switch response."""
        try:
            result = future.result()
            
            if result:
                self.add_message("This device is now active", False)
            else:
                self.add_message("Failed to set this device as active", False)
                
        except Exception as e:
            self.add_message(f"Error processing switch response: {e}", False)

class MobileAppExample(App):
    """Mobile app example."""
    
    def build(self):
        """Build the application."""
        # Load KV language string
        Builder.load_string(KV)
        
        # Create and return the root widget
        return AssistantApp()

if __name__ == "__main__":
    MobileAppExample().run()
"""
Device manager web interface.

This script provides a web interface for managing remote devices connected to the AI assistant.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import Flask
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory

# Create Flask app
app = Flask(__name__)

# Devices data
devices_data = {}
active_device = None

# Load devices
def load_devices():
    """Load devices from file."""
    global devices_data, active_device
    
    try:
        # Devices file
        devices_path = Path("./devices")
        devices_file = devices_path / "devices.json"
        
        # Check if file exists
        if devices_file.exists():
            # Load devices
            with open(devices_file, "r") as f:
                data = json.load(f)
            
            # Set devices
            devices_data = data.get("devices", {})
            
            # Set active device
            active_device = data.get("active_device")
            
            logging.info(f"Loaded {len(devices_data)} devices")
        else:
            # Initialize empty devices
            devices_data = {}
            active_device = None
            
            logging.info("No devices file found")
        
    except Exception as e:
        logging.error(f"Error loading devices: {e}")
        
        # Initialize empty devices
        devices_data = {}
        active_device = None

# Save devices
def save_devices():
    """Save devices to file."""
    global devices_data, active_device
    
    try:
        # Devices file
        devices_path = Path("./devices")
        devices_file = devices_path / "devices.json"
        
        # Create directory if it doesn't exist
        devices_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare devices data
        data = {
            "devices": devices_data,
            "active_device": active_device
        }
        
        # Save devices
        with open(devices_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Saved {len(devices_data)} devices")
        
    except Exception as e:
        logging.error(f"Error saving devices: {e}")

# Create templates directory
def create_templates():
    """Create templates directory and files."""
    try:
        # Templates directory
        templates_dir = Path("./templates")
        
        # Create directory if it doesn't exist
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index.html
        index_html = templates_dir / "index.html"
        
        if not index_html.exists():
            with open(index_html, "w") as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Device Manager</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .device {
            display: flex;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .device:last-child {
            border-bottom: none;
        }
        .device-info {
            flex-grow: 1;
        }
        .device-name {
            font-weight: bold;
            font-size: 18px;
        }
        .device-details {
            color: #666;
            font-size: 14px;
        }
        .device-status {
            margin-left: 10px;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
        }
        .connected {
            background-color: #d4edda;
            color: #155724;
        }
        .disconnected {
            background-color: #f8d7da;
            color: #721c24;
        }
        .active {
            background-color: #cce5ff;
            color: #004085;
        }
        .actions {
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 8px 12px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        .btn-danger {
            background-color: #dc3545;
            color: white;
        }
        .btn-success {
            background-color: #28a745;
            color: white;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        .token-section {
            margin-top: 30px;
        }
        .token-form {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .token-input {
            flex-grow: 1;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        .token-list {
            margin-top: 10px;
        }
        .token-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 3px;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Device Manager</h1>
        
        <div class="card">
            <h2>Connected Devices</h2>
            <div id="devices-list">
                {% if devices %}
                    {% for device_id, device in devices.items() %}
                        <div class="device">
                            <div class="device-info">
                                <div class="device-name">{{ device.info.name }}</div>
                                <div class="device-details">
                                    ID: {{ device_id }}<br>
                                    Type: {{ device.info.type }}<br>
                                    OS: {{ device.info.os }}<br>
                                    Last seen: {{ device.last_seen|timestamp_to_date }}
                                </div>
                            </div>
                            <div class="device-status {% if device.connected %}connected{% else %}disconnected{% endif %}">
                                {{ "Connected" if device.connected else "Disconnected" }}
                            </div>
                            {% if device_id == active_device %}
                                <div class="device-status active">Active</div>
                            {% endif %}
                            <div class="actions">
                                {% if device_id != active_device %}
                                    <form action="/set_active" method="post">
                                        <input type="hidden" name="device_id" value="{{ device_id }}">
                                        <button type="submit" class="btn btn-primary">Set Active</button>
                                    </form>
                                {% endif %}
                                {% if device.whitelisted %}
                                    <form action="/remove_whitelist" method="post">
                                        <input type="hidden" name="device_id" value="{{ device_id }}">
                                        <button type="submit" class="btn btn-danger">Remove from Whitelist</button>
                                    </form>
                                {% else %}
                                    <form action="/add_whitelist" method="post">
                                        <input type="hidden" name="device_id" value="{{ device_id }}">
                                        <button type="submit" class="btn btn-success">Add to Whitelist</button>
                                    </form>
                                {% endif %}
                                <form action="/remove_device" method="post" onsubmit="return confirm('Are you sure you want to remove this device?');">
                                    <input type="hidden" name="device_id" value="{{ device_id }}">
                                    <button type="submit" class="btn btn-danger">Remove</button>
                                </form>
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No devices connected</p>
                {% endif %}
            </div>
        </div>
        
        <div class="card token-section">
            <h2>Authentication Tokens</h2>
            <div class="token-list">
                {% if tokens %}
                    {% for token in tokens %}
                        <div class="token-item">
                            <div>{{ token }}</div>
                            <form action="/remove_token" method="post">
                                <input type="hidden" name="token" value="{{ token }}">
                                <button type="submit" class="btn btn-danger">Remove</button>
                            </form>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No tokens added</p>
                {% endif %}
            </div>
            <div class="token-form">
                <form action="/add_token" method="post" style="display: flex; width: 100%;">
                    <input type="text" name="token" placeholder="Enter new token" class="token-input" required>
                    <button type="submit" class="btn btn-primary">Add Token</button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
""")
        
        logging.info("Created templates")
        
    except Exception as e:
        logging.error(f"Error creating templates: {e}")

# Load configuration
def load_config():
    """Load configuration from file."""
    try:
        # Config file
        config_path = Path("./config/config.json")
        
        # Check if file exists
        if config_path.exists():
            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)
            
            return config
        else:
            logging.warning("No configuration file found")
            return {}
        
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}

# Save configuration
def save_config(config):
    """Save configuration to file."""
    try:
        # Config file
        config_path = Path("./config/config.json")
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logging.info("Saved configuration")
        
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")

# Initialize
def initialize():
    """Initialize the application."""
    # Create templates
    create_templates()
    
    # Load devices
    load_devices()
    
    # Register filters
    @app.template_filter('timestamp_to_date')
    def timestamp_to_date(timestamp):
        """Convert timestamp to date string."""


