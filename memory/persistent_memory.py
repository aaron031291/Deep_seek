#!/usr/bin/env python3
import redis
import numpy as np
import json
import time
import threading
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import faiss
from datetime import datetime
import logging
from uuid import uuid4

from deepseek.core import Config, SecurityProvider, StorageProvider
from deepseek.core.errors import StorageError, ValidationError

logger = logging.getLogger(__name__)

class MemoryVector:
    """Vector representation of a memory with metadata."""
    
    def __init__(self, 
                 content: Any, 
                 vector: np.ndarray,
                 metadata: Dict[str, Any] = None,
                 memory_id: str = None,
                 timestamp: float = None):
        """Initialize a memory vector.
        
        Args:
            content: The actual content of the memory
            vector: Vector representation of the content
            metadata: Additional metadata about the memory
            memory_id: Unique identifier for the memory
            timestamp: Creation time of the memory
        """
        self.content = content
        self.vector = vector
        self.metadata = metadata or {}
        self.memory_id = memory_id or str(uuid4())
        self.timestamp = timestamp or time.time()
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.importance_score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "importance_score": self.importance_score
        }
   
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryVector':
        """Create memory from dictionary."""
        memory = cls(
            content=data["content"],
            vector=np.array(data["vector"]) if isinstance(data["vector"], list) else data["vector"],
            metadata=data["metadata"],
            memory_id=data["memory_id"],
            timestamp=data["timestamp"]
        )
        memory.access_count = data.get("access_count", 0)
        memory.last_accessed = data.get("last_accessed", memory.timestamp)
        memory.importance_score = data.get("importance_score", 0.0)
        return memory
    
    def access(self) -> None:
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = time.time()


class PersistentMemorySystem:
    """Distributed persistent memory system with real-time access."""
    
    def __init__(self, 
                 vector_dimension: int = 768,
                 consolidation_interval: int = 3600,
                 importance_decay_factor: float = 0.99,
                 namespace: str = "global"):
        """Initialize persistent memory system.
        
        Args:
            vector_dimension: Dimension of memory vectors
            consolidation_interval: Seconds between memory consolidation runs
            importance_decay_factor: Factor for decaying importance scores
            namespace: Memory namespace for multi-tenant isolation
        """
        self.vector_dimension = vector_dimension
        self.consolidation_interval = consolidation_interval
        self.importance_decay_factor = importance_decay_factor
        self.namespace = namespace
        
        # Initialize storage
        self.storage = StorageProvider()
        
        # Initialize vector index
        self.index = faiss.IndexFlatL2(vector_dimension)
        
        # Memory ID to position mapping
        self.id_to_position: Dict[str, int] = {}
        
        # Local cache for frequently accessed memories
        self.cache: Dict[str, MemoryVector] = {}
        self.cache_size = Config.get("MEMORY_CACHE_SIZE", 10000)
        
        # Load existing memories
        self._load_memories()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _load_memories(self) -> None:
        """Load existing memories from storage."""
        try:
            # Get all memory IDs
            memory_keys = self.storage.list(f"memory:{self.namespace}:*")
            
            if not memory_keys:
                logger.info(f"No existing memories found for namespace {self.namespace}")
                return
            
            logger.info(f"Loading {len(memory_keys)} memories from storage")
            
            # Load memories in batches
            batch_size = 1000
            vectors = []
            
            for i in range(0, len(memory_keys), batch_size):
                batch_keys = memory_keys[i:i+batch_size]
                batch_data = self.storage.read_many(batch_keys)
                
                for key, data in batch_data.items():
                    if data:
                        memory = MemoryVector.from_dict(data)
                        memory_id = memory.memory_id
                        
                        # Add to position mapping
                        position = len(vectors)
                        self.id_to_position[memory_id] = position
                        
                        # Add to vectors
                        vectors.append(memory.vector)
                        
                        # Add to cache if recently accessed
                        if len(self.cache) < self.cache_size:
                            self.cache[memory_id] = memory
            
            # Add vectors to index
            if vectors:
                vectors_array = np.array(vectors).astype('float32')
                self.index.add(vectors_array)
                
            logger.info(f"Loaded {len(vectors)} memories into vector index")
            
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            raise StorageError(f"Failed to load memories: {str(e)}")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for memory management."""
        # Memory consolidation thread
        consolidation_thread = threading.Thread(
            target=self._memory_consolidation_loop,
            daemon=True
        )
        consolidation_thread.start()
        
        # Cache management thread
        cache_thread = threading.Thread(
            target=self._cache_management_loop,
            daemon=True
        )
        cache_thread.start()
    
    def _memory_consolidation_loop(self) -> None:
        """Background loop for memory consolidation."""
        while True:
            try:
                self._consolidate_memories()
            except Exception as e:
                logger.error(f"Error in memory consolidation: {str(e)}")
            
            # Sleep until next consolidation
            time.sleep(self.consolidation_interval)
    
    def _cache_management_loop(self) -> None:
        """Background loop for cache management."""
        while True:
            try:
                # If cache exceeds size limit, remove least important items
                if len(self.cache) > self.cache_size:
                    # Sort by importance score and recency
                    sorted_items = sorted(
                        self.cache.items(),
                        key=lambda x: (x[1].importance_score, x[1].last_accessed)
                    )
                    
                    # Remove least important items
                    items_to_remove = len(self.cache) - self.cache_size
                    for i in range(items_to_remove):
                        memory_id, _ = sorted_items[i]
                        del self.cache[memory_id]
            except Exception as e:
                logger.error(f"Error in cache management: {str(e)}")
            
            # Sleep for a while
            time.sleep(60)
    
    def _consolidate_memories(self) -> None:
        """Consolidate memories based on importance and recency."""
        logger.info("Starting memory consolidation")
        
        # Get all memory IDs
        memory_keys = self.storage.list(f"memory:{self.namespace}:*")
        
        if not memory_keys:
            logger.info("No memories to consolidate")
            return
        
        # Process in batches
        batch_size = 1000
        consolidated_count = 0
        removed_count = 0
        
        for i in range(0, len(memory_keys), batch_size):
            batch_keys = memory_keys[i:i+batch_size]
            batch_data = self.storage.read_many(batch_keys)
            
            for key, data in batch_data.items():
                if not data:
                    continue
                
                memory = MemoryVector.from_dict(data)
                
                # Update importance score based on recency and access count
                time_factor = np.exp(-(time.time() - memory.last_accessed) / (3600 * 24))
                access_factor = np.log1p(memory.access_count)
                
                # Decay existing score and add new factors
                memory.importance_score = (
                    memory.importance_score * self.importance_decay_factor +
                    (time_factor * access_factor)
                )
                
                # Decide whether to keep or remove memory
                if memory.importance_score < 0.01 and time.time() - memory.last_accessed > 30 * 24 * 3600:
                    # Remove from storage
                    self.storage.delete(key)
                    
                    # Remove from index if present
                    if memory.memory_id in self.id_to_position:
                        # Note: In a production system, we'd need to rebuild the index
                        # or use a more sophisticated approach for removal
                        pass
                    
                    removed_count += 1
                else:
                    # Update in storage
                    self.storage.write(key, memory.to_dict())
                    consolidated_count += 1
        
        logger.info(f"Memory consolidation complete: {consolidated_count} consolidated, {removed_count} removed")
    
    def store(self, 
              content: Any, 
              vector: np.ndarray,
              metadata: Dict[str, Any] = None,
              memory_id: str = None) -> str:
        """Store a new memory.
        
        Args:
            content: The content to store
            vector: Vector representation of the content
            metadata: Additional metadata
            memory_id: Optional memory ID (generated if not provided)
            
        Returns:
            The memory ID
        """
        # Validate vector
        if len(vector) != self.vector_dimension:
            raise ValidationError(f"Vector dimension mismatch: expected {self.vector_dimension}, got {len(vector)}")
        
        # Create memory
        memory = MemoryVector(
            content=content,
            vector=vector,
            metadata=metadata,
            memory_id=memory_id
        )
        
        # Store in persistent storage
        key = f"memory:{self.namespace}:{memory.memory_id}"
        self.storage.write(key, memory.to_dict())
        
        # Add to index
        position = self.index.ntotal
        self.id_to_position[memory.memory_id] = position
        self.index.add(np.array([vector]).astype('float32'))
        
        # Add to cache
        if len(self.cache) < self.cache_size:
            self.cache[memory.memory_id] = memory
        
        return memory.memory_id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryVector]:
        """Retrieve a specific memory by ID.
        
        Args:
            memory_id: The memory ID to retrieve
            
        Returns:
            The memory if found, None otherwise
        """
        # Check cache first
        if memory_id in self.cache:
            memory = self.cache[memory_id]
            memory.access()
            return memory
        
        # Retrieve from storage
        key = f"memory:{self.namespace}:{memory_id}"
        data = self.storage.read(key)
        
        if not data:
            return None
        
        # Create memory object
        memory = MemoryVector.from_dict(data)
        memory.access()
        
        # Update access info in storage
        self.storage.write(key, memory.to_dict())
        
        # Add to cache
        if len(self.cache) < self.cache_size:
            self.cache[memory_id] = memory
        
        return memory
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 10,
               filter_func: Optional[callable] = None) -> List[MemoryVector]:
        """Search for similar memories.
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filter_func: Optional function to filter results
            
        Returns:
            List of similar memories
        """
        # Validate vector
        if len(query_vector) != self.vector_dimension:
            raise ValidationError(f"Vector dimension mismatch: expected {self.vector_dimension}, got {len(query_vector)}")
        
        # Search index
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), 
            k * 2  # Get more results in case some are filtered
        )
        
        # Get memory IDs for results
        results = []
        position_to_id = {pos: id for id, pos in self.id_to_position.items()}
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= self.index.ntotal:
                continue
                
            memory_id = position_to_id.get(idx)
            if not memory_id:
                continue
                
            # Retrieve memory
            memory = self.retrieve(memory_id)
            if not memory:
                continue
                
            # Apply filter if provided
            if filter_func and not filter_func(memory):
                continue
                
            # Add distance information
            memory.metadata["search_distance"] = float(distances[0][i])
            results.append(memory)
            
            # Stop if we have enough results
            if len(results) >= k:
                break
        
        return results
    
    def update(self, 
               memory_id: str, 
               content: Optional[Any] = None,
               vector: Optional[np.ndarray] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory.
        
        Args:
            memory_id: The memory ID to update
            content: New content (if None, keep existing)
            vector: New vector (if None, keep existing)
            metadata: New metadata (if None, keep existing)
            
        Returns:
            True if updated, False if not found
        """
        # Retrieve existing memory
        memory = self.retrieve(memory_id)
        if not memory:
            return False
        
        # Update fields
        if content is not None:
            memory.content = content
        
        if vector is not None:
            # Validate vector
            if len(vector) != self.vector_dimension:
                raise ValidationError(f"Vector dimension mismatch: expected {self.vector_dimension}, got {len(vector)}")
            
            memory.vector = vector
            
            # Update index
