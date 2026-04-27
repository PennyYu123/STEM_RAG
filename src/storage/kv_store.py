
import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class KVStore(ABC):
  
    def __init__(self):
        """Initialize key-value store."""
        self._initialized = False
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all key-value pairs."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the store and cleanup resources."""
        pass


class InMemoryKVStore(KVStore):
    """
    In-memory key-value store implementation.
    """
    
    def __init__(self):
        super().__init__()
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        self._initialized = True
        logger.info("Initialized InMemoryKVStore")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        # Check if key has expired
        if key in self.expiry:
            import time
            if time.time() > self.expiry[key]:
                # Key has expired, remove it
                self.delete(key)
                return None
        
        return self.data.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set key-value pair with optional TTL."""
        self.data[key] = value
        
        if ttl is not None:
            import time
            self.expiry[key] = time.time() + ttl
        elif key in self.expiry:
            # Remove expiry if TTL is None
            del self.expiry[key]
    
    def delete(self, key: str) -> bool:
        """Delete key-value pair."""
        existed = key in self.data
        if existed:
            del self.data[key]
            if key in self.expiry:
                del self.expiry[key]
        return existed
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        # Check expiry first
        if key in self.expiry:
            import time
            if time.time() > self.expiry[key]:
                self.delete(key)
                return False
        
        return key in self.data
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        import time
        
        # Clean up expired keys first
        expired_keys = []
        for key, expiry_time in self.expiry.items():
            if time.time() > expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        # Filter keys by pattern
        if pattern == "*":
            return list(self.data.keys())
        else:
            return [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
    
    def clear(self) -> None:
        """Clear all key-value pairs."""
        self.data.clear()
        self.expiry.clear()
        logger.info("Cleared InMemoryKVStore")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        import time
        
        # Count non-expired keys
        current_time = time.time()
        valid_keys = []
        expired_keys = 0
        
        for key, expiry_time in self.expiry.items():
            if current_time > expiry_time:
                expired_keys += 1
            else:
                valid_keys.append(key)
        
        # Add keys without expiry
        valid_keys.extend([key for key in self.data.keys() if key not in self.expiry])
        
        return {
            "total_keys": len(valid_keys),
            "expired_keys": expired_keys,
            "memory_usage_mb": len(str(self.data)) / (1024 * 1024)  # Rough estimate
        }
    
    def close(self) -> None:
        """Close the store."""
        self.clear()
        logger.info("Closed InMemoryKVStore")
    
    def save_to_file(self, filepath: str) -> None:
        """Save store to file."""
        data = {
            'data': self.data,
            'expiry': self.expiry
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved InMemoryKVStore to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Load store from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.data = data['data']
        self.expiry = data['expiry']
        
        logger.info(f"Loaded InMemoryKVStore from {filepath}")


class DiskKVStore(KVStore):
    """
    Disk-based key-value store implementation using JSON files.
    """
    
    def __init__(self, data_dir: str = "./data/kv_store"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info(f"Initialized DiskKVStore at {data_dir}")
    
    def _get_key_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Sanitize key for filesystem
        safe_key = "".join(c for c in key if c.isalnum() or c in "-_.").rstrip()
        if not safe_key:
            safe_key = "empty_key"
        
        return self.data_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        key_path = self._get_key_path(key)
        
        if not key_path.exists():
            return None
        
        try:
            with open(key_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check TTL
                if 'expiry' in data:
                    import time
                    if time.time() > data['expiry']:
                        # Key has expired, remove it
                        key_path.unlink()
                        return None
                
                return data['value']
        except Exception as e:
            logger.error(f"Error reading key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set key-value pair with optional TTL."""
        key_path = self._get_key_path(key)
        
        data = {'value': value}
        
        if ttl is not None:
            import time
            data['expiry'] = time.time() + ttl
        
        try:
            with open(key_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing key {key}: {e}")
            raise
    
    def delete(self, key: str) -> bool:
        """Delete key-value pair."""
        key_path = self._get_key_path(key)
        
        if key_path.exists():
            try:
                key_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting key {key}: {e}")
                return False
        
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        key_path = self._get_key_path(key)
        
        if not key_path.exists():
            return False
        
        # Check if key has expired
        try:
            with open(key_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if 'expiry' in data:
                    import time
                    if time.time() > data['expiry']:
                        key_path.unlink()
                        return False
                
                return True
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        import time
        
        keys = []
        
        for file_path in self.data_dir.glob("*.json"):
            key = file_path.stem
            
            # Check if key has expired
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if 'expiry' in data:
                        if time.time() > data['expiry']:
                            file_path.unlink()
                            continue
                
                if fnmatch.fnmatch(key, pattern):
                    keys.append(key)
                    
            except Exception as e:
                logger.error(f"Error reading key file {file_path}: {e}")
                continue
        
        return keys
    
    def clear(self) -> None:
        """Clear all key-value pairs."""
        for file_path in self.data_dir.glob("*.json"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info("Cleared DiskKVStore")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        import time
        
        total_files = 0
        valid_files = 0
        expired_files = 0
        total_size = 0
        
        for file_path in self.data_dir.glob("*.json"):
            total_files += 1
            total_size += file_path.stat().st_size
            
            # Check if file has expired
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if 'expiry' in data:
                        if time.time() > data['expiry']:
                            expired_files += 1
                        else:
                            valid_files += 1
                    else:
                        valid_files += 1
                        
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue
        
        return {
            "total_files": total_files,
            "valid_files": valid_files,
            "expired_files": expired_files,
            "total_size_mb": total_size / (1024 * 1024),
            "data_dir": str(self.data_dir)
        }
    
    def close(self) -> None:
        """Close the store."""
        logger.info("Closed DiskKVStore")


class RedisKVStore(KVStore):
    """
    Redis-based key-value store implementation.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,  # Use different DB than vector store
        password: Optional[str] = None,
        key_prefix: str = "stem:kv:"
    ):
        super().__init__()
        
        try:
            import redis
        except ImportError:
            raise ImportError("Redis is required for RedisKVStore. Install with: pip install redis")
        
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=True
        )
        
        self.key_prefix = key_prefix
        
        # Test connection
        try:
            self.redis_client.ping()
            self._initialized = True
            logger.info(f"Connected to Redis KV store at {host}:{port}")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        full_key = self._full_key(key)
        
        try:
            value = self.redis_client.get(full_key)
            if value is None:
                return None
            
            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set key-value pair with optional TTL."""
        full_key = self._full_key(key)
        
        # Convert value to JSON if it's not a simple type
        if isinstance(value, (dict, list, tuple)):
            value_str = json.dumps(value)
        else:
            value_str = str(value)
        
        try:
            if ttl is not None:
                self.redis_client.setex(full_key, ttl, value_str)
            else:
                self.redis_client.set(full_key, value_str)
                
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            raise
    
    def delete(self, key: str) -> bool:
        """Delete key-value pair."""
        full_key = self._full_key(key)
        
        try:
            result = self.redis_client.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._full_key(key)
        
        try:
            return bool(self.redis_client.exists(full_key))
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        full_pattern = f"{self.key_prefix}{pattern}"
        
        try:
            keys = self.redis_client.keys(full_pattern)
            # Remove prefix from keys
            return [key[len(self.key_prefix):] for key in keys]
        except Exception as e:
            logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []
    
    def clear(self) -> None:
        """Clear all key-value pairs."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            self.redis_client.delete(*keys)
        
        logger.info("Cleared RedisKVStore")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis_client.keys(pattern)
        
        return {
            "total_keys": len(keys),
            "redis_info": self.redis_client.info(),
            "key_prefix": self.key_prefix
        }
    
    def close(self) -> None:
        """Close the store."""
        self.redis_client.close()
        logger.info("Closed RedisKVStore")


def create_kv_store(
    store_type: str,
    **kwargs
) -> KVStore:

    if store_type == "memory":
        return InMemoryKVStore()
    elif store_type == "disk":
        return DiskKVStore(**kwargs)
    elif store_type == "redis":
        return RedisKVStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")