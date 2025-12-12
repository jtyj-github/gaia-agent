"""
Simple file-based cache with TTL (Time To Live) support.
Used to cache tool results and reduce redundant API calls.
"""

from typing import Any, Optional
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path


class SimpleCache:
    """Simple file-based cache for tool results with TTL expiration."""

    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl

    def _get_key(self, key: str) -> str:
        """
        Generate cache key hash from string key.

        Args:
            key: Original cache key

        Returns:
            MD5 hash of the key
        """
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value if exists and not expired, None otherwise
        """
        cache_file = self.cache_dir / f"{self._get_key(key)}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check expiration
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                cache_file.unlink()  # Delete expired cache
                return None

            return data['value']
        except Exception:
            # If cache read fails, delete corrupt cache file
            if cache_file.exists():
                cache_file.unlink()
            return None

    def set(self, key: str, value: Any):
        """
        Cache a value.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
        """
        cache_file = self.cache_dir / f"{self._get_key(key)}.json"

        data = {
            'timestamp': datetime.now().isoformat(),
            'value': value
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Silently fail if cache write fails
            pass

    def clear(self):
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def clear_expired(self):
        """Remove only expired cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cached_time = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                    cache_file.unlink()
            except Exception:
                # Delete corrupt cache files
                cache_file.unlink()
