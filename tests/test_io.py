"""Tests for I/O utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechinterp_qwen3.io import read_jsonl, sha256_file, write_json, write_jsonl


class TestWriteJson:
    """Test the write_json function."""
    
    def test_write_simple_dict(self, temp_dir):
        """Test writing a simple dictionary."""
        data = {"key": "value", "number": 42}
        path = temp_dir / "test.json"
        
        write_json(path, data)
        
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data
    
    def test_write_nested_dict(self, temp_dir):
        """Test writing nested dictionaries."""
        data = {
            "outer": {
                "inner": {
                    "value": 123
                }
            },
            "list": [1, 2, 3]
        }
        path = temp_dir / "nested.json"
        
        write_json(path, data)
        
        loaded = json.loads(path.read_text())
        assert loaded == data
    
    def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        path = temp_dir / "subdir" / "nested" / "file.json"
        data = {"test": "value"}
        
        write_json(path, data)
        
        assert path.exists()
        assert path.parent.exists()
    
    def test_formatted_output(self, temp_dir):
        """Test that output is formatted with indentation."""
        data = {"a": 1, "b": 2}
        path = temp_dir / "formatted.json"
        
        write_json(path, data)
        
        content = path.read_text()
        assert "\n" in content  # Should have newlines from formatting
        assert "  " in content  # Should have indentation


class TestReadJsonl:
    """Test the read_jsonl function."""
    
    def test_read_simple_jsonl(self, temp_dir):
        """Test reading simple JSONL file."""
        path = temp_dir / "test.jsonl"
        lines = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
        ]
        
        with path.open("w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        
        result = read_jsonl(path)
        assert result == lines
    
    def test_read_empty_lines(self, temp_dir):
        """Test that empty lines are skipped."""
        path = temp_dir / "with_empty.jsonl"
        
        with path.open("w") as f:
            f.write('{"id": 1}\n')
            f.write('\n')  # Empty line
            f.write('{"id": 2}\n')
            f.write('   \n')  # Whitespace line
            f.write('{"id": 3}\n')
        
        result = read_jsonl(path)
        assert len(result) == 3
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
        assert result[2]["id"] == 3
    
    def test_read_unicode(self, temp_dir):
        """Test reading JSONL with Unicode characters."""
        path = temp_dir / "unicode.jsonl"
        data = [
            {"text": "Hello 世界"},
            {"text": "Привет мир"},
        ]
        
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        result = read_jsonl(path)
        assert result == data


class TestWriteJsonl:
    """Test the write_jsonl function."""
    
    def test_write_simple_jsonl(self, temp_dir):
        """Test writing simple JSONL file."""
        data = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
        ]
        path = temp_dir / "output.jsonl"
        
        write_jsonl(path, data)
        
        assert path.exists()
        result = read_jsonl(path)
        assert result == data
    
    def test_write_unicode(self, temp_dir):
        """Test writing JSONL with Unicode."""
        data = [
            {"text": "Hello 世界"},
            {"emoji": "🎉"},
        ]
        path = temp_dir / "unicode_out.jsonl"
        
        write_jsonl(path, data)
        
        result = read_jsonl(path)
        assert result == data
    
    def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        path = temp_dir / "nested" / "dir" / "file.jsonl"
        data = [{"test": 1}]
        
        write_jsonl(path, data)
        
        assert path.exists()


class TestRoundTrip:
    """Test round-trip read/write operations."""
    
    def test_jsonl_round_trip(self, temp_dir, sample_prompt_data):
        """Test JSONL write then read produces same data."""
        path = temp_dir / "roundtrip.jsonl"
        
        write_jsonl(path, sample_prompt_data)
        result = read_jsonl(path)
        
        assert result == sample_prompt_data
    
    def test_json_round_trip(self, temp_dir):
        """Test JSON write then read produces same data."""
        data = {
            "model": "test-model",
            "metrics": {
                "accuracy": 0.95,
                "count": 100
            }
        }
        path = temp_dir / "roundtrip.json"
        
        write_json(path, data)
        result = json.loads(path.read_text())
        
        assert result == data


class TestSha256File:
    """Test the sha256_file function."""
    
    def test_hash_consistency(self, temp_dir):
        """Test that same content produces same hash."""
        path = temp_dir / "test.txt"
        path.write_text("Hello, World!")
        
        hash1 = sha256_file(path)
        hash2 = sha256_file(path)
        
        assert hash1 == hash2
    
    def test_different_content_different_hash(self, temp_dir):
        """Test that different content produces different hash."""
        path1 = temp_dir / "file1.txt"
        path2 = temp_dir / "file2.txt"
        
        path1.write_text("Content A")
        path2.write_text("Content B")
        
        hash1 = sha256_file(path1)
        hash2 = sha256_file(path2)
        
        assert hash1 != hash2
    
    def test_hash_format(self, temp_dir):
        """Test that hash is in correct format."""
        path = temp_dir / "test.txt"
        path.write_text("test")
        
        hash_val = sha256_file(path)
        
        # SHA256 hash should be 64 hex characters
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)
    
    def test_large_file(self, temp_dir):
        """Test hashing of larger file."""
        path = temp_dir / "large.txt"
        # Write ~2MB of data
        path.write_text("x" * (2 * 1024 * 1024))
        
        hash_val = sha256_file(path)
        
        assert len(hash_val) == 64
