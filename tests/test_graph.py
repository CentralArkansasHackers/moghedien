import os
import unittest
from pathlib import Path

from moghedien.bloodhound.parser import BloodHoundParser

class TestBloodHoundParser(unittest.TestCase):
    """Test cases for the BloodHound parser."""
    
    def setUp(self):
        """Set up test case."""
        self.parser = BloodHoundParser()
        
    def test_initialization(self):
        """Test parser initialization."""
        self.assertIsInstance(self.parser, BloodHoundParser)
        self.assertEqual(len(self.parser.domains), 0)
        self.assertEqual(len(self.parser.computers), 0)
        self.assertEqual(len(self.parser.users), 0)
        self.assertEqual(len(self.parser.groups), 0)
        
    def test_parse_invalid_file(self):
        """Test parsing an invalid file."""
        # Create a temporary invalid JSON file
        with open("test_invalid.json", "w") as f:
            f.write("{invalid json}")
            
        # Test that parsing raises an exception
        with self.assertRaises(Exception):
            self.parser.parse_file("test_invalid.json")
            
        # Clean up
        if os.path.exists("test_invalid.json"):
            os.remove("test_invalid.json")
            
    def test_get_object_by_id(self):
        """Test getting an object by ID."""
        # Add a mock object to the parser
        mock_object = {"ObjectIdentifier": "test-id", "Properties": {"name": "test-name"}}
        self.parser.domains["test-id"] = mock_object
        
        # Test getting the object
        result = self.parser.get_object_by_id("test-id")
        self.assertEqual(result, mock_object)
        
        # Test getting a non-existent object
        result = self.parser.get_object_by_id("non-existent-id")
        self.assertIsNone(result)
        
    def test_get_object_by_name(self):
        """Test getting objects by name."""
        # Add mock objects to the parser
        mock_object1 = {"ObjectIdentifier": "test-id-1", "Properties": {"name": "test-name-1"}}
        mock_object2 = {"ObjectIdentifier": "test-id-2", "Properties": {"name": "test-name-2"}}
        self.parser.domains["test-id-1"] = mock_object1
        self.parser.computers["test-id-2"] = mock_object2
        
        # Test getting objects by name
        results = self.parser.get_object_by_name("test-name")
        self.assertEqual(len(results), 2)
        
        # Test getting objects by specific name
        results = self.parser.get_object_by_name("test-name-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], mock_object1)
        
        # Test getting objects by non-existent name
        results = self.parser.get_object_by_name("non-existent-name")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
