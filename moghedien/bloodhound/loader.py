import os
import json
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from moghedien.bloodhound.parser import BloodHoundParser

logger = logging.getLogger(__name__)


class BloodHoundLoader:
    """
    Loader for BloodHound data that handles various formats and sources.
    Provides a unified interface for loading data from files or directories.
    """

    def __init__(self, parser: Optional[BloodHoundParser] = None):
        """
        Initialize the loader with an optional parser.

        Args:
            parser: A BloodHoundParser instance, or None to create a new one
        """
        self.parser = parser or BloodHoundParser()

    def load_from_directory(self, directory: str) -> BloodHoundParser:
        """
        Load all BloodHound JSON files from a directory.

        Args:
            directory: Path to directory containing BloodHound data

        Returns:
            The parser instance with loaded data
        """
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        logger.info(f"Loading BloodHound data from directory: {directory}")

        # Count of loaded files by type
        file_counts = {
            'domains': 0,
            'computers': 0,
            'users': 0,
            'groups': 0,
            'ous': 0,
            'gpos': 0,
            'containers': 0,
            'unknown': 0
        }

        # Process all JSON files in the directory
        for file_path in directory_path.glob('*.json'):
            try:
                data = self._load_json_file(file_path)

                # Check if this is a BloodHound data file
                if isinstance(data, dict) and 'data' in data and 'meta' in data:
                    data_type = data.get('meta', {}).get('type', '').lower()

                    if data_type in file_counts:
                        file_counts[data_type] += 1
                    else:
                        file_counts['unknown'] += 1
                else:
                    logger.warning(f"File does not appear to be BloodHound data: {file_path}")
                    file_counts['unknown'] += 1

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")

        # Have the parser process the directory
        self.parser.parse_directory(str(directory_path))

        # Log summary
        total_loaded = sum(file_counts.values()) - file_counts['unknown']
        logger.info(f"Loaded {total_loaded} BloodHound files:")
        for data_type, count in file_counts.items():
            if data_type != 'unknown' and count > 0:
                logger.info(f"  - {data_type.capitalize()}: {count} files")

        if file_counts['unknown'] > 0:
            logger.warning(f"Skipped {file_counts['unknown']} non-BloodHound files")

        return self.parser

    def load_from_files(self, filepaths: List[str]) -> BloodHoundParser:
        """
        Load BloodHound data from specific files.

        Args:
            filepaths: List of file paths to load

        Returns:
            The parser instance with loaded data
        """
        logger.info(f"Loading BloodHound data from {len(filepaths)} files")

        for filepath in filepaths:
            try:
                # Load and parse the file
                data = self._load_json_file(filepath)
                self._process_data_file(data, filepath)
            except Exception as e:
                logger.error(f"Error loading file {filepath}: {str(e)}")

        return self.parser

    def _load_json_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            The parsed JSON data
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            raise

    def _process_data_file(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Process a BloodHound data file and add it to the parser.
        This is a placeholder for future custom processing logic.

        Args:
            data: The parsed JSON data
            filepath: Original file path for logging
        """
        # Currently, we rely on the parser to process the files directly
        # This method can be expanded later for custom processing
        logger.debug(f"Processed file: {filepath}")