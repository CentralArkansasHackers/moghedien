import json
import os
from typing import Dict, Any, List, Optional


class BloodHoundParser:
    """Parser for BloodHound JSON data files."""

    def __init__(self):
        self.domains = {}
        self.computers = {}
        self.users = {}
        self.groups = {}
        self.ous = {}
        self.gpos = {}
        self.containers = {}

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a single BloodHound JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'data' not in data or 'meta' not in data:
            raise ValueError(f"Invalid BloodHound data format in {filepath}")
            
        return data
    
    def parse_directory(self, directory: str) -> None:
        """Parse all BloodHound JSON files in a directory."""
        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(directory, filename)
            data = self.parse_file(filepath)
            
            # Determine the type of data based on meta.type
            data_type = data.get('meta', {}).get('type', '').lower()
            
            # Store the data in the appropriate dictionary
            if data_type == 'domains':
                self._process_domains(data['data'])
            elif data_type == 'computers':
                self._process_computers(data['data'])
            elif data_type == 'users':
                self._process_users(data['data'])
            elif data_type == 'groups':
                self._process_groups(data['data'])
            elif data_type == 'ous':
                self._process_ous(data['data'])
            elif data_type == 'gpos':
                self._process_gpos(data['data'])
            elif data_type == 'containers':
                self._process_containers(data['data'])
    
    def _process_domains(self, data: List[Dict[str, Any]]) -> None:
        """Process domain data."""
        for domain in data:
            self.domains[domain['ObjectIdentifier']] = domain
    
    def _process_computers(self, data: List[Dict[str, Any]]) -> None:
        """Process computer data."""
        for computer in data:
            self.computers[computer['ObjectIdentifier']] = computer
            
    def _process_users(self, data: List[Dict[str, Any]]) -> None:
        """Process user data."""
        for user in data:
            self.users[user['ObjectIdentifier']] = user
            
    def _process_groups(self, data: List[Dict[str, Any]]) -> None:
        """Process group data."""
        for group in data:
            self.groups[group['ObjectIdentifier']] = group
            
    def _process_ous(self, data: List[Dict[str, Any]]) -> None:
        """Process OU data."""
        for ou in data:
            self.ous[ou['ObjectIdentifier']] = ou
            
    def _process_gpos(self, data: List[Dict[str, Any]]) -> None:
        """Process GPO data."""
        for gpo in data:
            self.gpos[gpo['ObjectIdentifier']] = gpo
            
    def _process_containers(self, data: List[Dict[str, Any]]) -> None:
        """Process container data."""
        for container in data:
            self.containers[container['ObjectIdentifier']] = container
    
    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get an object by its identifier from any category."""
        for collection in [self.domains, self.computers, self.users, 
                          self.groups, self.ous, self.gpos, self.containers]:
            if object_id in collection:
                return collection[object_id]
        return None
    
    def get_object_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search for objects by name across all categories."""
        results = []
        for collection in [self.domains, self.computers, self.users, 
                          self.groups, self.ous, self.gpos, self.containers]:
            for obj_id, obj in collection.items():
                obj_name = obj.get('Properties', {}).get('name', '').upper()
                if name.upper() in obj_name:
                    results.append(obj)
        return results
