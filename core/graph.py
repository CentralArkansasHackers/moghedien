import networkx as nx
from typing import Dict, Any, List, Optional, Set, Tuple

class ADGraph:
    """
    A graph representation of Active Directory objects and their relationships.
    Uses NetworkX for the underlying graph data structure.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_types = {}  # Maps node IDs to their types (user, computer, etc.)
        
    def build_from_bloodhound(self, parser) -> None:
        """
        Build a graph from BloodHound data using the parser.
        
        Args:
            parser: A BloodHoundParser instance with loaded data
        """
        # Add nodes for each object type
        self._add_nodes_from_collection(parser.domains, 'Domain')
        self._add_nodes_from_collection(parser.computers, 'Computer')
        self._add_nodes_from_collection(parser.users, 'User')
        self._add_nodes_from_collection(parser.groups, 'Group')
        self._add_nodes_from_collection(parser.ous, 'OU')
        self._add_nodes_from_collection(parser.gpos, 'GPO')
        self._add_nodes_from_collection(parser.containers, 'Container')
        
        # Add edges for each relationship type
        self._add_member_of_edges(parser)
        self._add_admin_edges(parser)
        self._add_acl_edges(parser)
        self._add_session_edges(parser)
        self._add_gpo_edges(parser)
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def _add_nodes_from_collection(self, collection: Dict[str, Any], node_type: str) -> None:
        """Add nodes from a collection with the specified type."""
        for obj_id, obj in collection.items():
            name = obj.get('Properties', {}).get('name', obj_id)
            self.graph.add_node(
                obj_id, 
                name=name,
                type=node_type,
                properties=obj.get('Properties', {})
            )
            self.node_types[obj_id] = node_type
    
    def _add_member_of_edges(self, parser) -> None:
        """Add edges for group membership relationships."""
        # Process group memberships
        for group_id, group in parser.groups.items():
            if 'Members' in group:
                for member in group.get('Members', []):
                    member_id = member.get('ObjectIdentifier')
                    if member_id and self.graph.has_node(member_id):
                        self.graph.add_edge(
                            member_id, 
                            group_id, 
                            relationship='MemberOf',
                            weight=1  # Base weight for membership
                        )
    
    def _add_admin_edges(self, parser) -> None:
        """Add edges for administrative relationships."""
        # Process local admin rights
        for computer_id, computer in parser.computers.items():
            for group in computer.get('LocalGroups', []):
                if group.get('Name', '').upper().endswith('ADMINISTRATORS'):
                    for member in group.get('Results', []):
                        member_id = member.get('ObjectIdentifier')
                        if member_id and self.graph.has_node(member_id):
                            self.graph.add_edge(
                                member_id,
                                computer_id,
                                relationship='AdminTo',
                                weight=1  # Base weight for admin rights
                            )
    
    def _add_acl_edges(self, parser) -> None:
        """Add edges for ACL relationships."""
        # Process ACL rights
        for collection in [parser.computers, parser.users, parser.groups, parser.domains, parser.ous]:
            for obj_id, obj in collection.items():
                for ace in obj.get('Aces', []):
                    principal_sid = ace.get('PrincipalSID')
                    if principal_sid and self.graph.has_node(principal_sid):
                        right_name = ace.get('RightName', '')
                        
                        # Add edge with ACL right
                        self.graph.add_edge(
                            principal_sid,
                            obj_id,
                            relationship=right_name,
                            weight=self._get_acl_weight(right_name)
                        )
    
    def _add_session_edges(self, parser) -> None:
        """Add edges for user sessions on computers."""
        for computer_id, computer in parser.computers.items():
            if 'Sessions' in computer and 'Results' in computer['Sessions']:
                for session in computer['Sessions'].get('Results', []):
                    user_id = session.get('ObjectIdentifier')
                    if user_id and self.graph.has_node(user_id):
                        self.graph.add_edge(
                            user_id,
                            computer_id,
                            relationship='HasSession',
                            weight=1  # Base weight for sessions
                        )
    
    def _add_gpo_edges(self, parser) -> None:
        """Add edges for GPO links."""
        # Process GPO links
        for ou_id, ou in parser.ous.items():
            for link in ou.get('Links', []):
                gpo_id = link.get('GUID')
                if gpo_id and self.graph.has_node(gpo_id):
                    self.graph.add_edge(
                        gpo_id,
                        ou_id,
                        relationship='GPLink',
                        enforced=link.get('IsEnforced', False),
                        weight=1  # Base weight for GPO links
                    )
    
    def _get_acl_weight(self, right_name: str) -> float:
        """
        Calculate edge weight based on the ACL right.
        Lower weight = more valuable/easier path.
        """
        # Define weights for different ACL rights
        if right_name in ['GenericAll', 'AllExtendedRights', 'WriteDacl', 'WriteOwner']:
            return 0.5  # High-value rights
        elif right_name in ['GenericWrite', 'AddMember', 'AddSelf']:
            return 0.7  # Medium-value rights
        else:
            return 1.0  # Standard weight
    
    def find_shortest_paths(self, source_id: str, target_id: str, max_paths: int = 3) -> List[List[str]]:
        """
        Find the shortest paths from source to target.
        
        Args:
            source_id: The starting node ID
            target_id: The target node ID
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        if not (self.graph.has_node(source_id) and self.graph.has_node(target_id)):
            return []
            
        try:
            # Use nx.shortest_simple_paths which returns paths ordered by length
            paths = list(nx.shortest_simple_paths(self.graph, source_id, target_id, weight='weight'))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_all_high_value_targets(self) -> List[str]:
        """Get all high-value targets in the graph."""
        high_value_nodes = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            properties = attrs.get('properties', {})
            if properties.get('highvalue', False):
                high_value_nodes.append(node_id)
                
        return high_value_nodes
    
    def get_attack_path_description(self, path: List[str]) -> List[Dict[str, str]]:
        """
        Generate a human-readable description of an attack path.
        
        Args:
            path: A list of node IDs representing the path
            
        Returns:
            A list of dictionaries with step descriptions
        """
        description = []
        
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i+1]
            
            source_name = self.graph.nodes[source_id].get('name', source_id)
            target_name = self.graph.nodes[target_id].get('name', target_id)
            
            edge_data = self.graph.get_edge_data(source_id, target_id)
            relationship = edge_data.get('relationship', 'Unknown')
            
            step = {
                'step': i + 1,
                'source': source_name,
                'target': target_name,
                'relationship': relationship,
                'description': f"{source_name} -> {relationship} -> {target_name}"
            }
            
            description.append(step)
            
        return description
