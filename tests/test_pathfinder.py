import unittest
from unittest.mock import MagicMock, patch

from moghedien.core.pathfinder import AttackPathFinder
from moghedien.core.graph import ADGraph

class TestAttackPathFinder(unittest.TestCase):
    """Test cases for the Attack Path Finder."""
    
    def setUp(self):
        """Set up test case."""
        self.graph = ADGraph()
        self.pathfinder = AttackPathFinder(self.graph)
        
    def test_initialization(self):
        """Test pathfinder initialization."""
        self.assertIsInstance(self.pathfinder, AttackPathFinder)
        self.assertEqual(self.pathfinder.graph, self.graph)
        self.assertEqual(self.pathfinder.nx_graph, self.graph.graph)
        
    def test_find_domain_admin_groups(self):
        """Test finding domain admin groups."""
        # Add a domain admin group
        self.graph.graph.add_node("DA", name="DOMAIN ADMINS", type="Group")
        self.graph.graph.add_node("OTHER", name="OTHER GROUP", type="Group")
        
        # Test finding domain admin groups
        groups = self.pathfinder._find_domain_admin_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], "DA")
        
    def test_score_path(self):
        """Test path scoring."""
        # Create a simple path
        self.graph.graph.add_node("A", name="Node A", type="User")
        self.graph.graph.add_node("B", name="Node B", type="Group")
        self.graph.graph.add_edge("A", "B", relationship="MemberOf")
        
        # Test scoring the path
        path = ["A", "B"]
        score_data = self.pathfinder.score_path(path)
        
        # Check that score_data contains expected keys
        self.assertIn("path", score_data)
        self.assertIn("steps", score_data)
        self.assertIn("length", score_data)
        self.assertIn("success_score", score_data)
        self.assertIn("stealth_score", score_data)
        self.assertIn("total_score", score_data)
        
        # Check that steps contain expected keys
        step = score_data["steps"][0]
        self.assertIn("source", step)
        self.assertIn("target", step)
        self.assertIn("relationship", step)
        self.assertIn("techniques", step)
        
    def test_get_attack_techniques(self):
        """Test getting attack techniques."""
        # Add nodes and edges for different relationship types
        self.graph.graph.add_node("USER", name="Test User", type="User")
        self.graph.graph.add_node("GROUP", name="Test Group", type="Group")
        self.graph.graph.add_node("COMPUTER", name="Test Computer", type="Computer")
        
        # Test MemberOf relationship
        techniques = self.pathfinder._get_attack_techniques("USER", "GROUP", "MemberOf")
        self.assertEqual(len(techniques), 1)
        self.assertEqual(techniques[0]["name"], "Group Membership")
        
        # Test AdminTo relationship
        techniques = self.pathfinder._get_attack_techniques("USER", "COMPUTER", "AdminTo")
        self.assertEqual(len(techniques), 2)  # Should have two techniques
        
        # Test GenericAll relationship to User
        techniques = self.pathfinder._get_attack_techniques("USER", "USER", "GenericAll")
        self.assertEqual(len(techniques), 1)
        self.assertEqual(techniques[0]["name"], "Force Password Reset")
        
        # Test GenericAll relationship to Group
        techniques = self.pathfinder._get_attack_techniques("USER", "GROUP", "GenericAll")
        self.assertEqual(len(techniques), 1)
        self.assertEqual(techniques[0]["name"], "Add to Group")
        
        # Test unknown relationship
        techniques = self.pathfinder._get_attack_techniques("USER", "GROUP", "UnknownRelationship")
        self.assertEqual(len(techniques), 1)
        self.assertEqual(techniques[0]["name"], "Exploit UnknownRelationship")


if __name__ == "__main__":
    unittest.main()
