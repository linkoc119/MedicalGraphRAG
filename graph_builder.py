"""Knowledge Graph Builder for Medical Domain"""


import logging
from typing import List, Dict, Tuple
from datetime import datetime
import json
from pathlib import Path
from neo4j import GraphDatabase, exceptions
from config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    RELATIONSHIP_TYPES, DATA_DIR
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Neo4jGraphBuilder:
    """Build and manage Neo4j knowledge graph for medical data"""
    
    def __init__(self, uri: str = NEO4J_URI, username: str = NEO4J_USERNAME, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.session = None
        
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def clear_graph(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared graph database")
    
    def create_constraints(self):
        """Create unique constraints for node IDs"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Indication) REQUIRE i.indication_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.side_effect_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:ActiveIngredient) REQUIRE a.ingredient_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contraindication) REQUIRE c.contraindication_id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except exceptions.ClientError as e:
                    if "already exists" not in str(e):
                        logger.error(f"Constraint error: {e}")
        
        logger.info("Constraints created")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.drug_name_vi)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.atc_code)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Indication) ON (i.indication_name_vi)",
            "CREATE INDEX IF NOT EXISTS FOR (s:SideEffect) ON (s.side_effect_name_vi)",
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                except exceptions.ClientError as e:
                    if "already exists" not in str(e):
                        logger.error(f"Index error: {e}")
        
        logger.info("Indexes created")
    
    def create_drug_node(self, drug_data: Dict) -> bool:
        """Create a DRUG node"""
        try:
            query = """
            CREATE (d:Drug {
                drug_id: $drug_id,
                drug_name_vi: $drug_name_vi,
                drug_name_en: $drug_name_en,
                atc_code: $atc_code,
                drug_type: $drug_type,
                created_at: $created_at
            })
            RETURN d
            """
            
            with self.driver.session() as session:
                session.run(
                    query,
                    drug_id=drug_data.get('drug_id'),
                    drug_name_vi=drug_data.get('drug_name_vi'),
                    drug_name_en=drug_data.get('drug_name_en'),
                    atc_code=drug_data.get('atc_code'),
                    drug_type=drug_data.get('drug_type'),
                    created_at=datetime.now().isoformat()
                )
            return True
        except Exception as e:
            logger.error(f"Error creating drug node: {e}")
            return False
    
    def create_indication_node(self, indication_data: Dict) -> bool:
        """Create an INDICATION node"""
        try:
            query = """
            CREATE (i:Indication {
                indication_id: $indication_id,
                indication_name_vi: $indication_name_vi,
                indication_name_en: $indication_name_en,
                created_at: $created_at
            })
            RETURN i
            """
            
            with self.driver.session() as session:
                session.run(
                    query,
                    indication_id=indication_data.get('indication_id'),
                    indication_name_vi=indication_data.get('indication_name_vi'),
                    indication_name_en=indication_data.get('indication_name_en', ''),
                    created_at=datetime.now().isoformat()
                )
            return True
        except Exception as e:
            logger.error(f"Error creating indication node: {e}")
            return False
    
    def create_side_effect_node(self, side_effect_data: Dict) -> bool:
        """Create a SIDE_EFFECT node"""
        try:
            query = """
            CREATE (s:SideEffect {
                side_effect_id: $side_effect_id,
                side_effect_name_vi: $side_effect_name_vi,
                side_effect_name_en: $side_effect_name_en,
                severity: $severity,
                created_at: $created_at
            })
            RETURN s
            """
            
            with self.driver.session() as session:
                session.run(
                    query,
                    side_effect_id=side_effect_data.get('side_effect_id'),
                    side_effect_name_vi=side_effect_data.get('side_effect_name_vi'),
                    side_effect_name_en=side_effect_data.get('side_effect_name_en', ''),
                    severity=side_effect_data.get('severity', 'Unknown'),
                    created_at=datetime.now().isoformat()
                )
            return True
        except Exception as e:
            logger.error(f"Error creating side effect node: {e}")
            return False
    
    def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict = None) -> bool:
        """Create a relationship between two nodes"""
        try:
            node_pairs = {
                "HAS_INDICATION": ("Drug", "Indication"),
                "CAUSES_SIDE_EFFECT": ("Drug", "SideEffect"),
                "HAS_CONTRAINDICATION": ("Drug", "Contraindication"),
                "INTERACTS_WITH": ("Drug", "Drug"),
                "CONTAINS": ("Drug", "ActiveIngredient"),
            }
            
            source_label, target_label = node_pairs.get(rel_type, ("Node", "Node"))
            
            if properties is None:
                properties = {}
            
            query = f"""
            MATCH (source:{source_label} {{drug_id: $source_id}})
            MATCH (target:{target_label} {{drug_id: $target_id}})
            CREATE (source)-[r:{rel_type} $props]->(target)
            RETURN r
            """
            
            with self.driver.session() as session:
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    props=properties
                )
                return result.single() is not None
        except Exception as e:
            logger.debug(f"Error creating relationship: {e}")
            return False
    
    def bulk_create_drugs(self, drugs: List[Dict]) -> int:
        """Create multiple drug nodes in batch"""
        created_count = 0
        
        query = """
        UNWIND $drugs AS drug
        CREATE (d:Drug {
            drug_id: drug.drug_id,
            drug_name_vi: drug.drug_name_vi,
            drug_name_en: drug.drug_name_en,
            atc_code: drug.atc_code,
            drug_type: drug.drug_type,
            created_at: datetime()
        })
        RETURN count(d) as count
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, drugs=drugs)
                created_count = result.single()['count']
            logger.info(f"Created {created_count} drug nodes")
        except Exception as e:
            logger.error(f"Error bulk creating drugs: {e}")
        
        return created_count
    
    def create_drug_relationships(self, drug_id: str, drug_data: Dict):
        """Create relationships for a drug from extracted data"""
        
        for indication in drug_data.get('indications', [])[:5]:
            if indication:
                indication_id = indication.lower().replace(' ', '_')
                self._create_or_link_indication(drug_id, indication, indication_id)
        
        for side_effect in drug_data.get('side_effects', [])[:10]:
            if side_effect and isinstance(side_effect, dict):
                se_id = side_effect.get('name', '').lower().replace(' ', '_')
                if se_id:
                    self._create_or_link_side_effect(drug_id, side_effect)
        
        for interaction in drug_data.get('interactions', [])[:5]:
            if interaction and isinstance(interaction, dict):
                pass
    
    def _create_or_link_indication(self, drug_id: str, indication_name: str, indication_id: str):
        """Create indication node if not exists and link to drug"""
        try:
            query = """
            MERGE (i:Indication {indication_id: $indication_id})
            ON CREATE SET 
                i.indication_name_vi = $indication_name,
                i.created_at = datetime()
            WITH i
            MATCH (d:Drug {drug_id: $drug_id})
            MERGE (d)-[:HAS_INDICATION]->(i)
            """
            
            with self.driver.session() as session:
                session.run(
                    query,
                    drug_id=drug_id,
                    indication_id=indication_id,
                    indication_name=indication_name
                )
        except Exception as e:
            logger.debug(f"Error linking indication: {e}")
    
    def _create_or_link_side_effect(self, drug_id: str, side_effect_data: Dict):
        """Create side effect node if not exists and link to drug"""
        try:
            se_name = side_effect_data.get('name', '')
            se_id = se_name.lower().replace(' ', '_')
            
            query = """
            MERGE (s:SideEffect {side_effect_id: $side_effect_id})
            ON CREATE SET 
                s.side_effect_name_vi = $side_effect_name,
                s.severity = $severity,
                s.created_at = datetime()
            WITH s
            MATCH (d:Drug {drug_id: $drug_id})
            MERGE (d)-[:CAUSES_SIDE_EFFECT {severity: $severity}]->(s)
            """
            
            with self.driver.session() as session:
                session.run(
                    query,
                    drug_id=drug_id,
                    side_effect_id=se_id,
                    side_effect_name=se_name,
                    severity=side_effect_data.get('severity', 'Unknown')
                )
        except Exception as e:
            logger.debug(f"Error linking side effect: {e}")
    
    def get_drug_by_name(self, drug_name: str) -> Dict:
        """Get drug information by name"""
        query = """
        MATCH (d:Drug)
        WHERE d.drug_name_vi CONTAINS $name OR d.drug_name_en CONTAINS $name
        RETURN d {.*}
        LIMIT 5
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, name=drug_name)
                return [record['d'] for record in result]
        except Exception as e:
            logger.error(f"Error querying drug: {e}")
            return []
    
    def get_drug_indications(self, drug_id: str) -> List[Dict]:
        """Get indications for a drug"""
        query = """
        MATCH (d:Drug {drug_id: $drug_id})-[:HAS_INDICATION]->(i:Indication)
        RETURN i {.*} as indication
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, drug_id=drug_id)
                return [record['indication'] for record in result]
        except Exception as e:
            logger.error(f"Error querying indications: {e}")
            return []
    
    def get_drug_side_effects(self, drug_id: str) -> List[Dict]:
        """Get side effects for a drug"""
        query = """
        MATCH (d:Drug {drug_id: $drug_id})-[r:CAUSES_SIDE_EFFECT]->(s:SideEffect)
        RETURN s {.*} as side_effect, r.severity as severity
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, drug_id=drug_id)
                return [
                    {**record['side_effect'], 'severity': record['severity']}
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Error querying side effects: {e}")
            return []
    
    def get_drug_interactions(self, drug_id: str, depth: int = 1) -> List[Dict]:
        """Get drug interactions"""
        query = f"""
        MATCH (d1:Drug {{drug_id: $drug_id}})-[:INTERACTS_WITH]->(d2:Drug)
        RETURN d2 {{.*}} as drug
        LIMIT 10
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, drug_id=drug_id)
                return [record['drug'] for record in result]
        except Exception as e:
            logger.error(f"Error querying interactions: {e}")
            return []
    
    def graph_statistics(self) -> Dict:
        """Get graph statistics"""
        queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'drug_count': "MATCH (d:Drug) RETURN count(d) as count",
            'indication_count': "MATCH (i:Indication) RETURN count(i) as count",
            'side_effect_count': "MATCH (s:SideEffect) RETURN count(s) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
        }
        
        stats = {}
        try:
            with self.driver.session() as session:
                for stat_name, query in queries.items():
                    result = session.run(query)
                    stats[stat_name] = result.single()['count']
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
        
        return stats



if __name__ == "__main__":
    builder = Neo4jGraphBuilder()
    
    if builder.connect():
        builder.create_constraints()
        builder.create_indexes()
        
        drugs = []
        with open(Path(DATA_DIR) / "extracted_drugs.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    drugs.append(json.loads(line))
        
        logger.info(f"Loaded {len(drugs)} drugs from file")
        
        # Tạo drug nodes
        builder.bulk_create_drugs(drugs)
        
        # Tạo relationships
        for drug in drugs:
            builder.create_drug_relationships(drug.get('drug_id'), drug)
        
        # Thống kê
        stats = builder.graph_statistics()
        logger.info(f"Graph Statistics: {stats}")
        
        builder.close()
