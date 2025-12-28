"""Embedding and Retrieval Module for Medical GraphRAG"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    TOP_K_VECTOR_RESULTS, TOP_K_GRAPH_RESULTS,
    RERANKING_TOP_K, SIMILARITY_THRESHOLD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manage Vietnamese medical text embeddings"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}, falling back to MiniLM")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts in batch"""
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return np.zeros((len(texts), self.dimension))
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def top_k_similar(self, query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 10) -> List[int]:
        """Find top-k most similar embeddings"""
        similarities = []
        for i, embedding in enumerate(embeddings):
            sim = self.similarity(query_embedding, embedding)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:k]]


class GraphRetriever:
    """Retrieve relevant information from knowledge graph"""
    
    def __init__(self, graph_driver):
        self.driver = graph_driver
    
    def retrieve_drugs_by_indication(self, indication: str, limit: int = TOP_K_GRAPH_RESULTS) -> List[Dict]:
        """Retrieve drugs for a given indication"""
        query = """
        MATCH (d:Drug)-[:HAS_INDICATION]->(i:Indication)
        WHERE i.indication_name_vi CONTAINS $indication
        RETURN d {.*} as drug
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, indication=indication, limit=limit)
                return [record['drug'] for record in result]
        except Exception as e:
            logger.error(f"Error retrieving drugs by indication: {e}")
            return []
    
    def retrieve_drugs_by_side_effect(self, side_effect: str, limit: int = TOP_K_GRAPH_RESULTS) -> List[Dict]:
        """Retrieve drugs that cause a given side effect"""
        query = """
        MATCH (d:Drug)-[:CAUSES_SIDE_EFFECT]->(s:SideEffect)
        WHERE s.side_effect_name_vi CONTAINS $side_effect
        RETURN d {.*} as drug
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, side_effect=side_effect, limit=limit)
                return [record['drug'] for record in result]
        except Exception as e:
            logger.error(f"Error retrieving drugs by side effect: {e}")
            return []
    
    def retrieve_drug_interactions(self, drug_id: str, limit: int = TOP_K_GRAPH_RESULTS) -> List[Dict]:
        """Retrieve drugs that interact with a given drug"""
        query = """
        MATCH (d1:Drug {drug_id: $drug_id})-[:INTERACTS_WITH]-(d2:Drug)
        RETURN d2 {.*} as drug
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, drug_id=drug_id, limit=limit)
                return [record['drug'] for record in result]
        except Exception as e:
            logger.error(f"Error retrieving interactions: {e}")
            return []
    
    def retrieve_context_for_drug(self, drug_id: str, depth: int = 2) -> Dict:
        """Retrieve full context for a drug"""
        query = f"""
        MATCH (d:Drug {{drug_id: $drug_id}})
        
        OPTIONAL MATCH (d)-[:HAS_INDICATION]->(i:Indication)
        OPTIONAL MATCH (d)-[:CAUSES_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (d)-[:INTERACTS_WITH]-(d2:Drug)
        
        RETURN 
            d {{.*}} as drug,
            collect(DISTINCT i {{.*}}) as indications,
            collect(DISTINCT s {{.*}}) as side_effects,
            collect(DISTINCT d2 {{.*}}) as interactions
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, drug_id=drug_id)
                record = result.single()
                if record:
                    return {
                        'drug': record['drug'],
                        'indications': [i for i in record['indications'] if i],
                        'side_effects': [s for s in record['side_effects'] if s],
                        'interactions': [d for d in record['interactions'] if d]
                    }
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
        
        return {'drug': {}, 'indications': [], 'side_effects': [], 'interactions': []}
    
    def search_drug_by_name_fuzzy(self, drug_name: str, limit: int = TOP_K_GRAPH_RESULTS) -> List[Dict]:
        """Fuzzy search for drugs by name"""
        query = """
        MATCH (d:Drug)
        WHERE d.drug_name_vi CONTAINS $name OR d.drug_name_en CONTAINS $name
        RETURN d {.*} as drug
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, name=drug_name.lower(), limit=limit)
                return [record['drug'] for record in result]
        except Exception as e:
            logger.error(f"Error fuzzy searching drugs: {e}")
            return []


class HybridRetriever:
    """Combine vector and graph-based retrieval"""
    
    def __init__(self, embedding_manager: EmbeddingManager, graph_retriever: GraphRetriever):
        self.embedding_manager = embedding_manager
        self.graph_retriever = graph_retriever
        self.reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    def hybrid_search(
        self,
        query: str,
        drugs: List[Dict],
        chunks: List[Dict],
        top_k: int = RERANKING_TOP_K
    ) -> List[Dict]:
        """Perform hybrid search with vector + graph"""
        
        query_embedding = self.embedding_manager.embed_text(query)
        chunk_embeddings = self.embedding_manager.embed_texts([c.get('content', '') for c in chunks])
        
        vector_results = []
        for idx in range(len(chunks)):
            if idx < len(chunk_embeddings):
                similarity = self.embedding_manager.similarity(query_embedding, chunk_embeddings[idx])
                if similarity > SIMILARITY_THRESHOLD:
                    vector_results.append({
                        'chunk': chunks[idx],
                        'score': similarity,
                        'source': 'vector'
                    })
        
        vector_results.sort(key=lambda x: x['score'], reverse=True)
        
        graph_results = self._graph_search(query)
        
        combined_results = vector_results[:TOP_K_VECTOR_RESULTS] + graph_results[:TOP_K_GRAPH_RESULTS]
        
        seen = set()
        unique_results = []
        for result in combined_results:
            content_key = result.get('chunk', {}).get('content', result.get('drug', {}).get('drug_id', ''))
            if content_key not in seen:
                seen.add(content_key)
                unique_results.append(result)
        
        if len(unique_results) > 0:
            texts = [
                query,
                [r.get('chunk', {}).get('content', '') or str(r.get('drug', {})) 
                 for r in unique_results]
            ]
            try:
                scores = self.reranker.predict(
                    [[query, r.get('chunk', {}).get('content', '') or str(r.get('drug', {}))]
                     for r in unique_results]
                )
                
                for i, result in enumerate(unique_results):
                    result['rerank_score'] = float(scores[i])
                
                unique_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
        
        return unique_results[:top_k]
    
    def _graph_search(self, query: str) -> List[Dict]:
        """Simple graph search based on keywords"""
        results = []
        
        keywords = query.lower().split()
        medical_keywords = [kw for kw in keywords if len(kw) > 3]
        
        for keyword in medical_keywords[:3]:
            drugs = self.graph_retriever.search_drug_by_name_fuzzy(keyword, limit=5)
            for drug in drugs:
                results.append({
                    'drug': drug,
                    'score': 0.7,
                    'source': 'graph'
                })
        
        seen = set()
        unique = []
        for r in results:
            drug_id = r.get('drug', {}).get('drug_id')
            if drug_id not in seen:
                seen.add(drug_id)
                unique.append(r)
        
        return unique


class VectorStore:
    """Simple in-memory vector store"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.vectors = {}
        self.metadata = {}
    
    def add(self, doc_id: str, text: str, metadata: Dict = None):
        """Add document to vector store"""
        embedding = self.embedding_manager.embed_text(text)
        self.vectors[doc_id] = embedding
        self.metadata[doc_id] = metadata or {'text': text}
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_embedding = self.embedding_manager.embed_text(query)
        
        results = []
        for doc_id, vector in self.vectors.items():
            similarity = self.embedding_manager.similarity(query_embedding, vector)
            results.append((doc_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_metadata(self, doc_id: str) -> Dict:
        """Get metadata for a document"""
        return self.metadata.get(doc_id, {})


if __name__ == "__main__":
    manager = EmbeddingManager()
    
    test_texts = [
        "Paracetamol trị sốt cao",
        "Thuốc kháng sinh amoxicillin",
        "Viêm đường hô hấp"
    ]
    
    embeddings = manager.embed_texts(test_texts)
    print(f"Embedded {len(embeddings)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    
    query = "Thuốc hạ sốt"
    query_emb = manager.embed_text(query)
    
    for i, text in enumerate(test_texts):
        sim = manager.similarity(query_emb, embeddings[i])
        print(f"'{text}' - Similarity: {sim:.4f}")
