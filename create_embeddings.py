"""Create embeddings for chunks and drugs - MODULAR (run separately)"""

import json
import logging
import numpy as np
from pathlib import Path
from embedding_retrieval import EmbeddingManager
from config import DATA_DIR, EMBEDDING_DIMENSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_chunk_embeddings():
    """Create and save embeddings for all chunks"""
    
    chunks_path = Path(DATA_DIR) / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error(f"❌ File not found: {chunks_path}")
        logger.error("Please run: python data_ingestion.py first")
        return False

    logger.info(f"📖 Loading chunks from {chunks_path}")
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"✅ Loaded {len(chunks)} chunks")

    logger.info("🧠 Loading embedding model...")
    embedding_manager = EmbeddingManager()
    logger.info(f"✅ Embedding model loaded (Dimension: {embedding_manager.dimension})")

    texts = [chunk.get('content', '') for chunk in chunks if chunk.get('content')]
    logger.info(f"📝 Processing {len(texts)} texts...")

    embeddings = embedding_manager.embed_texts(texts, batch_size=32)
    logger.info(f"✅ Created embeddings with shape: {embeddings.shape}")

    for i, chunk in enumerate(chunks):
        if i < len(embeddings):
            chunk['embedding'] = embeddings[i].tolist()
        else:
            chunk['embedding'] = [0.0] * EMBEDDING_DIMENSION

    embeddings_path = Path(DATA_DIR) / "chunks_with_embeddings.jsonl"
    logger.info(f"💾 Saving chunk embeddings to {embeddings_path}")
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    logger.info("=" * 60)
    logger.info("✅ Chunk embeddings created and saved successfully!")
    logger.info("=" * 60)
    logger.info(f"📊 Summary:")
    logger.info(f" - Total chunks: {len(chunks)}")
    logger.info(f" - Embedding dimension: {EMBEDDING_DIMENSION}")
    logger.info(f" - Output file: {embeddings_path}")
    logger.info(f" - Format: JSONL (1 JSON per line)")
    logger.info("=" * 60)
    
    return True


def create_drug_embeddings():
    """Create and save embeddings for all drugs"""
    
    drugs_path = Path(DATA_DIR) / "extracted_drugs.jsonl"
    if not drugs_path.exists():
        logger.error(f"❌ File not found: {drugs_path}")
        logger.error("Please run: python data_ingestion.py first")
        return False

    logger.info(f"📖 Loading drugs from {drugs_path}")
    drugs = []
    with open(drugs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                drugs.append(json.loads(line))

    logger.info(f"✅ Loaded {len(drugs)} drugs")

    logger.info("🧠 Loading embedding model...")
    embedding_manager = EmbeddingManager()
    logger.info(f"✅ Embedding model loaded (Dimension: {embedding_manager.dimension})")

    logger.info(f"📝 Building drug texts...")
    drug_texts = []
    for i, drug in enumerate(drugs):
        drug_text = f"{drug.get('drug_name_vi', '')} {drug.get('drug_name_en', '')} "
        
        if isinstance(drug.get('indications'), list):
            ind_texts = []
            for ind in drug['indications'][:3]:
                if isinstance(ind, str):
                    ind_texts.append(ind)
                elif isinstance(ind, dict):
                    ind_texts.append(ind.get('name', ''))
            drug_text += " ".join(ind_texts)
        
        drug_text += " "
        
        if isinstance(drug.get('side_effects'), list):
            se_texts = []
            for se in drug['side_effects'][:3]:
                if isinstance(se, str):
                    se_texts.append(se)
                elif isinstance(se, dict):
                    se_texts.append(se.get('name', ''))
            drug_text += " ".join(se_texts)
        
        if drug_text.strip():
            drug_texts.append(drug_text)
        else:
            drug_texts.append("")

    logger.info(f"✅ Built {len(drug_texts)} drug texts")

    logger.info(f"📊 Computing embeddings for {len(drug_texts)} drugs...")
    embeddings = embedding_manager.embed_texts(drug_texts, batch_size=32)
    logger.info(f"✅ Created embeddings with shape: {embeddings.shape}")

    embeddings_path = Path(DATA_DIR) / "drug_embeddings.jsonl"
    logger.info(f"💾 Saving drug embeddings to {embeddings_path}")
    
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        for i, drug in enumerate(drugs):
            emb_data = {
                'drug_id': drug.get('drug_id'),
                'drug_name_vi': drug.get('drug_name_vi'),
                'drug_name_en': drug.get('drug_name_en'),
                'embedding': embeddings[i].tolist() if i < len(embeddings) else [0.0] * EMBEDDING_DIMENSION
            }
            f.write(json.dumps(emb_data, ensure_ascii=False) + '\n')

    logger.info("=" * 60)
    logger.info("✅ Drug embeddings created and saved successfully!")
    logger.info("=" * 60)
    logger.info(f"📊 Summary:")
    logger.info(f" - Total drugs: {len(drugs)}")
    logger.info(f" - Embedding dimension: {EMBEDDING_DIMENSION}")
    logger.info(f" - Output file: {embeddings_path}")
    logger.info(f" - Format: JSONL (1 JSON per line)")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    import sys
    
    logger.info("🚀 Embedding Creation Tool\n")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "chunks":
            logger.info("📌 Mode: Create CHUNK embeddings only")
            success = create_chunk_embeddings()
            exit(0 if success else 1)
        
        elif mode == "drugs":
            logger.info("📌 Mode: Create DRUG embeddings only")
            success = create_drug_embeddings()
            exit(0 if success else 1)
        
        else:
            print("❌ Unknown mode. Use: python create_embeddings.py [chunks|drugs|all]")
            exit(1)
    
    else:
        # Default: create ALL
        logger.info("📌 Mode: Create ALL embeddings (chunks + drugs)\n")
        
        success_chunks = create_chunk_embeddings()
        logger.info("")
        success_drugs = create_drug_embeddings()
        
        logger.info("\n" + "=" * 60)
        if success_chunks and success_drugs:
            logger.info("✅ ALL EMBEDDINGS CREATED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. python -m chainlit run chainlit.py")
            exit(0)
        else:
            logger.error("❌ EMBEDDING CREATION FAILED!")
            logger.error("=" * 60)
            exit(1)