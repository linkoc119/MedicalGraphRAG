"""Chainlit Frontend with RRF (Reciprocal Rank Fusion) + BGE Reranker v2 M3 - OPTIMIZED"""


import chainlit as cl
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np
import re


# SUPPRESS LIBRARY LOGS
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from config import (
    OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
)
from data_ingestion import MedicalPDFIngestor
from graph_builder import Neo4jGraphBuilder
from embedding_retrieval import EmbeddingManager, GraphRetriever, HybridRetriever, VectorStore
from ollama_integration import OllamaLLM


try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


chat_history_store = {}


graph_builder: Optional[Neo4jGraphBuilder] = None
embedding_manager: Optional[EmbeddingManager] = None
graph_retriever: Optional[GraphRetriever] = None
reranker: Optional[FlagReranker] = None
llm: Optional[OllamaLLM] = None
chunks: List[dict] = []
chunk_embeddings: List[np.ndarray] = []
drugs: List[dict] = []
drug_embeddings: List[np.ndarray] = []


@cl.on_chat_start
async def start():
    """Initialize the chatbot on session start"""
    global graph_builder, embedding_manager, graph_retriever, reranker, llm, chunks, chunk_embeddings, drugs, drug_embeddings
    
    logger.info("🚀 Starting system initialization...")
    
    try:
        # Connect to Neo4j
        logger.info("📦 Connecting to Neo4j...")
        graph_builder = Neo4jGraphBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        if not graph_builder.connect():
            logger.error("❌ Failed to connect to Neo4j")
            await cl.Message(
                content="❌ Lỗi: Không thể kết nối đến Neo4j. Vui lòng kiểm tra cấu hình.",
                author="System"
            ).send()
            return
        logger.info("✅ Neo4j connected")
        
        # Load embedding model
        logger.info("🧠 Loading embedding model...")
        embedding_manager = EmbeddingManager()
        graph_retriever = GraphRetriever(graph_builder.driver)
        logger.info("✅ Embedding model loaded")
        
        # Load reranker
        if RERANKER_AVAILABLE:
            logger.info("Loading BGE Reranker v2 M3...")
            try:
                reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
                logger.info("✅ BGE Reranker v2 M3 loaded")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Reranking disabled.")
                reranker = None
        else:
            logger.warning("FlagEmbedding not installed")
            reranker = None
        
        # Initialize LLM
        logger.info("Initializing Qwen LLM...")
        llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        logger.info("✅ LLM initialized")
        
        # Load data
        logger.info("📖 Loading chunk embeddings...")
        chunks = []
        chunk_embeddings = []
        with open("data/chunks_with_embeddings.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks.append(chunk)
                    if 'embedding' in chunk:
                        chunk_embeddings.append(np.array(chunk['embedding'], dtype=np.float32))
                    else:
                        chunk_embeddings.append(None)
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        
        # Load drugs
        logger.info("Loading drugs...")
        drugs = []
        with open("data/extracted_drugs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    drugs.append(json.loads(line))
        logger.info(f"✅ Loaded {len(drugs)} drugs")
        
        # Load drug embeddings
        logger.info("📊 Loading pre-computed drug embeddings...")
        drug_embeddings = []
        drug_embeddings_path = "data/drug_embeddings.jsonl"
        
        if Path(drug_embeddings_path).exists():
            with open(drug_embeddings_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        emb_data = json.loads(line)
                        drug_embeddings.append(np.array(emb_data['embedding'], dtype=np.float32))
            logger.info(f"✅ Loaded {len(drug_embeddings)} drug embeddings")
        else:
            logger.error(f"Drug embeddings file not found: {drug_embeddings_path}")
            await cl.Message(
                content=f"❌ Lỗi: Không tìm thấy {drug_embeddings_path}\n\nVui lòng chạy trước:\n``````"
            ).send()
            return
        
        logger.info("✅ System initialization completed")
        
        # Welcome
        await cl.Message(
            content="""
# 🏥 Hệ thống Chatbot Y tế - Dược thư Việt Nam

Chào mừng bạn đến với hệ thống hỏi đáp thông minh về Dược thư quốc gia Việt Nam 2018.

## 📚 Tính năng:
- 🔍 Tìm kiếm thông tin thuốc
- 💊 Kiểm tra chỉ định, liều dùng
- ⚠️ Xem tác dụng phụ tiềm ẩn
- 🔗 Cảnh báo tương tác giữa các thuốc
- 📊 Hỏi đáp thông minh về y tế

## ❓ Ví dụ câu hỏi:
- "Thuốc Paracetamol dùng để trị cái gì?"
- "Amoxicillin có tác dụng phụ gì?"
- "Clarithromycin tương tác với những thuốc nào?"
- "Cách dùng thuốc kháng sinh thế nào?"

## ✅ Hệ thống đã sẵn sàng!

Bạn có thể bắt đầu hỏi về thuốc, chỉ định, tác dụng phụ, v.v.
            """,
            author="System"
        ).send()
        
        # Store in session
        cl.user_session.set("graph_builder", graph_builder)
        cl.user_session.set("embedding_manager", embedding_manager)
        cl.user_session.set("graph_retriever", graph_retriever)
        cl.user_session.set("reranker", reranker)
        cl.user_session.set("llm", llm)
        cl.user_session.set("chunks", chunks)
        cl.user_session.set("chunk_embeddings", chunk_embeddings)
        cl.user_session.set("drugs", drugs)
        cl.user_session.set("drug_embeddings", drug_embeddings)
    
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        await cl.Message(
            content=f"❌ Lỗi khởi tạo: {str(e)}",
            author="System"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    
    graph_builder = cl.user_session.get("graph_builder")
    embedding_manager = cl.user_session.get("embedding_manager")
    graph_retriever = cl.user_session.get("graph_retriever")
    reranker = cl.user_session.get("reranker")
    llm = cl.user_session.get("llm")
    chunks = cl.user_session.get("chunks", [])
    chunk_embeddings = cl.user_session.get("chunk_embeddings", [])
    drugs = cl.user_session.get("drugs", [])
    drug_embeddings = cl.user_session.get("drug_embeddings", [])
    
    if not all([graph_builder, embedding_manager, llm, graph_retriever]):
        await cl.Message(
            content="❌ Hệ thống chưa được khởi tạo đúng. Vui lòng làm mới trang."
        ).send()
        return
    
    session_id = cl.user_session.get("id")
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    chat_history = chat_history_store[session_id]
    
    msg = cl.Message(content="🔄 Đang xử lý câu hỏi...")
    await msg.send()
    
    try:
        chat_history.append({
            "role": "user",
            "content": message.content
        })
        
        if not chunk_embeddings:
            msg.content = "❌ Không có embedding. Hãy chạy create_embeddings.py trước."
            await msg.update()
            return
        
        query_embedding = embedding_manager.embed_text(message.content)
        
        vector_results = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            if chunk_emb is not None:
                sim = embedding_manager.similarity(query_embedding, chunk_emb)
                if sim > 0.3:
                    vector_results.append({
                        'type': 'chunk',
                        'chunk': chunks[i],
                        'score': sim,
                        'source': 'vector'
                    })
        
        vector_results.sort(key=lambda x: x['score'], reverse=True)
        vector_results = vector_results[:5]
        
        graph_results = _search_graph_drugs(
            query=message.content,
            graph_retriever=graph_retriever,
            graph_builder=graph_builder,
            embedding_manager=embedding_manager,
            drugs=drugs,
            drug_embeddings=drug_embeddings
        )
        
        if not vector_results and not graph_results:
            msg.content = "⚠️ Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            await msg.update()
            return
        
        msg.content = "Đang kết hợp kết quả..."
        await msg.update()
        
        all_results = _reciprocal_rank_fusion(vector_results, graph_results, k=60)
        
        # Only rerank top 5 results
        if reranker and len(all_results) > 5:
            msg.content = "Đang rerank top 5 kết quả..."
            await msg.update()
            
            top_results = all_results[:5]
            all_results = _rerank_results(
                query=message.content,
                results=top_results,
                reranker=reranker
            )
        elif reranker:
            msg.content = "Đang rerank kết quả..."
            await msg.update()
            
            all_results = _rerank_results(
                query=message.content,
                results=all_results,
                reranker=reranker
            )

        context_text = _build_hybrid_context(all_results)
        
        response = llm.generate(
            prompt=_build_prompt(message.content, context_text, chat_history),
            stream=True
        )
        
        msg.content = ""
        
        async for chunk in response:
            msg.content += chunk
            await msg.update()
        
        chat_history.append({
            "role": "assistant",
            "content": msg.content
        })
        
        if all_results:
            sources_text = _format_sources(all_results)
            await cl.Message(
                content=f"📚 **Nguồn tham khảo:**\n{sources_text}",
                author="System"
            ).send()
            
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        msg.content = f"❌ Lỗi: {str(e)}"
        await msg.update()


def _search_graph_drugs(query: str, graph_retriever, graph_builder, embedding_manager, drugs, drug_embeddings) -> List[dict]:
    """Search drugs from Neo4j graph - OPTIMIZED WITH LOWER THRESHOLD"""
    results = []
    found_drug_ids = set()
    
    keywords = query.lower().split()
    medical_keywords = [kw for kw in keywords if len(kw) > 3]
    
    for keyword in medical_keywords[:3]:
        drugs_found = graph_retriever.search_drug_by_name_fuzzy(keyword, limit=5)
        
        for drug in drugs_found:
            drug_id = drug.get('drug_id')
            if drug_id and drug_id not in found_drug_ids:
                found_drug_ids.add(drug_id)
                
                context = {
                    'indications': graph_builder.get_drug_indications(drug_id),
                    'side_effects': graph_builder.get_drug_side_effects(drug_id),
                    'interactions': graph_builder.get_drug_interactions(drug_id)
                }
                
                score = _calculate_graph_score(query, drug, context)
                
                results.append({
                    'type': 'drug',
                    'drug': drug,
                    'context': context,
                    'score': score,
                    'source': 'graph'
                })
    
    if drug_embeddings:
        query_embedding = embedding_manager.embed_text(query)
        
        vector_matches = []
        for i, drug_emb in enumerate(drug_embeddings):
            if drug_emb is not None and drugs[i].get('drug_id') not in found_drug_ids:
                sim = embedding_manager.similarity(query_embedding, drug_emb)
                if sim > 0.3:
                    vector_matches.append((i, sim))
        
        for i, sim in sorted(vector_matches, key=lambda x: x[1], reverse=True):
            drug_id = drugs[i].get('drug_id')
            found_drug_ids.add(drug_id)
            
            context = {
                'indications': graph_builder.get_drug_indications(drug_id),
                'side_effects': graph_builder.get_drug_side_effects(drug_id),
                'interactions': graph_builder.get_drug_interactions(drug_id)
            }
            
            results.append({
                'type': 'drug',
                'drug': drugs[i],
                'context': context,
                'score': sim,
                'source': 'graph'
            })
    
    return results


def _calculate_graph_score(query: str, drug: dict, context: dict) -> float:
    """Calculate score for graph retrieval"""
    query_lower = query.lower()
    drug_name_vi = drug.get('drug_name_vi', '').lower()
    drug_name_en = drug.get('drug_name_en', '').lower()
    
    if drug_name_vi in query_lower or drug_name_en in query_lower:
        base_score = 1.0
    elif any(word in query_lower for word in [drug_name_vi, drug_name_en]):
        base_score = 0.85
    else:
        base_score = 0.5
    
    bonus = 0
    if context.get('indications'):
        bonus += 0.1
    if context.get('side_effects'):
        bonus += 0.1
    if context.get('interactions'):
        bonus += 0.1
    
    final_score = min(base_score + bonus, 1.0)
    return final_score


def _reciprocal_rank_fusion(vector_results: List[dict], graph_results: List[dict], k: int = 60) -> List[dict]:
    """RRF (Reciprocal Rank Fusion)"""
    
    doc_ranks = {}
    
    for rank, result in enumerate(vector_results, 1):
        doc_key = _get_doc_key(result)
        rrf_score = 1 / (k + rank)
        if doc_key not in doc_ranks:
            doc_ranks[doc_key] = {
                'result': result.copy(),
                'rrf_score': 0,
                'sources': set()
            }
        doc_ranks[doc_key]['rrf_score'] += rrf_score
        doc_ranks[doc_key]['sources'].add('vector')
    
    for rank, result in enumerate(graph_results, 1):
        doc_key = _get_doc_key(result)
        rrf_score = 1 / (k + rank)
        if doc_key not in doc_ranks:
            doc_ranks[doc_key] = {
                'result': result.copy(),
                'rrf_score': 0,
                'sources': set()
            }
        doc_ranks[doc_key]['rrf_score'] += rrf_score
        doc_ranks[doc_key]['sources'].add('graph')
    
    fused_results = sorted(
        [{'result': v['result'], 'rrf_score': v['rrf_score'], 'sources': v['sources']} 
         for v in doc_ranks.values()],
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    
    final_results = []
    for item in fused_results:
        result = item['result']
        result['original_source_score'] = result.get('score')
        result['rrf_score'] = item['rrf_score']
        result['sources'] = list(item['sources'])
        result['score'] = item['rrf_score']
        final_results.append(result)
    
    logger.info(f"RRF fused {len(final_results)} results from {len(vector_results)} vector + {len(graph_results)} graph")
    return final_results


def _get_doc_key(result: dict) -> str:
    """Generate unique key for deduplication"""
    if result['type'] == 'chunk':
        content = result['chunk'].get('content', '')[:50]
        return f"chunk_{hash(content)}"
    else:
        drug_id = result['drug'].get('drug_id', '')
        return f"drug_{drug_id}"


def _rerank_results(query: str, results: List[dict], reranker) -> List[dict]:
    """Rerank RRF-fused results with BGE Reranker v2 M3"""
    
    if not results:
        return results
    
    pairs = []
    
    for result in results:
        if result['type'] == 'chunk':
            doc_text = result['chunk'].get('content', '')[:500]
        else:
            drug = result['drug']
            context = result['context']
            doc_text = f"Thuốc: {drug.get('drug_name_vi', '')} ({drug.get('drug_name_en', '')}). "
            
            if context.get('indications'):
                ind_names = [i.get('indication_name_vi', '') for i in context['indications'][:3]]
                doc_text += f"Chỉ định: {', '.join(ind_names)}. "
            
            if context.get('side_effects'):
                se_names = [s.get('side_effect_name_vi', '') for s in context['side_effects'][:3]]
                doc_text += f"Tác dụng phụ: {', '.join(se_names)}. "
            
            if context.get('interactions'):
                int_names = [d.get('drug_name_vi', '') for d in context['interactions'][:3]]
                doc_text += f"Tương tác với: {', '.join(int_names)}. "
        
        pairs.append([query, doc_text])
    
    try:
        reranked_scores = reranker.compute_score(pairs)
        
        for i, score in enumerate(reranked_scores):
            results[i]['original_rrf_score'] = results[i].get('rrf_score')
            results[i]['reranked_score'] = float(score)
            results[i]['score'] = float(score)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Reranked {len(results)} RRF-fused results with BGE Reranker v2 M3")
        
    except Exception as e:
        logger.error(f"Reranking error: {e}", exc_info=True)
        results.sort(key=lambda x: x.get('rrf_score', x['score']), reverse=True)
    
    return results[:5]


def _build_hybrid_context(results: List[dict]) -> str:
    """Build context from RRF + reranked results"""
    context_parts = []
    
    for result in results[:3]:
        if result['type'] == 'chunk':
            content = result['chunk'].get('content', '')[:500]
            context_parts.append(f"📄 {content}")
        
        elif result['type'] == 'drug':
            drug = result['drug']
            context = result['context']
            
            drug_info = f"💊 **{drug.get('drug_name_vi', 'N/A')}** ({drug.get('drug_name_en', 'N/A')})\n"
            drug_info += f"- ATC: {drug.get('atc_code', 'N/A')}\n"
            
            if context.get('indications'):
                indication_names = [i.get('indication_name_vi', '') for i in context['indications'][:3]]
                drug_info += f"- Chỉ định: {', '.join(indication_names)}\n"
            
            if context.get('side_effects'):
                side_effect_names = [s.get('side_effect_name_vi', '') for s in context['side_effects'][:3]]
                drug_info += f"- Tác dụng phụ: {', '.join(side_effect_names)}\n"
            
            if context.get('interactions'):
                interaction_names = [d.get('drug_name_vi', '') for d in context['interactions'][:3]]
                drug_info += f"- Tương tác với: {', '.join(interaction_names)}\n"
            
            context_parts.append(drug_info)
    
    return "\n\n".join(context_parts)


def _build_prompt(query: str, context: str, chat_history: List[dict]) -> str:
    """Build prompt for LLM"""
    
    history_text = ""
    if len(chat_history) > 1:
        for msg in chat_history[-5:-1]:
            if msg["role"] == "user":
                history_text += f"Người dùng: {msg['content']}\n"
            else:
                history_text += f"Trợ lý: {msg['content'][:200]}...\n"
    
    prompt = f"""Bạn là một chuyên gia dược học và y tế. Bạn sẽ trả lời câu hỏi dựa trên thông tin từ Dược thư quốc gia Việt Nam.


**Lịch sử cuộc trò chuyện:**
{history_text if history_text else "Đây là câu hỏi đầu tiên"}


**Thông tin liên quan (RRF + BGE Reranked):**
{context}


**Câu hỏi hiện tại của người dùng:**
{query}


**Hướng dẫn:**
1. Trả lời dựa trên thông tin được cung cấp VÀ lịch sử cuộc trò chuyện
2. Nếu người dùng hỏi "liều lượng như nào", hãy nhớ câu hỏi trước về thuốc gì
3. Giữ tính nhất quán trong cuộc trò chuyện
4. Nếu không có thông tin, hãy nói rõ "Thông tin này không có trong cơ sở dữ liệu"
5. Luôn nhắc người dùng tham khảo ý kiến bác sĩ khi cần thiết
6. Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu


**Trả lời:**"""
    
    return prompt


def _format_sources(results: List[dict]) -> str:
    """Format sources - Hide scores and tags, keep calculation"""
    sources = []
    
    for i, result in enumerate(results[:5], 1):
        if result['type'] == 'chunk':
            content = result['chunk'].get('content', '')[:100]
            sources.append(f"{i}. {content}...")
        
        elif result['type'] == 'drug':
            drug = result['drug']
            drug_name = drug.get('drug_name_vi', 'N/A')
            sources.append(f"{i}. {drug_name}")
    
    return "\n".join(sources)


@cl.on_chat_end
async def end():
    """Clean up on session end"""
    graph_builder = cl.user_session.get("graph_builder")
    if graph_builder:
        graph_builder.close()
        logger.info("Closed database connection")


if __name__ == "__main__":
    cl.run()
