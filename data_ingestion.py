"""Data Ingestion Module for Medical GraphRAG System (Updated regex for VN Drug Formulary)"""

import re
import logging
from typing import List, Dict, Generator, Optional
from pathlib import Path
import json

import pypdf

from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VIETNAMESE_STOPWORDS,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalPDFIngestor:
    """Ingest and process Vietnamese medical PDF documents, with batch + streaming."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    # ---------------------- PDF READING ---------------------- #

    def extract_text_from_pdf(
        self,
        start_page: int = 0,
        end_page: Optional[int] = None,
    ) -> str:
        logger.info(f"Extracting text from {self.pdf_path} (pages {start_page}–{end_page})")
        text = ""
        try:
            with open(self.pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                if end_page is None or end_page > num_pages:
                    end_page = num_pages
                if start_page < 0:
                    start_page = 0

                for page_num in range(start_page, end_page):
                    page = pdf_reader.pages[page_num]

                    page_text = page.extract_text() or ""
                    text += "\n" + page_text + "\n"
                    
                    if (page_num - start_page) % 50 == 0:
                        logger.info(f"  Extracted page {page_num}/{end_page}")
                        
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

        return text

    # ---------------------- CHUNKING ---------------------- #

    def split_into_chunks_stream(self, text: str) -> Generator[Dict, None, None]:
        start = 0
        chunk_id = 0
        length = len(text)

        while start < length:
            end = min(start + CHUNK_SIZE, length)
            
            if end < length:
                last_period = text.rfind(".", start, end)
                if last_period > start + CHUNK_SIZE // 2:
                    end = last_period + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:
                yield {
                    "chunk_id": f"chunk_{chunk_id}",
                    "content": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "token_count": len(chunk_text.split()),
                }
                chunk_id += 1

            next_start = end - CHUNK_OVERLAP
            if next_start <= start:
                next_start = end
            start = next_start

    # ---------------------- DRUG EXTRACTION ---------------------- #

    def extract_drug_entries_stream(self, text: str) -> Generator[Dict, None, None]:
        """
        Trích xuất thuốc dựa trên cấu trúc Dược thư VN:
        Tiêu đề IN HOA -> Tên chung quốc tế -> Mã ATC -> ...
        """
        logger.info("Extracting drug entries...")
        
        seen_ids = set()

        drug_start_pattern = re.compile(r'\n([A-ZÂÊÔĂƠƯĐ0-9\-\s\(\)]{3,})\n(?=.*?(?:Tên chung quốc tế|Mã ATC|Loại thuốc))', re.DOTALL)
        
        matches = list(drug_start_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            start_idx = match.start()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start_idx:end_idx].strip()
            
            drug_name_vi = match.group(1).strip()
            
            if len(drug_name_vi) < 3 or "MỤC LỤC" in drug_name_vi or "DƯỢC THƯ" in drug_name_vi:
                continue

            drug_id = drug_name_vi.lower().replace(" ", "_").replace("-", "_")
            
            if drug_id in seen_ids:
                logger.debug(f"Skipping duplicate drug_id: {drug_id}")
                continue
            
            if re.match(r'^\d+_\d+', drug_id) or 'dtqgvn' in drug_id.lower():
                logger.debug(f"Skipping garbage drug_id: {drug_id}")
                continue
            
            seen_ids.add(drug_id)

            drug_data = {
                "drug_id": drug_id,
                "drug_name_vi": drug_name_vi,
                "drug_name_en": self._extract_field(section_text, r"Tên chung quốc tế:\s*(.*?)(?:\n|Mã ATC|Loại thuốc)"),
                "atc_code": self._extract_field(section_text, r"Mã ATC:\s*([A-Z0-9]+)"),
                "drug_type": self._extract_field(section_text, r"Loại thuốc:\s*(.*?)(?:\n|Dạng thuốc)"),
                
                "indications": self._extract_multiline(section_text, "Chỉ định"),
                "contraindications": self._extract_multiline(section_text, "Chống chỉ định"),
                "dosages": self._extract_dosages(section_text),
                "side_effects": self._extract_side_effects(section_text),
                "interactions": self._extract_multiline(section_text, "Tương tác thuốc"),
                
                "raw_text": section_text[:500]
            }

            if len(drug_data["indications"]) > 0 and drug_data["atc_code"]:
                yield drug_data


    # ---------------------- HELPER EXTRACTORS ---------------------- #

    def _extract_field(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_multiline(self, text: str, header: str) -> List[str]:
        """
        Lấy nội dung từ sau `header` đến `header tiếp theo` (các header thường in đậm hoặc viết hoa đầu dòng)
        """
        start_match = re.search(rf"{header}\b", text, re.IGNORECASE)
        if not start_match:
            return []
            
        start_idx = start_match.end()
        
        next_headers = ["Chống chỉ định", "Thận trọng", "Thời kỳ mang thai", "Tác dụng không mong muốn", "Liều lượng", "Tương tác thuốc", "Độ ổn định", "Quá liều", "Thông tin quy chế"]
        
        min_end_idx = len(text)
        
        for next_h in next_headers:
            if next_h.lower() == header.lower(): continue
            
            nm = re.search(rf"{next_h}\b", text[start_idx:], re.IGNORECASE)
            if nm:
                real_idx = start_idx + nm.start()
                if real_idx < min_end_idx:
                    min_end_idx = real_idx
                    
        content = text[start_idx:min_end_idx].strip()
        
        items = [s.strip() for s in re.split(r'[;\n•-]\s+', content) if len(s.strip()) > 5]
        return items[:15]
    def _extract_dosages(self, text: str) -> List[Dict]:
        
        match = re.search(r"Liều lượng và cách dùng(.*?)(?:Quá liều|Tương tác|$)", text, re.DOTALL | re.IGNORECASE)
        if not match: return []
        
        content = match.group(1)
        dosages = []
        # Pattern tìm số lượng + đơn vị (mg, g, ml...)
        patterns = r"(\d+(?:[.,]\d+)?)\s*(mg|g|ml|IU|viên|giọt)"
        
        for m in re.finditer(patterns, content):
            dosages.append({
                "amount": m.group(1),
                "unit": m.group(2)
            })
        return dosages[:10]

    def _extract_side_effects(self, text: str) -> List[Dict]:
        # Tìm phần ADR
        match = re.search(r"Tác dụng không mong muốn(.*?)(?:Hướng dẫn|Liều lượng|$)", text, re.DOTALL | re.IGNORECASE)
        if not match: return []
        
        content = match.group(1)
        effects = []
        
        # Phân loại theo tần suất (Thường gặp, Ít gặp...)
        frequencies = ["Thường gặp", "Ít gặp", "Hiếm gặp"]
        current_freq = "Không xác định"
        
        # Tách dòng để xử lý sơ bộ
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            for freq in frequencies:
                if line.lower().startswith(freq.lower()):
                    current_freq = freq
                    break
            
            # Tách các triệu chứng trong dòng (thường cách nhau bằng dấu phẩy)
            # Loại bỏ từ khóa tần suất khỏi dòng nếu có
            clean_line = line
            if current_freq in line:
                clean_line = line.replace(current_freq, "")
            
            symptoms = [s.strip() for s in clean_line.split(',') if len(s.strip()) > 3]
            for s in symptoms:
                if len(s) < 50: # Chỉ lấy các cụm từ ngắn gọn
                    effects.append({"name": s, "severity": current_freq})
                    
        return effects[:20]

# ---------------------- MAIN (BATCH + STREAM) ---------------------- #

if __name__ == "__main__":
    from math import ceil

    pdf_path = PDF_PATH
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = data_dir / "chunks.jsonl"
    drugs_path = data_dir / "extracted_drugs.jsonl"

    ingestor = MedicalPDFIngestor(pdf_path)

    # Đọc tổng số trang
    with open(pdf_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        total_pages = len(pdf_reader.pages)
    
    logger.info(f"Start processing PDF: {total_pages} pages")

    # Xóa file cũ
    if chunks_path.exists(): chunks_path.unlink()
    if drugs_path.exists(): drugs_path.unlink()

    # Cấu hình Batch
    BATCH_SIZE = 200
    MAX_CHUNKS = 200000

    total_chunks = 0
    total_drugs = 0
    num_batches = ceil(total_pages / BATCH_SIZE)

    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total_pages)
        
        logger.info(f"=== Batch {i+1}/{num_batches}: Pages {start}-{end} ===")
        
        # 1. Đọc text batch
        text_batch = ingestor.extract_text_from_pdf(start, end)
        
        # 2. Chunking (Stream)
        with open(chunks_path, "a", encoding="utf-8") as f:
            for chunk in ingestor.split_into_chunks_stream(text_batch):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
                if total_chunks >= MAX_CHUNKS:
                    break
        
        if total_chunks >= MAX_CHUNKS:
            logger.warning("Reached MAX_CHUNKS limit.")
            break

        # 3. Extract Drugs (Stream)
        with open(drugs_path, "a", encoding="utf-8") as f:
            for drug in ingestor.extract_drug_entries_stream(text_batch):
                f.write(json.dumps(drug, ensure_ascii=False) + "\n")
                total_drugs += 1
        
        logger.info(f"  -> Current totals: {total_chunks} chunks, {total_drugs} drugs")
        
        # Free memory
        del text_batch

    logger.info(f"COMPLETED. Final: {total_chunks} chunks, {total_drugs} drugs.")
