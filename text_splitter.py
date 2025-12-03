from typing import List, Dict
from tqdm import tqdm


class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """将文本切分为块"""
        if not text:
            return []

        # 计算最大允许的字符数（根据token限制估算）
        # 对于中文，保守估计：1个token ≈ 3个字符
        max_chars_per_chunk = self.chunk_size #默认为500
        
        # 如果文本很短，直接返回
        if len(text) <= max_chars_per_chunk:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # 句子结束符
        sentence_endings = ['。', '！', '？', '.', '!', '?', '\n\n', '\r\n\r\n']
        
        while start < text_length:
            # 计算块的结束位置，确保不超过最大长度
            end = min(start + max_chars_per_chunk, text_length)
            
            # 如果还没到文本末尾，尝试在句子边界处切分
            if end < text_length:
                # 查找最接近的句子结束位置
                for ending in sentence_endings:
                    # 在end附近查找句子结束符
                    search_start = max(start, end - 100)  # 向前搜索100字符
                    pos = text.rfind(ending, search_start, end + 100)
                    if pos != -1:
                        end = pos + len(ending)
                        break
            
            chunk = text[start:end]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk.strip())
            
            # 更新start位置，考虑重叠
            start = end - self.chunk_overlap
            
            # 确保有进展
            if start <= end - self.chunk_overlap:
                start = end - self.chunk_overlap
            
            # 防止死循环
            if start >= end:
                start = end
        
        return chunks

    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """切分多个文档"""
        chunks_with_metadata = []

        for doc in tqdm(documents, desc="处理文档", unit="文档"):
            content = doc.get("content", "")
            filetype = doc.get("filetype", "")

            if filetype in [".pdf", ".pptx"]:
                # PDF和PPT已经按页分割，不再二次切分
                chunk_data = {
                    "content": content,
                    "filename": doc.get("filename", "unknown"),
                    "filepath": doc.get("filepath", ""),
                    "filetype": filetype,
                    "page_number": doc.get("page_number", 0),
                    "chunk_id": 0,
                    "images": doc.get("images", []),
                }
                chunks_with_metadata.append(chunk_data)

            elif filetype in [".docx", ".txt"]:
                # DOCX和TXT需要进行文本切分
                chunks = self.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "content": chunk,
                        "filename": doc.get("filename", "unknown"),
                        "filepath": doc.get("filepath", ""),
                        "filetype": filetype,
                        "page_number": 0,
                        "chunk_id": i,
                        "images": [],
                    }
                    chunks_with_metadata.append(chunk_data)

        print(f"\n文档处理完成，共 {len(chunks_with_metadata)} 个块")
        return chunks_with_metadata