import os
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
)


class VectorStore:

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        # 初始化ChromaDB
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"description": "课程材料向量数据库"}
        )

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        try:
            # 检查并截断文本长度
            # 对于中文，粗略估计token数量：1个token ≈ 2-3个中文字符
            # 2048个token ≈ 4000-6000个中文字符
            max_char_length = 2000  # 安全字符数
            
            if len(text) > max_char_length:
                print(f"警告：文本长度 {len(text)} 超过限制，截断至 {max_char_length}")
                print(text)
                text = text[:max_char_length]
            
            response = self.client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {str(e)}")
            print(f"文本长度: {len(text)}")
            print(f"文本前100字符: {text[:100]}...")
            # 返回一个默认的零向量
            return [0.0] * 1536
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """添加文档块到向量数据库"""
        if not chunks:
            print("没有文档块可添加")
            return
        
        print(f"开始添加 {len(chunks)} 个文档块到向量数据库...")
        
        successful_count = 0
        failed_count = 0
        
        for i, chunk in enumerate(tqdm(chunks, desc="添加文档", unit="块")):
            try:
                # 生成唯一ID
                chunk_id = f"{chunk['filename']}_{chunk.get('page_number', 0)}_{chunk.get('chunk_id', 0)}"
                
                # 获取文本内容
                content = chunk.get("content", "")
                
                # 获取embedding
                embedding = self.get_embedding(content)
                
                # 准备元数据
                metadata = {
                    "filename": chunk.get("filename", ""),
                    "filepath": chunk.get("filepath", ""),
                    "filetype": chunk.get("filetype", ""),
                    "page_number": chunk.get("page_number", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                }
                
                # 添加到向量数据库
                self.collection.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"\n添加文档块失败: {chunk.get('filename', 'unknown')}")
                print(f"错误: {str(e)}")
                print(f"内容长度: {len(chunk.get('content', ''))}")
                print(f"跳过此文档块...")
                continue
        
        print(f"\n文档添加完成:")
        print(f"  成功: {successful_count}")
        print(f"  失败: {failed_count}")
        print(f"  总计: {self.collection.count()}")
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """搜索相关文档"""
        try:
            # 获取查询的embedding
            query_embedding = self.get_embedding(query)
            
            # 在向量数据库中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": 1 - distance,  # 将距离转换为相似度分数
                        "index": i
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"向量搜索失败: {str(e)}")
            return []

    def clear_collection(self) -> None:
        """清空collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass  # 如果集合不存在，忽略错误
        
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name, metadata={"description": "课程向量数据库"}
        )
        print("向量数据库已清空")

    def get_collection_count(self) -> int:
        """获取collection中的文档数量"""
        return self.collection.count()