from typing import List, Dict, Optional, Tuple

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    MODEL_NAME,
    TOP_K,
)
from vector_store import VectorStore


class RAGAgent:
    def __init__(
        self,
        model: str = MODEL_NAME,
    ):
        self.model = model

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        self.vector_store = VectorStore()

        # 系统提示词 - 定义助教的角色
        self.system_prompt = """你是一位专业、耐心、严谨的课程助教。你的任务是帮助学生理解课程内容，解答学习中的疑问。

请遵循以下原则：
1. **基于课程内容回答**：你的回答必须基于提供的课程材料，不能凭空捏造信息。
2. **准确引用来源**：明确标注引用内容的来源（文件名和页码）。
3. **专业但友好**：使用专业的学术语言，但保持友好和鼓励的态度。
4. **分点回答**：对于复杂问题，使用分点或列表形式回答，使结构清晰。
5. **诚实承认未知**：如果课程材料中没有相关信息，诚实地告诉学生你不知道，并建议参考其他资料或咨询授课老师。
6. **鼓励思考**：在解答问题后，可以提出引导性问题，帮助学生深入思考。

请确保你的回答准确、有帮助，并且严格基于提供的课程材料。"""

    def retrieve_context(
        self, query: str, top_k: int = TOP_K
    ) -> Tuple[str, List[Dict]]:
        """检索相关上下文"""
        # 1. 使用向量数据库检索相关文档
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        
        if not retrieved_docs:
            return "（未检索到相关课程材料）", []
        
        # 2. 格式化检索结果，构建上下文字符串
        context_parts = ["检索到的相关课程内容：\n"]
        
        for i, doc in enumerate(retrieved_docs):
            content = doc["content"]
            metadata = doc["metadata"]
            filename = metadata.get("filename", "未知文件")
            page_num = metadata.get("page_number", 0)
            
            # 添加来源信息
            source_info = f"来源：{filename}"
            if page_num > 0:
                source_info += f" 第{page_num}页"
            
            context_part = f"\n【{i+1}】{source_info}\n{content}\n"
            context_parts.append(context_part)
        
        context_str = "\n".join(context_parts)
        
        return context_str, retrieved_docs

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """生成回答"""
        messages = [{"role": "system", "content": self.system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        # 构建用户提示词
        user_text = f"""请基于以下课程内容回答学生的问题：

{context}

学生问题：{query}

请确保：
1. 回答严格基于上述课程内容
2. 明确标注引用来源（如：根据《机器学习》第3页的内容...）
3. 如果课程内容中没有相关信息，请诚实地说明
4. 保持回答的专业性和教育性

请开始回答："""

        messages.append({"role": "user", "content": user_text})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                temperature=0.7, 
                max_tokens=1500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def answer_question(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> Dict[str, any]:
        """回答问题"""
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)

        if not context or context == "（未检索到相关课程材料）":
            context = "（未检索到特别相关的课程材料）"

        answer = self.generate_response(query, context, chat_history)

        return {
            "answer": answer,
            "context": context,
            "retrieved_docs": retrieved_docs
        }

    def chat(self) -> None:
        """交互式对话"""
        print("=" * 60)
        print("欢迎使用智能课程助教系统！")
        print("=" * 60)
        print("系统说明：")
        print("1. 我是一个基于课程内容的智能助教")
        print("2. 我会根据课程材料回答你的问题")
        print("3. 我的回答会标注来源信息")
        print("4. 输入 '退出' 或 'quit' 结束对话")
        print("=" * 60)

        chat_history = []

        while True:
            try:
                query = input("\n学生: ").strip()

                if not query:
                    continue
                
                if query.lower() in ['退出', 'quit', 'exit', 'bye']:
                    print("\n助教: 再见！祝你学习顺利！")
                    break

                print("助教: 正在思考...")
                result = self.answer_question(query, chat_history=chat_history)
                answer = result["answer"]
                
                print(f"\n助教: {answer}")
                
                # 显示来源信息（可选）
                if result["retrieved_docs"]:
                    print("\n【参考来源】")
                    for doc in result["retrieved_docs"]:
                        metadata = doc["metadata"]
                        filename = metadata.get("filename", "未知文件")
                        page_num = metadata.get("page_number", 0)
                        source = filename
                        if page_num > 0:
                            source += f" 第{page_num}页"
                        print(f"- {source}")

                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})
                
                # 限制历史记录长度
                if len(chat_history) > 10:  # 保留最近5轮对话
                    chat_history = chat_history[-10:]

            except KeyboardInterrupt:
                print("\n\n助教: 对话结束，祝你学习顺利！")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}")