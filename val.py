# llm_eval.py
import os
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# 复用你项目中已有的 openai 客户端和配置
from rag_agent import RAGAgent
from config import MODEL_NAME, OPENAI_API_KEY, OPENAI_API_BASE
from openai import OpenAI

class LLMEvaluator:
    def __init__(self):
        # 初始化你的RAG智能体和用于评估的LLM客户端（使用同一个）
        self.rag_agent = RAGAgent(model=MODEL_NAME)
        self.eval_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        # ！！！核心：请替换成你精心设计的5-10个测试问题 ！！！
        self.test_questions = [
            "根据康奈尔（Connell）的理论，什么是‘霸权男性气质’（Hegemonic masculinity）？",
            "请简述性别秩序（Gender Order）这一概念的核心内容。",
            "交叉性（Intersectionality）视角如何帮助我们分析社会不平等？",
            "LGBT是什么？",
            "栈和队列的区别是什么？",
            "简述查找的几种方法",
            "如何利用二叉树进行排序？",
            # ... 请在此添加更多基于你课程资料的具体问题
        ]

    def ask_rag(self, question):
        """调用你的RAG系统获取答案和检索到的上下文。"""
        # 注意：这里需要根据你 rag_agent.answer_question 的实际返回值来调整
        result = self.rag_agent.answer_question(question)
        answer = result.get("answer", "")
        # 关键：提取出检索到的原始上下文文本，用于后续评估
        # 假设你的RAG返回的检索结果在 'retrieved_docs' 字段中
        contexts = []
        if 'retrieved_docs' in result and result['retrieved_docs']:
            for doc in result['retrieved_docs']:
                contexts.append(doc.get('content', ''))
        # 如果格式不同，你可能需要这样调整：
        # contexts = [result.get("context", "")]
        return answer, contexts

    def llm_as_judge(self, question, answer, contexts):
        """让LLM作为裁判，对RAG的答案进行评估。"""
        # 将上下文拼接成一个字符串
        context_text = "\n---\n".join(contexts)

        # 设计评估提示词（你可以根据需求调整维度和标准）
        evaluation_prompt = f"""
请你作为一名严格的学术助教，评估以下问答的质量。

【学生问题】
{question}

【助教参考的课程材料（上下文）】
{context_text if context_text.strip() else '（无相关内容）'}

【助教给出的答案】
{answer}

请从以下两个维度进行评估，并分别给出1-5分的整数打分（1分最差，5分最好），以及一句简短的评语。

评估维度：
1. **忠实度**：答案是否严格基于上方提供的“课程材料（上下文）”，是否包含无法从上下文中推断出的信息或“幻觉”。
2. **相关度**：答案是否直接、完整地回应了“学生问题”，是否答非所问或遗漏关键点。

请以严格的JSON格式输出，格式如下：
{{
    "scores": {{
        "faithfulness": ...,
        "relevancy": ...
    }},
    "comments": {{
        "faithfulness": "...",
        "relevancy": "..."
    }}
}}
"""

        try:
            response = self.eval_client.chat.completions.create(
                model=MODEL_NAME,  # 使用同一个模型进行评估
                messages=[
                    {"role": "system", "content": "你是一个公正、严格的评估者，总是输出有效的JSON。"},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,  # 低温度以保证评估稳定性
                response_format={"type": "json_object"}  # 要求返回JSON
            )
            evaluation_result = json.loads(response.choices[0].message.content)
            return evaluation_result
        except Exception as e:
            print(f"LLM评估出错: {e}")
            # 返回一个默认的评估结果
            return {
                "scores": {"faithfulness": 0, "relevancy": 0},
                "comments": {"faithfulness": "评估失败", "relevancy": "评估失败"}
            }

    def run_evaluation(self):
        """运行完整的评估流程。"""
        print("🧪 开始基于LLM的RAG系统评估...")
        all_results = []

        for question in tqdm(self.test_questions, desc="评估进度"):
            # 1. RAG系统生成答案
            answer, contexts = self.ask_rag(question)

            # 2. LLM对答案进行评估
            eval_result = self.llm_as_judge(question, answer, contexts)

            # 3. 记录结果
            record = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "scores": eval_result["scores"],
                "comments": eval_result["comments"]
            }
            all_results.append(record)

        # 4. 保存结果
        self.save_results(all_results)
        print(f"\n✅ 评估完成！结果已保存至 'llm_eval_results/' 目录。")

    def save_results(self, results):
        """将评估结果保存为JSON和CSV文件。"""
        output_dir = "llm_eval_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细的JSON结果
        json_path = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存简明的CSV分数总表
        df_data = []
        for r in results:
            df_data.append({
                "question": r["question"][:100] + "...",  # 问题摘要
                "faithfulness_score": r["scores"]["faithfulness"],
                "relevancy_score": r["scores"]["relevancy"],
                "faithfulness_comment": r["comments"]["faithfulness"],
                "relevancy_comment": r["comments"]["relevancy"]
            })
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(output_dir, f"scores_summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 打印平均分
        avg_faith = df["faithfulness_score"].mean()
        avg_relev = df["relevancy_score"].mean()
        print(f"\n📊 平均分数（满分5分）:")
        print(f"  忠实度 (Faithfulness): {avg_faith:.2f}")
        print(f"  相关度 (Relevancy): {avg_relev:.2f}")
        print(f"\n📁 详细结果文件:")
        print(f"  {json_path}")
        print(f"  {csv_path}")

def main():
    """主函数"""
    try:
        evaluator = LLMEvaluator()
        evaluator.run_evaluation()
    except Exception as e:
        print(f"评估过程出错: {e}")

if __name__ == "__main__":
    main()