# RAG智能课程助教系统

基于检索增强生成（RAG）的智能课程助教系统，能够从多格式课程文档中提取知识并智能回答学生问题。

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv RAG
```
### 2. 激活虚拟环境

```bash
RAG\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```
### 4.配置API密钥

```bash
OPENAI_API_KEY = "sk-你的API密钥"
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 5. 准备课堂文档
将PDF、PPTX、DOCX或TXT格式的课件放入data/目录

### 6. 数据预处理

```bash
python process_data.py
```

### 7. 运行对话系统

```bash
python main.py
```