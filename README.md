# AI 心理测试问答系统 (ai-psych-test)

基于大语言模型（DeepSeek）的 AI 心理测试问答系统。用户回答 10 道开放式心理探索问题，系统调用大模型生成深度心理分析报告，并进行情绪可视化。

## 功能特性
- 10 道开放式心理探索问题
- 调用 DeepSeek API 生成结构化心理分析报告
- 情绪雷达图、柱状图、关键词词云
- 历史记录保存、查看、删除
- 管理员后台
- 二维码分享，支持移动端访问

## 环境要求
- Python 3.10+
- DeepSeek API Key（在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请）

## 安装与运行

1. 克隆仓库：
```bash
git clone https://github.com/wjy777777/ai-psych-test.git
cd ai-psych-test
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置 API Key：
在项目根目录创建 `.env` 文件，写入：
```
DEEPSEEK_API_KEY=你的真实API Key
```

4. 运行：
```bash
streamlit run run_public.py
```

5. 浏览器访问：
```
http://localhost:8501
```
## 项目结构
```
ai-psych-test/
├── run_public.py          # 主程序
├── requirements.txt       # 依赖清单
├── .env.example           # API Key 配置示例
└── README.md              # 项目说明
```

## 免责声明
本系统仅供娱乐与自我探索参考，不构成任何医学或心理诊断建议。
