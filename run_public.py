import streamlit as st
import time
import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
from wordcloud import WordCloud
import jieba
from collections import Counter
import qrcode
from io import BytesIO
import random

# ---------- 设置matplotlib中字体 ----------
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans SC']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    pass

# ---------- 管理员密码 ----------
ADMIN_PASSWORD = "123"

# ---------- 加载API Key ----------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# ---------- 数据文件路径 ----------
HISTORY_FILE = "test_history.csv"

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="心灵镜语 · AI心理探索",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS美化 ----------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0 5px 0;
        letter-spacing: 4px;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 20px;
        font-style: italic;
        letter-spacing: 2px;
    }
    .disclaimer {
        background: linear-gradient(135deg, #fff5f5 0%, #fff3cd 100%);
        border-left: 5px solid #ff6b6b;
        padding: 12px 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 14px 0;
        border: none;
        border-radius: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        letter-spacing: 2px;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 35px rgba(102, 126, 234, 0.5);
        color: white;
    }
    .footer {
        text-align: center;
        color: #ccc;
        font-size: 0.8rem;
        margin-top: 50px;
        padding: 25px 0;
        border-top: 1px solid #f0f0f0;
        letter-spacing: 1px;
    }
    .report-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef1f9 100%);
        padding: 30px 35px;
        border-radius: 20px;
        border-left: 6px solid #667eea;
        margin-top: 20px;
        line-height: 2;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        white-space: pre-wrap;
    }
    .report-box h3 {
        color: #2d3748;
        margin-top: 22px;
        margin-bottom: 8px;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .report-box h4 {
        color: #4a5568;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 1rem;
        font-weight: 600;
    }
    .emotion-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 5px 20px;
        border-radius: 25px;
        font-size: 0.9rem;
        margin: 4px 8px 4px 0;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s;
    }
    .emotion-tag:hover {
        transform: scale(1.05);
    }
    .qr-container {
        background: transparent;
        padding: 0;
        border-radius: 0;
        text-align: center;
        margin-top: 5px;
    }
    .qr-container img {
        border: none !important;
        box-shadow: none !important;
        border-radius: 8px;
    }
    .admin-badge {
        background: linear-gradient(135deg, #dc3545, #e74c3c);
        color: white;
        padding: 3px 14px;
        border-radius: 14px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    .question-text {
        font-size: 1.05rem;
        font-weight: 500;
        color: #2d3748;
        margin-bottom: 6px;
        line-height: 1.6;
        padding: 10px 0 4px 0;
    }
    .quote-box {
        background: linear-gradient(135deg, #f5f0ff 0%, #f0e6ff 100%);
        padding: 16px 24px;
        border-radius: 16px;
        text-align: center;
        font-style: italic;
        color: #5a3d7a;
        margin: 10px 0 20px 0;
        border: 1px solid #e8d5f5;
        font-size: 1.05rem;
    }
    .section-divider {
        text-align: center;
        color: #ddd;
        font-size: 1.2rem;
        letter-spacing: 10px;
        margin: 20px 0;
    }
    .stTextArea > div > div > textarea {
        font-size: 0.95rem;
        line-height: 1.6;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 12px 14px;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.15);
    }
    .stTextArea > label {
        display: none !important;
    }
    .stTextArea > div > label {
        display: none !important;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
            letter-spacing: 2px;
        }
        .report-box {
            padding: 18px 20px;
        }
        .question-text {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------- 标题 ----------
st.markdown('<div class="main-title">🪞 心灵镜语</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">「 在对话中，遇见真实的自己 」</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
🌸 <b>温馨提示：</b>这是一场与自己的深度对话。请在一个安静的环境中，花10-15分钟真诚地回答每一个问题。
你的回答没有对错之分，重要的是真实的感受。本系统仅供自我探索参考，不构成任何医学或心理诊断。
</div>
""", unsafe_allow_html=True)

# ---------- 随机引言 ----------
quotes = [
    "「 认识你自己 」—— 苏格拉底",
    "「 未经审视的人生不值得过 」—— 苏格拉底",
    "「 我们每个人都是自己心灵的探险家 」—— 荣格",
    "「 向内看，你会发现无限的世界 」",
    "「 真正的发现之旅，不在于寻找新风景，而在于拥有新的眼睛 」—— 普鲁斯特"
]

if 'quote_index' not in st.session_state:
    st.session_state.quote_index = random.randint(0, len(quotes)-1)

# ========== 侧边栏 ==========
with st.sidebar:
    # ----- 二维码区域 -----
    st.header("📱 扫码体验")
    
    app_url = os.getenv("APP_URL", "http://localhost:8501")
    
    try:
        qr = qrcode.QRCode(version=1, box_size=8, border=2)
        qr.add_data(app_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#667eea", back_color="white")
        
        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        buf.seek(0)
        
        st.markdown('<div class="qr-container">', unsafe_allow_html=True)
        st.image(buf, caption="扫码开始心灵探索", use_container_width=True)
        st.caption(f"📌 {app_url}")
        
        st.download_button(
            label="⬇️ 保存二维码",
            data=buf.getvalue(),
            file_name="心灵镜语二维码.png",
            mime="image/png",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning("⚠️ 二维码生成失败")
        st.code(app_url)
    
    st.markdown("---")
    
    # ----- 进度提示 -----
    st.header("📊 探索进度")
    if 'saved_answers' in st.session_state:
        answered = sum(1 for a in st.session_state.saved_answers if a.strip())
        total = len(st.session_state.saved_answers) if 'saved_answers' in st.session_state else 10
        st.progress(answered / total if total > 0 else 0)
        st.caption(f"已完成 {answered}/{total} 个问题")
    
    st.markdown("---")
    
    # ====== 管理员入口 ======
    st.header("🔐 管理员")
    
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    
    if not st.session_state.is_admin:
        password = st.text_input("请输入管理员密码", type="password", placeholder="输入密码查看后台")
        if st.button("🔑 验证身份", use_container_width=True):
            if password == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("✅ 验证成功！")
                st.rerun()
            else:
                st.error("❌ 密码错误！")
        
        st.markdown("---")
        st.info("🔒 登录后可查看历史记录")
        
    else:
        st.success("✅ 管理员已登录")
        st.markdown('<span class="admin-badge">🔒 管理员</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("📜 历史记录")
        
        if os.path.exists(HISTORY_FILE):
            df_history = pd.read_csv(HISTORY_FILE)
            if len(df_history) > 0:
                st.success(f"✅ 共 {len(df_history)} 次心灵探索记录")
                
                if 'to_delete' not in st.session_state:
                    st.session_state.to_delete = []
                
                for idx, row in df_history.iterrows():
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        if st.checkbox("", key=f"del_{idx}"):
                            if idx not in st.session_state.to_delete:
                                st.session_state.to_delete.append(idx)
                        else:
                            if idx in st.session_state.to_delete:
                                st.session_state.to_delete.remove(idx)
                    with col2:
                        with st.expander(f"📅 {row['时间']}"):
                            st.markdown(f"**情绪标签**: {row['情绪标签']}")
                            st.markdown(f"**报告摘要**: {row['报告摘要'][:100]}...")
                            if st.button(f"查看完整报告", key=f"view_{idx}"):
                                st.session_state['view_history'] = idx
                                st.rerun()
                
                if st.session_state.to_delete:
                    if st.button(f"🗑️ 删除选中的 {len(st.session_state.to_delete)} 条记录", use_container_width=True):
                        df_history = df_history.drop(st.session_state.to_delete).reset_index(drop=True)
                        if len(df_history) > 0:
                            df_history.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
                        else:
                            os.remove(HISTORY_FILE)
                        st.session_state.to_delete = []
                        st.success("✅ 已删除！")
                        st.rerun()
                
                st.markdown("---")
                
                if st.button("🗑️ 清空所有记录", use_container_width=True):
                    st.session_state['show_confirm_delete'] = True

                if st.session_state.get('show_confirm_delete', False):
                    st.warning("⚠️ 确定要删除所有记录吗？此操作不可恢复！")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 确认删除", use_container_width=True):
                            if os.path.exists(HISTORY_FILE):
                                os.remove(HISTORY_FILE)
                            st.session_state['show_confirm_delete'] = False
                            st.rerun()
                    with col2:
                        if st.button("❌ 取消", use_container_width=True):
                            st.session_state['show_confirm_delete'] = False
                            st.rerun()
            else:
                st.info("暂无记录，开始你的第一次心灵探索吧")
        else:
            st.info("暂无记录，开始你的第一次心灵探索吧")
        
        if st.button("🚪 退出管理员", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()

# ---------- 心灵探索引言 ----------
st.markdown(f"""
<div class="quote-box">
    {quotes[st.session_state.quote_index]}
</div>
""", unsafe_allow_html=True)

# ---------- 10道深度探索问题 ----------
questions = [
    "🌑 你最近一次在深夜睡不着，脑子里反复在想的是什么？",

    "🕯️ 如果让你给现在的自己写一封告别信，你会写什么？",

    "🧱 你生活中有一堵你一直在绕开、不敢面对的高墙，它是什么？",

    "📦 你内心有多少个箱子，装着从未打开过的情绪？",

    "🔇 你有没有一句话，憋在心里很久很久，却从来没对任何人说过？",

    "🎭 你每天在扮演的那个角色，和你真实的自己，距离有多远？",

    "🩹 你曾经用什么东西来止痛？它现在还管用吗？",

    "👤 如果你可以杀死一个过去的自己，你会杀死哪一个？",

    "🌊 你一直在向外抓取的东西，是不是你内心本来就缺失的？",

    "💬 最后一个问题——你有多久，没有允许自己真正地软弱过了？"
]

st.markdown('<div class="section-divider">✦ ✦ ✦</div>', unsafe_allow_html=True)

# ---------- 问题展示 ----------
if 'saved_answers' not in st.session_state:
    st.session_state.saved_answers = [""] * len(questions)

answers = []
for i, q in enumerate(questions):
    st.markdown(f'<div class="question-text">{q}</div>', unsafe_allow_html=True)
    
    ans = st.text_area(
        label="",
        key=f"q_{i}",
        value=st.session_state.saved_answers[i],
        height=68,
        placeholder="写下你的真实感受...",
        label_visibility="collapsed"
    )
    st.session_state.saved_answers[i] = ans
    answers.append(ans)
    
    st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider">✦ ✦ ✦</div>', unsafe_allow_html=True)

# ---------- 提交按钮 ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submitted = st.button("🌿 提交 · 聆听内心的声音", use_container_width=True)

# ============================================================
# 工具函数
# ============================================================

def extract_emotion_keywords(texts):
    all_text = " ".join(texts)
    words = jieba.lcut(all_text)
    stopwords = {
        '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说',
        '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '什么', '他', '这', '那', '它', '但', '而', '等',
        '与', '或', '又', '更', '最', '能', '让', '给', '把', '被', '从', '向', '以', '还', '再', '就', '可', '以',
        '吧', '呢', '啊', '哦', '嗯', '然后', '因为', '所以', '但是', '而且', '并且', '或者', '如果', '那么',
        '一样', '做个', '好玩', '变成', '很多', '有点', '觉得', '感觉', '知道', '想', '会', '能', '可以',
        '这个', '那个', '什么', '怎么', '这么', '那么', '这样', '那样', '还是', '就是', '只是', '不过',
        '大概', '可能', '应该', '一定', '真的', '非常', '比较', '特别', '一般', '基本', '其实', '好像'
    }
    words = [w for w in words if len(w) > 1 and w not in stopwords and not w.isdigit()]
    return words

def create_wordcloud(texts):
    words = extract_emotion_keywords(texts)
    if not words:
        return None
    word_freq = Counter(words)
    word_freq = dict(word_freq.most_common(30))
    if not word_freq:
        return None
    try:
        wordcloud = WordCloud(
            font_path="C:/Windows/Fonts/simhei.ttf",
            width=600,
            height=400,
            background_color='white',
            max_words=30,
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except:
        wordcloud = WordCloud(
            width=600,
            height=400,
            background_color='white',
            max_words=30,
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig

def parse_emotion_scores(report_text):
    emotions = ['愉悦', '焦虑', '平静', '积极', '疲惫']
    scores = [50, 50, 50, 50, 50]
    
    if '焦虑' in report_text or '压力' in report_text or '不安' in report_text:
        scores[1] = min(90, scores[1] + 30)
    if '愉悦' in report_text or '开心' in report_text or '幸福' in report_text:
        scores[0] = min(90, scores[0] + 30)
    if '平静' in report_text or '淡定' in report_text or '安宁' in report_text:
        scores[2] = min(90, scores[2] + 30)
    if '积极' in report_text or '乐观' in report_text or '希望' in report_text:
        scores[3] = min(90, scores[3] + 30)
    if '疲惫' in report_text or '累' in report_text or '耗竭' in report_text:
        scores[4] = min(90, scores[4] + 30)
    
    for i in range(5):
        scores[i] = max(30, min(95, scores[i] + np.random.randint(-8, 8)))
    
    emotion_labels = []
    for emo, score in zip(emotions, scores):
        if score > 45:
            emotion_labels.append(emo)
    
    return emotions, scores, ",".join(emotion_labels) if emotion_labels else "待分析"

def create_radar_chart(emotions, scores):
    fig = px.line_polar(
        r=scores,
        theta=emotions,
        line_close=True,
        range_r=[0, 100],
        title="情绪维度雷达图"
    )
    fig.update_traces(fill='toself', fillcolor='rgba(102, 126, 234, 0.3)', line_color='#667eea')
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400
    )
    return fig

def create_emotion_bar(emotions, scores):
    default_emotions = ['愉悦', '焦虑', '平静', '积极', '疲惫']
    
    if len(emotions) != 5:
        emotions = default_emotions
    
    scores = [int(s) if isinstance(s, (int, float)) else 50 for s in scores]
    if len(scores) != 5:
        scores = [50, 50, 50, 50, 50]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#4CAF50', '#FF5722', '#2196F3', '#FF9800', '#9C27B0']
    
    bars = ax.bar(emotions, scores, color=colors, edgecolor='white', linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_ylabel('情绪强度 (%)', fontsize=12)
    ax.set_title('各情绪维度分布', fontsize=14, fontweight='bold')
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, fontsize=11, fontweight='500')
    
    plt.tight_layout()
    return fig

def analyze_emotions(texts):
    combined = "\n".join([f"Q{i+1}: {t}" for i, t in enumerate(texts)])
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    system_prompt = """
你是一位温润而深刻的心理探索引导者，拥有丰富的心理咨询与心灵陪伴经验。

## 你的任务
根据用户的10个心灵探索回答，写出一份600-900字的深度心灵解读报告。

## 分析原则
1. 不要复述用户的话
2. 看见用户没说出来的——那些在字里行间隐藏的渴望与恐惧
3. 捕捉矛盾与张力——用户回答中的微妙不一致
4. 给出真正有共鸣的反馈——不要套话，要真诚

## 报告格式（不要使用任何星号或markdown标记，直接输出纯文本）

核心主题：
用1-2句温暖而精准的话，概括用户当下心灵旅程的核心主题

内在景观：
描绘用户内心世界的图景——有哪些情感在流动？有哪些声音在对话？

未说出的渴望：
用户内心真正渴望被看见、被理解的是什么？

当前状态扫描：
- 情绪气候：
- 内在力量：
- 关系温度：
- 成长节律：
（每项用1-10分+一句温柔的说明）

给你的回响：
3条有温度的、可实践的心灵建议

一句赠言：
用一句有诗意、有力量的话，作为这次对话的礼物

最后的话：
这份解读是对你内心世界的温柔映照，请带着你的判断去感受它。
"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是我的心灵探索回答，请用心解读：\n\n{combined}"}
        ],
        temperature=0.85,
        stream=False
    )
    
    return response.choices[0].message.content

def save_history(answers, report, emotion_labels, scores):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        '时间': now,
        '情绪标签': emotion_labels,
        '报告摘要': report[:200] + "..." if len(report) > 200 else report,
        '完整报告': report,
        '情绪分数': json.dumps(scores),
        '回答内容': json.dumps(answers)
    }
    
    df_new = pd.DataFrame([data])
    
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')

def show_history_detail(idx):
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if idx < len(df):
            row = df.iloc[idx]
            st.markdown("---")
            st.markdown(f"## 📅 历史记录详情 - {row['时间']}")
            st.markdown(f"**情绪标签**: {row['情绪标签']}")
            st.markdown("### 📊 完整报告")
            st.markdown(f'<div class="report-box">{row["完整报告"]}</div>', unsafe_allow_html=True)
            
            try:
                scores = json.loads(row['情绪分数'])
                emotions = ['愉悦', '焦虑', '平静', '积极', '疲惫']
                st.markdown("### 📈 情绪可视化")
                col1, col2 = st.columns(2)
                with col1:
                    radar_fig = create_radar_chart(emotions, scores)
                    st.plotly_chart(radar_fig, use_container_width=True)
                with col2:
                    bar_fig = create_emotion_bar(emotions, scores)
                    st.pyplot(bar_fig, use_container_width=True)
                
                try:
                    answers = json.loads(row['回答内容'])
                    st.markdown("### ☁️ 关键词词云图")
                    wordcloud_fig = create_wordcloud(answers)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig, use_container_width=True)
                except:
                    pass
            except:
                pass

# ============================================================
# 主逻辑
# ============================================================

if submitted:
    empty_indices = [i for i, ans in enumerate(answers) if not ans.strip()]
    
    if empty_indices:
        st.error(f"⚠️ 还有 {len(empty_indices)} 个问题等待你的回应。请完成所有问题后再提交。")
    else:
        if not api_key:
            st.error("❌ 未检测到API Key！请检查配置。")
            st.stop()
        
        with st.spinner("🌿 正在聆听你的心灵之声，请稍候..."):
            try:
                report = analyze_emotions(answers)
                st.balloons()
                
                emotions, scores, emotion_labels = parse_emotion_scores(report)
                save_history(answers, report, emotion_labels, scores)
                
                st.success("💾 你的心灵探索已保存！")
                
                st.markdown("---")
                st.markdown("## 🌿 心灵解读报告")
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("## 📈 情绪景观图")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🎯 情绪维度雷达图")
                    radar_fig = create_radar_chart(emotions, scores)
                    st.plotly_chart(radar_fig, use_container_width=True)
                with col2:
                    st.markdown("### 📊 情绪分布柱状图")
                    bar_fig = create_emotion_bar(emotions, scores)
                    st.pyplot(bar_fig, use_container_width=True)
                
                st.markdown("### ☁️ 关键词词云图")
                wordcloud_fig = create_wordcloud(answers)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig, use_container_width=True)
                else:
                    st.info("💡 词云图需要更丰富的表达来生成，继续探索你的内心吧。")
                
                st.markdown("---")
                st.markdown("### 🏷️ 情绪标签")
                tag_html = ""
                for emo, score in zip(emotions, scores):
                    if score > 40:
                        tag_html += f'<span class="emotion-tag">{emo} {score}%</span>'
                st.markdown(tag_html, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; color: #888; font-size: 0.9rem; padding: 10px 0;">
                    🌸 感谢你完成了这次心灵探索。每一份真诚的回应，都是对自我的温柔凝视。
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ 解读过程中出现了一些问题：{str(e)}")
                st.info("请稍后重试，或检查API密钥配置。")

# ---------- 查看历史详情 ----------
if 'view_history' in st.session_state:
    show_history_detail(st.session_state['view_history'])

st.markdown("""
<div class="footer">
    🌱 心灵镜语 · 在对话中遇见真实的自己 · 2026
</div>
""", unsafe_allow_html=True)