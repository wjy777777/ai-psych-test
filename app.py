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

# ---------- 设置matplotlib中文字体 ----------
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    pass

# ---------- 管理员密码 ----------
ADMIN_PASSWORD = "admin123"

# ---------- 加载API Key ----------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# ---------- 数据文件路径 ----------
HISTORY_FILE = "test_history.csv"

# ---------- 页面配置 ----------
st.set_page_config(page_title="AI心理测试问答系统", page_icon="🧠", layout="wide")

# ---------- CSS美化 ----------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0 5px 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .disclaimer {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 12px 0;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        color: white;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px 0;
        border-top: 1px solid #eee;
    }
    .report-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf4 100%);
        padding: 20px 25px;
        border-radius: 16px;
        border-left: 6px solid #667eea;
        margin-top: 20px;
        line-height: 1.9;
        font-size: 1rem;
    }
    .report-box h3 {
        color: #2d3748;
        margin-top: 18px;
        margin-bottom: 6px;
        font-size: 1.1rem;
    }
    .report-box h4 {
        color: #4a5568;
        margin-top: 12px;
        margin-bottom: 4px;
        font-size: 1rem;
    }
    .emotion-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 4px 6px 4px 0;
    }
    .qr-container {
        background: #f8f9ff;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-top: 10px;
    }
    .admin-badge {
        background: #dc3545;
        color: white;
        padding: 2px 12px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .delete-checkbox {
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 标题 ----------
st.markdown('<div class="main-title">🧠 AI心理测试问答系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">基于自然语言处理的情绪与心理倾向分析</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <b>免责声明：</b>本系统仅供娱乐与自我参考，不构成任何医学或心理诊断建议。
</div>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    # ----- 二维码区域 -----
    st.header("📱 扫码访问")
    
    app_url = "https://retaining-july-stereo.ngrok-free.dev"
    
    try:
        qr = qrcode.QRCode(version=1, box_size=8, border=2)
        qr.add_data(app_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#667eea", back_color="white")
        
        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        buf.seek(0)
        
        st.markdown('<div class="qr-container">', unsafe_allow_html=True)
        st.image(buf, caption="扫码访问", use_container_width=True)
        st.caption(f"📌 {app_url}")
        
        st.download_button(
            label="⬇️ 下载二维码",
            data=buf.getvalue(),
            file_name="心理测试二维码.png",
            mime="image/png",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning("二维码生成失败，请复制链接分享")
        st.code(app_url)
    
    st.markdown("---")
    
    # ====== 管理员入口 ======
    st.header("🔐 管理员")
    
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    
    if not st.session_state.is_admin:
        password = st.text_input("请输入管理员密码", type="password", placeholder="输入密码查看后台数据")
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
                st.success(f"✅ 共 {len(df_history)} 条测试记录")
                
                st.markdown("**选择要删除的记录：**")
                
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
                            if st.button(f"查看详情", key=f"view_{idx}"):
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
                        st.success("✅ 已删除选中记录！")
                        st.rerun()
                
                st.markdown("---")
                
                if st.button("🗑️ 清空所有历史记录", use_container_width=True):
                    st.session_state['show_confirm_delete'] = True

                if st.session_state.get('show_confirm_delete', False):
                    st.warning("⚠️ 确定要删除所有历史记录吗？此操作不可恢复！")
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
                st.info("暂无历史记录")
        else:
            st.info("暂无历史记录")
        
        if st.button("🚪 退出管理员", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()

# ---------- 10道测试题 ----------
questions = [
    "1️⃣ 最近一周，你大多数时候的情绪状态是怎样的？请用几句话描述。",
    "2️⃣ 什么事情最容易让你感到压力或焦虑？请具体说明。",
    "3️⃣ 当你感到不开心时，你通常会怎么做来调节自己？",
    "4️⃣ 你对未来三个月有什么期待？又有什么担忧？",
    "5️⃣ 描述一个最近让你感到开心或满足的具体瞬间。",
    "6️⃣ 你觉得自己在人际关系中更偏向主动还是被动？为什么？",
    "7️⃣ 你对自己的整体评价偏向积极还是消极？请举例说明。",
    "8️⃣ 如果没有任何限制，你最想改变生活里的哪一点？",
    "9️⃣ 你觉得自己最近的能量水平（精力）如何？是充沛还是疲惫？",
    "🔟 用一句话总结你当前的心理状态。"
]

st.markdown("---")

if 'saved_answers' not in st.session_state:
    st.session_state.saved_answers = [""] * len(questions)

answers = []
for i, q in enumerate(questions):
    ans = st.text_area(
        q, 
        key=f"q_{i}", 
        value=st.session_state.saved_answers[i],
        height=70, 
        placeholder="请输入你的回答..."
    )
    st.session_state.saved_answers[i] = ans
    answers.append(ans)

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submitted = st.button("🚀 提交并分析我的心理状态", use_container_width=True)

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
    
    if '焦虑' in report_text or '压力' in report_text:
        scores[1] = min(90, scores[1] + 30)
    if '愉悦' in report_text or '开心' in report_text:
        scores[0] = min(90, scores[0] + 30)
    if '平静' in report_text or '淡定' in report_text:
        scores[2] = min(90, scores[2] + 30)
    if '积极' in report_text or '乐观' in report_text:
        scores[3] = min(90, scores[3] + 30)
    if '疲惫' in report_text or '累' in report_text:
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
你是一位资深心理动力学取向的心理咨询师，拥有10年以上临床经验。

## 你的任务
根据用户的10个回答，写出500-800字的深度心理分析报告。

## 分析原则
1. 不要复述用户的话
2. 洞察潜意识——用户没说出来的是什么？
3. 找矛盾点——用户回答中前后不一致的地方
4. 给出真正有用的建议——不要套话

## 报告格式（不要使用#号，直接用加粗标题）

**核心冲突：**
用1-2句话精炼概括用户当前最核心的心理冲突

**矛盾与防御：**
分析用户回答中的矛盾点，以及ta用了什么心理防御机制

**潜意识需求：**
用户真正渴望但没直接说出来的东西是什么？

**状态评估：**
- 情绪张力：
- 自我功能：
- 关系模式：
- 发展停滞：
每项用1-10分+一句话说明

**具体建议：**
3条针对性的建议，结合用户的具体情况

**一句话：**
一句有温度、有力量的话送给ta
"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是我的10个回答，请分析：\n\n{combined}"}
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
    if all(answers):
        if not api_key:
            st.error("❌ 未检测到API Key！")
            st.stop()
        
        with st.spinner("🧠 AI正在深度分析你的心理状态，请稍候..."):
            try:
                report = analyze_emotions(answers)
                st.balloons()
                
                emotions, scores, emotion_labels = parse_emotion_scores(report)
                save_history(answers, report, emotion_labels, scores)
                st.success("💾 测试结果已保存到历史记录！")
                
                st.markdown("---")
                st.markdown("## 📊 心理分析报告")
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("## 📈 情绪可视化分析")
                
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
                    st.info("💡 暂无足够关键词生成词云，请尝试更丰富的回答。")
                
                st.markdown("---")
                st.markdown("### 🏷️ 情绪标签")
                tag_html = ""
                for emo, score in zip(emotions, scores):
                    if score > 40:
                        tag_html += f'<span class="emotion-tag">{emo} {score}%</span>'
                st.markdown(tag_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ 分析失败：{str(e)}")
    else:
        st.error("⚠️ 请回答所有问题后再提交")

# ---------- 查看历史详情 ----------
if 'view_history' in st.session_state:
    show_history_detail(st.session_state['view_history'])

st.markdown("""
<div class="footer">
Made with ❤️ · AI心理测试问答系统 · 2026
</div>
""", unsafe_allow_html=True)