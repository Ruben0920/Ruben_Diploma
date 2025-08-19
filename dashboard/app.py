import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data_loader import DataLoader
import time

st.set_page_config(
    page_title="Interactive Multi-Modal Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem;
        border-left: 5px solid #3498db;
        background: #ecf0f1;
        border-radius: 5px;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin: 0.5rem 0;
    }
    .speaker-text {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 5px solid;
    }
    .tooltip-content {
        background: #2c3e50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the analysis data"""
    try:
        data_loader = DataLoader()
        return data_loader
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def format_time(seconds):
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_sentiment_color(sentiment_score):
    """Get color based on sentiment score"""
    if sentiment_score > 0.3:
        return "#2ecc71"
    elif sentiment_score < -0.3:
        return "#e74c3c"
    else:
        return "#f39c12"

def get_speaker_color(speaker_id):
    """Get consistent color for each speaker"""
    colors = {
        'SPEAKER_00': '#3498db',
        'SPEAKER_01': '#e74c3c',
        'SPEAKER_02': '#2ecc71',
    }
    return colors.get(speaker_id, '#95a5a6')

def create_sentiment_timeline(data_loader, speaker_id):
    """Create interactive sentiment timeline for a speaker"""
    segments_df = data_loader.get_speaker_segments(speaker_id)
    
    if segments_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=segments_df['start_time'],
        y=segments_df['sentiment_compound'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color=get_speaker_color(speaker_id), width=3),
        marker=dict(size=8, color=segments_df['sentiment_compound'].apply(get_sentiment_color)),
        hovertemplate='<b>Time:</b> %{x}<br>' +
                     '<b>Sentiment:</b> %{y:.3f}<br>' +
                     '<b>Topic:</b> ' + segments_df['topic'] + '<br>' +
                     '<b>Emotion:</b> ' + segments_df['emotion'] + '<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Sentiment Timeline - {speaker_id}",
        xaxis_title="Time (seconds)",
        yaxis_title="Sentiment Score",
        hovermode='closest',
        height=400,
        showlegend=False
    )
    
    return fig

def create_emotion_distribution(data_loader, speaker_id):
    """Create emotion distribution chart for a speaker"""
    emotion_df = data_loader.get_speaker_emotion_distribution(speaker_id)
    
    if emotion_df.empty:
        return go.Figure()
    
    fig = px.pie(
        emotion_df,
        values='count',
        names='emotion',
        title=f"Emotion Distribution - {speaker_id}",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>Emotion:</b> %{label}<br>' +
                     '<b>Count:</b> %{value}<br>' +
                     '<b>Percentage:</b> %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_topic_focus(data_loader, speaker_id):
    """Create topic focus chart for a speaker"""
    topic_df = data_loader.get_speaker_topic_focus(speaker_id)
    
    if topic_df.empty:
        return go.Figure()
    
    fig = px.bar(
        topic_df,
        x='segment_count',
        y='topic',
        orientation='h',
        title=f"Topic Focus - {speaker_id}",
        color='segment_count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Number of Segments",
        yaxis_title="Topics",
        height=400,
        showlegend=False
    )
    
    return fig

def create_sentiment_comparison(data_loader, topic):
    """Create sentiment comparison chart for a topic across speakers"""
    topic_df = data_loader.get_topic_sentiment_comparison(topic)
    
    if topic_df.empty:
        return go.Figure()
    
    fig = px.bar(
        topic_df,
        x='speaker_id',
        y='mean_sentiment',
        color='mean_sentiment',
        color_continuous_scale='RdBu',
        title=f"Sentiment Comparison - {topic}",
        text=topic_df['mean_sentiment'].round(3)
    )
    
    fig.update_layout(
        xaxis_title="Speaker",
        yaxis_title="Average Sentiment",
        height=400,
        showlegend=False
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def create_sentiment_heatmap(data_loader):
    """Create interactive sentiment heatmap for all topics and speakers"""
    topic_df = data_loader.data.get('topic_speaker_sentiment_df', pd.DataFrame())
    
    if topic_df.empty:
        return go.Figure()
    
    pivot_df = topic_df.pivot(index='topic', columns='speaker_id', values='mean_sentiment')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdBu',
        zmid=0,
        text=pivot_df.values.round(3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>Topic:</b> %{y}<br>' +
                     '<b>Speaker:</b> %{x}<br>' +
                     '<b>Sentiment:</b> %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Topic-Speaker Sentiment Heatmap",
        xaxis_title="Speaker",
        yaxis_title="Topic",
        height=500,
        yaxis={'categoryorder': 'category ascending'}
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">Interactive Multi-Modal Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Analyzing Political Debate Content: Sentiment, Emotion, and Topic Analysis")
    
    with st.spinner("Loading analysis data..."):
        data_loader = load_data()
    
    if data_loader is None:
        st.error("Failed to load data. Please check the output directory.")
        return
    
    st.sidebar.title("Navigation")
    section = st.sidebar.selectbox(
        "Choose a section:",
        ["Interactive Transcript Explorer", "Speaker Profile Deep Dive", "Topic Polarization Face-Off", "Overall Debate Analysis"]
    )
    
    if section == "Interactive Transcript Explorer":
        render_transcript_explorer(data_loader)
    elif section == "Speaker Profile Deep Dive":
        render_speaker_profile(data_loader)
    elif section == "Topic Polarization Face-Off":
        render_topic_faceoff(data_loader)
    elif section == "Overall Debate Analysis":
        render_overall_analysis(data_loader)

def render_transcript_explorer(data_loader):
    """Render Section 1: Interactive Transcript Explorer"""
    st.markdown('<h2 class="section-header">Interactive Transcript Explorer</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This section displays the full debate transcript with interactive analysis. 
    Hover over text segments to see detailed sentiment, emotion, and topic analysis.
    """)
    
    segments_df = data_loader.data.get('segments_df', pd.DataFrame())
    
    if segments_df.empty:
        st.warning("No transcript data available.")
        return
    
    st.subheader("Full Debate Transcript")
    
    for idx, segment in segments_df.iterrows():
        speaker_id = segment['speaker_id']
        text = segment['text']
        start_time = segment['start_time']
        end_time = segment['end_time']
        sentiment = segment['sentiment_compound']
        emotion = segment['emotion']
        emotion_conf = segment['emotion_confidence']
        topic = segment['topic']
        
        speaker_color = get_speaker_color(speaker_id)
        sentiment_color = get_sentiment_color(sentiment)
        
        st.markdown(f"""
        <div class="speaker-text" style="border-left-color: {speaker_color}; background-color: {speaker_color}20;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong style="color: {speaker_color};">{speaker_id}</strong>
                <span style="font-size: 0.9rem; color: #666;">
                    {format_time(start_time)} - {format_time(end_time)}
                </span>
            </div>
            <div style="margin-bottom: 0.5rem;">{text}</div>
            <div style="display: flex; gap: 1rem; font-size: 0.9rem;">
                <span style="color: {sentiment_color}; font-weight: bold;">
                    Sentiment: {sentiment:.3f}
                </span>
                <span style="color: #8e44ad;">
                    Emotion: {emotion} ({emotion_conf:.1%})
                </span>
                <span style="color: #2c3e50;">
                    Topic: {topic}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_speaker_profile(data_loader):
    """Render Section 2: Speaker Profile Deep Dive"""
    st.markdown('<h2 class="section-header">Speaker Profile Deep Dive</h2>', unsafe_allow_html=True)
    
    speakers = data_loader.get_speakers()
    if not speakers:
        st.warning("No speaker data available.")
        return
    
    selected_speaker = st.selectbox("Select a speaker to analyze:", speakers)
    
    if selected_speaker:
        st.subheader(f"Analysis for {selected_speaker}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.plotly_chart(create_sentiment_timeline(data_loader, selected_speaker), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_emotion_distribution(data_loader, selected_speaker), use_container_width=True)
        
        with col3:
            st.plotly_chart(create_topic_focus(data_loader, selected_speaker), use_container_width=True)
        
        st.subheader("Speaker Statistics")
        speaker_stats = data_loader.data.get('speaker_sentiment_df', pd.DataFrame())
        if not speaker_stats.empty:
            speaker_data = speaker_stats[speaker_stats['speaker_id'] == selected_speaker].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Sentiment", f"{speaker_data['mean_sentiment']:.3f}")
            with col2:
                st.metric("Total Segments", speaker_data['segment_count'])
            with col3:
                st.metric("Sentiment Std Dev", f"{speaker_data['sentiment_std']:.3f}")
            with col4:
                segments_df = data_loader.get_speaker_segments(selected_speaker)
                if not segments_df.empty:
                    total_time = segments_df['end_time'].max() - segments_df['start_time'].min()
                    st.metric("Total Speaking Time", format_time(total_time))

def render_topic_faceoff(data_loader):
    """Render Section 3: Topic Polarization Face-Off"""
    st.markdown('<h2 class="section-header">Topic Polarization Face-Off</h2>', unsafe_allow_html=True)
    
    topics = data_loader.get_topics()
    if not topics:
        st.warning("No topic data available.")
        return
    
    selected_topic = st.selectbox("Select a topic to analyze:", topics)
    
    if selected_topic:
        st.subheader(f"Analysis for: {selected_topic}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_sentiment_comparison(data_loader, selected_topic), use_container_width=True)
        
        with col2:
            st.subheader("Most Polarizing Statements")
            
            polarizing_statements = data_loader.get_most_polarizing_statements(selected_topic)
            
            st.markdown("**Most Positive Statements:**")
            for i, statement in enumerate(polarizing_statements['positive'], 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{i}. {statement['speaker_id']}</strong><br>
                    <em>"{statement['text'][:100]}{'...' if len(statement['text']) > 100 else ''}"</em><br>
                    <span style="color: #2ecc71;">Sentiment: {statement['sentiment_compound']:.3f}</span> | 
                    <span style="color: #8e44ad;">Emotion: {statement['emotion']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Most Negative Statements:**")
            for i, statement in enumerate(polarizing_statements['negative'], 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{i}. {statement['speaker_id']}</strong><br>
                    <em>"{statement['text'][:100]}{'...' if len(statement['text']) > 100 else ''}"</em><br>
                    <span style="color: #e74c3c;">Sentiment: {statement['sentiment_compound']:.3f}</span> | 
                    <span style="color: #8e44ad;">Emotion: {statement['emotion']}</span>
                </div>
                """, unsafe_allow_html=True)

def render_overall_analysis(data_loader):
    """Render Section 4: Overall Debate Analysis"""
    st.markdown('<h2 class="section-header">Overall Debate Analysis</h2>', unsafe_allow_html=True)
    
    st.subheader("Interactive Sentiment Heatmap")
    st.plotly_chart(create_sentiment_heatmap(data_loader), use_container_width=True)
    
    st.subheader("Key Performance Indicators")
    
    overall_stats = data_loader.get_overall_statistics()
    
    if overall_stats:
        st.markdown("**Speaking Time per Speaker:**")
        speaking_time_df = pd.DataFrame(overall_stats['speaking_time'])
        if not speaking_time_df.empty:
            speaking_time_df['total_time'] = speaking_time_df['total_time'].apply(format_time)
            st.dataframe(speaking_time_df, use_container_width=True)
        
        st.markdown("**Average Sentiment per Speaker:**")
        avg_sentiment_df = pd.DataFrame(overall_stats['avg_sentiment'])
        if not avg_sentiment_df.empty:
            st.dataframe(avg_sentiment_df, use_container_width=True)
        
        st.markdown("**Most Frequent Emotion per Speaker:**")
        emotion_df = pd.DataFrame(overall_stats['most_frequent_emotion'])
        if not emotion_df.empty:
            st.dataframe(emotion_df, use_container_width=True)
    
    st.subheader("Summary Statistics")
    
    segments_df = data_loader.data.get('segments_df', pd.DataFrame())
    if not segments_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Segments", len(segments_df))
        with col2:
            st.metric("Total Speakers", segments_df['speaker_id'].nunique())
        with col3:
            st.metric("Total Topics", segments_df['topic'].nunique())
        with col4:
            total_duration = segments_df['end_time'].max() - segments_df['start_time'].min()
            st.metric("Total Duration", format_time(total_duration))
        
        st.subheader("Overall Emotion Distribution")
        emotion_counts = segments_df['emotion'].value_counts()
        fig = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Emotion Distribution Across All Speakers"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
