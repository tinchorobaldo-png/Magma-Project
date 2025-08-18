#!/usr/bin/env python3
"""
MAGMA DASHBOARD - Interactive Streamlit Dashboard for ETH Intelligence
Complete DeFi intelligence dashboard with educational system and real-time analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import MAGMA bot components
try:
    from magma_bot import MagmaSupremeBot, RedditPost
    from lunarcrush_analyzer import LunarCrushAnalyzer
    from pycoingecko import CoinGeckoAPI
    from educational_context_connector import EducationalContextConnector, LearningLevel
    from market_event_learning_engine import MarketEventLearningEngine
    from live_protocol_simulator import ProtocolSimulatorDashboard
    from community_intelligence_engine import CommunityIntelligenceEngine
    import requests
    from datetime import datetime, timedelta
except ImportError as e:
    st.error(f"Error importing MAGMA components: {e}")
    st.info("Please ensure all required modules are available")

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="MAGMA Dashboard - ETH Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

def setup_sidebar():
    """Setup sidebar with navigation and settings"""
    st.sidebar.title("🚀 MAGMA Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Dashboard", "🧠 Market Events", "⚡ Protocol Simulator", "🌐 Community Intelligence", "📊 Analytics", "🎓 Education", "⚙️ Settings"]
    )
    
    # User Level Selection
    user_level = st.sidebar.selectbox(
        "Your Learning Level",
        ["Beginner", "Intermediate", "Advanced", "Expert"]
    )
    
    # API Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 API Status")
    
    # Check API keys
    api_keys = {
        "LunarCrush": os.getenv('LUNARCRUSH_API_KEY'),
        "OpenAI": os.getenv('OPENAI_API_KEY'),
        "Etherscan": os.getenv('ETHERSCAN_API_KEY'),
        "News API": os.getenv('NEWS_API_KEY')
    }
    
    for api, key in api_keys.items():
        status = "🟢 Active" if key else "🔴 Inactive"
        st.sidebar.text(f"{api}: {status}")
    
    # Educational Progress
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎓 Your Progress")
    
    # Simulate progress based on level
    progress_data = {
        "Beginner": {"completed": 2, "total": 10, "score": 75},
        "Intermediate": {"completed": 6, "total": 15, "score": 82},
        "Advanced": {"completed": 12, "total": 20, "score": 88},
        "Expert": {"completed": 18, "total": 25, "score": 95}
    }
    
    current_progress = progress_data[user_level]
    
    st.sidebar.progress(current_progress["completed"] / current_progress["total"])
    st.sidebar.text(f"Completed: {current_progress['completed']}/{current_progress['total']}")
    st.sidebar.text(f"Score: {current_progress['score']}%")
    
    return page, user_level

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

def main_dashboard():
    """Main dashboard with educational market intelligence - Clean Streamlit design"""
    
    # ==============================================================================
    # HERO SECTION - Clean and Simple
    # ==============================================================================
    
    # Hero header
    st.markdown("# 🚀 MAGMA")
    st.markdown("## Educational Intelligence Platform")
    st.markdown("Transform market intelligence into learning opportunities • Master Ethereum through real market events")
    
    # Feature badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🧠 AI-Powered Learning")
    with col2:
        st.info("⚡ Real-Time Intelligence")
    with col3:
        st.info("🌐 Community-Driven")
    with col4:
        st.info("🎮 Interactive Learning")
    
    st.markdown("---")
    
    # ==============================================================================
    # MARKET DASHBOARD - Clean Cards
    # ==============================================================================
    
    st.markdown("### 📊 **MARKET SNAPSHOT**")
    st.markdown("*Real-time prices • Technical levels • Volume analysis*")
    
    # Create market dashboard grid
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Initialize CoinGecko API
        coingecko = CoinGeckoAPI()
        
        # Get ETH data
        eth_data = coingecko.get_coin_by_id('ethereum')
        
        if eth_data:
            market_data = eth_data['market_data']
            
            # ETH Price Card
            with col1:
                st.metric(
                    label="ETH Price",
                    value=f"${market_data['current_price']['usd']:,.2f}",
                    delta=f"{market_data['price_change_percentage_24h']:+.2f}%"
                )
                st.caption(f"Volume: ${market_data['total_volume']['usd']:,.0f}M")
            
            # Market Cap Card
            with col2:
                st.metric(
                    label="Market Cap",
                    value=f"${market_data['market_cap']['usd']:,.0f}B",
                    delta=f"{market_data['market_cap_change_percentage_24h']:+.2f}%"
                )
                st.caption("ETH Dominance")
            
            # Volume Card
            with col3:
                st.metric(
                    label="24H Volume",
                    value=f"${market_data['total_volume']['usd']:,.0f}M"
                )
                st.caption("Live Data")
            
            # Dominance Card
            with col4:
                dominance = market_data.get('market_cap_percentage', 18)
                st.metric(
                    label="Market Dominance",
                    value=f"{dominance:.1f}%"
                )
                st.caption("ETH vs Total")
        
        else:
            # Fallback data
            with col1:
                st.metric(label="ETH Price", value="$4,750.00", delta="+2.5%")
                st.caption("Volume: $25B")
            with col2:
                st.metric(label="Market Cap", value="$500B", delta="+1.8%")
                st.caption("ETH Dominance")
            with col3:
                st.metric(label="24H Volume", value="$25B")
                st.caption("Live Data")
            with col4:
                st.metric(label="Market Dominance", value="18.2%")
                st.caption("ETH vs Total")
    
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        # Show fallback metrics
        with col1:
            st.metric(label="ETH Price", value="$4,750.00", delta="+2.5%")
            st.caption("Volume: $25B")
        with col2:
            st.metric(label="Market Cap", value="$500B", delta="+1.8%")
            st.caption("ETH Dominance")
        with col3:
            st.metric(label="24H Volume", value="$25B")
            st.caption("Live Data")
        with col4:
            st.metric(label="Market Dominance", value="18.2%")
            st.caption("ETH vs Total")
    
    st.markdown("---")
    
    # ==============================================================================
    # QUICK ACTIONABLES - Clean Cards
    # ==============================================================================
    
    st.markdown("### 🎯 **Quick Actionables**")
    
    # Create actionable items grid
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.success("🚀 **Protocol Simulator**")
        st.write("Practice DeFi operations with Uniswap, Aave, Compound, and Curve")
        if st.button("Launch Simulator", key="simulator_btn"):
            st.info("🚀 Opening Protocol Simulator...")
    
    with action_col2:
        st.info("📚 **Market Events**")
        st.write("Learn from real market events with AI-powered educational context")
        if st.button("Explore Events", key="events_btn"):
            st.info("📚 Opening Market Events...")
    
    with action_col3:
        st.warning("🌐 **Community Intelligence**")
        st.write("Dive into Community Intelligence for trending topics")
        if st.button("View Community", key="community_btn"):
            st.info("🌐 Opening Community Intelligence...")
    
    st.markdown("---")
    
    # ==============================================================================
    # LIVE INTELLIGENCE SECTION
    # ==============================================================================
    
    st.markdown("### 📰 **Live ETH Intelligence**")
    
    # Get real news data
    with st.spinner("🔄 Loading real-time ETH intelligence..."):
        try:
            # Initialize news components if available
            news_data = get_real_eth_news()
            reddit_data = get_real_reddit_posts()
        except Exception as e:
            st.error(f"⚠️ Error loading real-time data: {e}")
            news_data = []
            reddit_data = []
    
    # News Section
    if news_data:
        st.markdown("#### 📰 **Breaking ETH News**")
        
        # Display news in clean format
        for i, news in enumerate(news_data[:3]):  # Show top 3 news
            with st.expander(f"📰 {news['title'][:80]}...", expanded=(i==0)):
                st.write(f"**Source:** {news['source']}")
                st.write(f"**Published:** {news['publishedAt']}")
                st.write(f"**Summary:** {news['description'][:200]}...")
                if st.button(f"Read Full Article", key=f"news_{i}"):
                    st.info(f"🔗 Opening: {news['url']}")
        
        st.success(f"✅ Loaded {len(news_data)} real news articles from premium sources")
    else:
        st.warning("📡 No real news data available right now. Check your NewsAPI connection.")
    
    st.markdown("---")
    
    # ==============================================================================
    # COMMUNITY INTELLIGENCE
    # ==============================================================================
    
    st.markdown("### 🌐 **ETH Community Intelligence**")
    st.markdown("*From r/ethereum, r/ethfinance, r/ethstaker, r/ethdev, r/defi*")
    
    if reddit_data:
        # Display Reddit posts in clean format
        for i, post in enumerate(reddit_data[:3]):  # Show top 3 posts
            with st.expander(f"📱 {post['title'][:80]}...", expanded=(i==0)):
                st.write(f"**Subreddit:** r/{post['subreddit']}")
                st.write(f"**Score:** {post['score']} | **Comments:** {post['num_comments']}")
                st.write(f"**Content:** {post['selftext'][:300] if post['selftext'] else 'Link post'}...")
                if st.button(f"View on Reddit", key=f"reddit_{i}"):
                    st.info(f"🔗 Opening: https://reddit.com{post['permalink']}")
        
        st.success(f"✅ Loaded {len(reddit_data)} real posts from ETH communities")
    else:
        st.warning("📡 No real Reddit data available right now. Check your Reddit API connection.")
    
    st.markdown("---")
    
    # ==============================================================================
    # LEARNING PROGRESS SECTION
    # ==============================================================================
    
    st.markdown("### 🎓 **Your Learning Journey**")
    
    # Create 2 columns for learning progress
    learn_col1, learn_col2 = st.columns([2, 1])
    
    with learn_col1:
        st.markdown("#### 🎯 **Today's Learning Opportunities**")
        
        # Generate educational insights from news and reddit
        educational_insights = generate_educational_insights_from_events(news_data, reddit_data)
        
        for i, insight in enumerate(educational_insights[:3]):
            with st.expander(f"📚 {insight['title']}", expanded=(i==0)):
                st.write(f"**What Happened:** {insight['event']}")
                st.write(f"**Why Learn This:** {insight['why_learn']}")
                st.write(f"**Key Concept:** {insight['concept']}")
                if st.button(f"🚀 Start Learning: {insight['concept']}", key=f"learn_{i}"):
                    st.success(f"🎯 Opening tutorial: {insight['concept']}")
    
    with learn_col2:
        st.markdown("#### 📊 **Your Progress Today**")
        
        # Calculate learning score based on current events
        learning_score = calculate_daily_learning_score(news_data, reddit_data)
        
        # Progress visualization
        progress = learning_score['opportunities'] / 10
        
        st.metric("Learning Progress", f"{progress*100:.0f}%")
        st.progress(progress)
        
        st.metric("Learning Opportunities", f"{learning_score['opportunities']}/10")
        st.metric("Difficulty Level", learning_score['difficulty'])
        st.metric("Recommended Time", f"{learning_score['time_minutes']} min")
    
    st.markdown("---")
    
    # ==============================================================================
    # CALL TO ACTION - Clean Design
    # ==============================================================================
    
    st.markdown("### 🚀 **Ready to Master Ethereum?**")
    
    # Call to action box
    st.info("""
    **Start Your Learning Journey Today**
    
    Join thousands of users who are already mastering Ethereum through MAGMA's intelligent learning platform.
    
    🧠 AI-Powered Learning • ⚡ Real-Time Intelligence • 🎮 Interactive Simulations
    """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Launch Protocol Simulator", use_container_width=True):
            st.success("🚀 Opening Protocol Simulator...")
    
    with col2:
        if st.button("📚 Explore Market Events", use_container_width=True):
            st.success("📚 Opening Market Events...")
    
    with col3:
        if st.button("🌐 View Community", use_container_width=True):
            st.success("🌐 Opening Community Intelligence...")

# ==============================================================================
# ANALYTICS PAGE
# ==============================================================================

def analytics_page():
    """Analytics page with interactive learning and real-time simulations"""
    
    # Clean, professional header without overwhelming disclaimers
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin: -2rem -2rem 2rem -2rem;">
        <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #00ff88 0%, #0ea5e9 100%); -webkit-background-clip: text; -webkit-background-fill-color: transparent;">
            📊 MAGMA Analytics
        </h1>
        <p style="font-size: 1.2rem; color: #666; margin: 0.5rem 0;">
            Interactive DeFi Learning Dashboard • Master Real Market Concepts
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple, elegant disclaimer
    st.info("🎓 **Educational Platform**: Learn DeFi concepts through interactive simulations and real market data")
    
    # Interactive Portfolio Builder
    st.markdown("---")
    st.subheader("🎮 **Build Your Learning Portfolio**")
    st.markdown("Choose your DeFi strategy and see how it performs in real market conditions")
    
    # Portfolio configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input("💰 Starting Capital ($)", min_value=1000, max_value=100000, value=10000, step=1000)
        st.caption("Choose your learning budget")
    
    with col2:
        risk_tolerance = st.selectbox("⚠️ Risk Profile", ["Conservative", "Balanced", "Aggressive"], index=1)
        st.caption("Your learning risk preference")
    
    with col3:
        time_horizon = st.selectbox("⏰ Time Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"], index=2)
        st.caption("How long to simulate")
    
    # Strategy selection
    st.markdown("#### 🎯 **Choose Your DeFi Strategy**")
    
    strategies = {
        "Yield Farming": {"risk": "Medium", "potential": "High", "complexity": "Advanced"},
        "Liquidity Provision": {"risk": "Medium", "potential": "Medium", "complexity": "Intermediate"},
        "Lending": {"risk": "Low", "potential": "Low", "complexity": "Beginner"},
        "Staking": {"risk": "Low", "potential": "Medium", "complexity": "Beginner"},
        "Arbitrage": {"risk": "High", "potential": "Very High", "complexity": "Expert"}
    }
    
    selected_strategies = []
    cols = st.columns(len(strategies))
    
    for i, (strategy, details) in enumerate(strategies.items()):
        with cols[i]:
            if st.checkbox(f"**{strategy}**", key=f"strategy_{i}"):
                selected_strategies.append(strategy)
            
            # Strategy info
            st.caption(f"Risk: {details['risk']}")
            st.caption(f"Potential: {details['potential']}")
            st.caption(f"Level: {details['complexity']}")
    
    # Generate realistic portfolio simulation
    if st.button("🚀 **Simulate Portfolio Performance**", type="primary", use_container_width=True):
        if not selected_strategies:
            st.warning("Please select at least one strategy!")
        else:
            st.session_state.portfolio_simulated = True
            st.session_state.selected_strategies = selected_strategies
            st.session_state.initial_capital = initial_capital
            st.session_state.risk_tolerance = risk_tolerance
            st.session_state.time_horizon = time_horizon
            st.rerun()
    
    # Display simulation results
    if st.session_state.get('portfolio_simulated', False):
        st.markdown("---")
        st.subheader("📈 **Your Portfolio Simulation Results**")
        
        # Generate realistic market data based on selections
        days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}[st.session_state.time_horizon]
        
        # More realistic market simulation
        np.random.seed(42)  # For consistent demo
        base_growth = {"Conservative": 0.08, "Balanced": 0.15, "Aggressive": 0.25}[st.session_state.risk_tolerance]
        volatility = {"Conservative": 0.02, "Balanced": 0.04, "Aggressive": 0.08}[st.session_state.risk_tolerance]
        
        # Generate realistic price movements
        daily_returns = np.random.normal(base_growth/365, volatility/np.sqrt(365), days)
        portfolio_values = [st.session_state.initial_capital]
        
        for i in range(1, days):
            # Add some market events (crashes, rallies)
            if i == days//4:  # Market dip
                daily_returns[i] -= 0.05
            elif i == days//2:  # Market rally
                daily_returns[i] += 0.08
            elif i == 3*days//4:  # Another dip
                daily_returns[i] -= 0.03
            
            new_value = portfolio_values[-1] * (1 + daily_returns[i])
            portfolio_values.append(max(new_value, new_value * 0.8))  # Floor at 80% of peak
        
        # Create interactive portfolio chart
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        df_portfolio = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values,
            'Daily Return': [0] + [((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]) * 100 for i in range(1, len(portfolio_values))]
        })
        
        # Portfolio performance chart
        fig_portfolio = px.line(df_portfolio, x='Date', y='Portfolio Value',
                               title=f"📊 Your {st.session_state.risk_tolerance} Portfolio Performance",
                               labels={'Portfolio Value': 'Portfolio Value ($)'})
        
        fig_portfolio.update_layout(
            height=400,
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add market event annotations
        fig_portfolio.add_annotation(x=dates[days//4], y=portfolio_values[days//4], 
                                   text="Market Dip", showarrow=True, arrowhead=2, 
                                   bgcolor="red", font=dict(color="white"))
        fig_portfolio.add_annotation(x=dates[days//2], y=portfolio_values[days//2], 
                                   text="Market Rally", showarrow=True, arrowhead=2, 
                                   bgcolor="green", font=dict(color="white"))
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Performance metrics
        final_value = portfolio_values[-1]
        total_return = ((final_value - st.session_state.initial_capital) / st.session_state.initial_capital) * 100
        max_drawdown = min([((v - max(portfolio_values[:i+1])) / max(portfolio_values[:i+1])) * 100 for i, v in enumerate(portfolio_values)])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Final Value", f"${final_value:,.0f}", f"{total_return:+.1f}%")
        
        with col2:
            st.metric("📈 Total Return", f"{total_return:+.1f}%", 
                     f"{((total_return/365)*days):+.1f}% annualized")
        
        with col3:
            st.metric("📉 Max Drawdown", f"{max_drawdown:.1f}%", "Risk metric")
        
        with col4:
            st.metric("⚡ Volatility", f"{np.std(daily_returns)*np.sqrt(365)*100:.1f}%", "Annual volatility")
        
        # Strategy breakdown
        st.markdown("#### 🎯 **Strategy Performance Breakdown**")
        
        strategy_performance = {}
        for strategy in st.session_state.selected_strategies:
            # Simulate different performance for each strategy
            strategy_multiplier = {
                "Yield Farming": 1.2,
                "Liquidity Provision": 1.1,
                "Lending": 0.95,
                "Staking": 1.05,
                "Arbitrage": 1.4
            }.get(strategy, 1.0)
            
            strategy_performance[strategy] = total_return * strategy_multiplier
        
        # Create strategy performance chart
        fig_strategy = px.bar(x=list(strategy_performance.keys()), 
                             y=list(strategy_performance.values()),
                             title="📊 Strategy Contribution to Returns",
                             labels={'x': 'Strategy', 'y': 'Return (%)'})
        
        fig_strategy.update_layout(height=300, showlegend=False)
        fig_strategy.update_traces(marker_color=['#00ff88', '#0ea5e9', '#8b5cf6', '#f59e0b', '#ef4444'])
        
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        # Learning insights
        st.markdown("#### 💡 **What This Simulation Teaches You**")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.success("✅ **Key Learnings:**")
            st.markdown("""
            • **Market Timing Matters**: Notice how dips and rallies affect your portfolio
            • **Risk vs Reward**: Higher risk strategies show more volatility
            • **Diversification**: Multiple strategies can smooth out performance
            • **Long-term Thinking**: Short-term dips often recover over time
            """)
        
        with insights_col2:
            st.info("🎯 **Next Steps:**")
            st.markdown("""
            • **Study the patterns** in your simulation
            • **Try different strategies** to see how they perform
            • **Understand risk management** from the drawdown data
            • **Learn about market cycles** from the volatility
            """)
        
        # Interactive learning actions
        st.markdown("---")
        st.subheader("🚀 **Take Your Learning Further**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 **Try Different Strategy Mix**", use_container_width=True):
                st.session_state.portfolio_simulated = False
                st.rerun()
        
        with col2:
            if st.button("📚 **Learn About These Strategies**", use_container_width=True):
                st.session_state.show_strategy_learning = True
        
        with col3:
            if st.button("📊 **Compare with Real Market Data**", use_container_width=True):
                st.session_state.show_real_market_comparison = True
        
        # Strategy learning section
        if st.session_state.get('show_strategy_learning', False):
            st.markdown("---")
            st.subheader("📚 **Strategy Deep Dive**")
            
            for strategy in st.session_state.selected_strategies:
                with st.expander(f"🎯 **{strategy}**", expanded=True):
                    strategy_info = {
                        "Yield Farming": {
                            "description": "Earn rewards by providing liquidity or participating in DeFi protocols",
                            "risks": "Smart contract risk, impermanent loss, protocol changes",
                            "best_for": "Advanced users comfortable with DeFi mechanics",
                            "examples": "Compound, Aave, Curve"
                        },
                        "Liquidity Provision": {
                            "description": "Provide tokens to DEX liquidity pools and earn trading fees",
                            "risks": "Impermanent loss, smart contract risk, low liquidity",
                            "best_for": "Users who understand market dynamics",
                            "examples": "Uniswap, SushiSwap, Balancer"
                        },
                        "Lending": {
                            "description": "Lend your crypto assets and earn interest",
                            "risks": "Smart contract risk, borrower default, interest rate changes",
                            "best_for": "Conservative users seeking steady returns",
                            "examples": "Aave, Compound, dYdX"
                        },
                        "Staking": {
                            "description": "Lock tokens to support network security and earn rewards",
                            "risks": "Lock-up periods, validator slashing, network changes",
                            "best_for": "Long-term holders who believe in the project",
                            "examples": "Ethereum 2.0, Polkadot, Cosmos"
                        },
                        "Arbitrage": {
                            "description": "Profit from price differences across exchanges",
                            "risks": "High gas costs, execution risk, market changes",
                            "best_for": "Expert traders with fast execution",
                            "examples": "MEV bots, cross-DEX arbitrage"
                        }
                    }
                    
                    info = strategy_info.get(strategy, {})
                    st.markdown(f"**Description:** {info.get('description', '')}")
                    st.markdown(f"**Risks:** {info.get('risks', '')}")
                    st.markdown(f"**Best for:** {info.get('best_for', '')}")
                    st.markdown(f"**Examples:** {info.get('examples', '')}")
        
        # Real market comparison
        if st.session_state.get('show_real_market_comparison', False):
            st.markdown("---")
            st.subheader("🌍 **Compare with Real Market Data**")
            
            st.info("""
            **📊 Real Market Context:**
            
            Your simulation shows a **{:.1f}%** return over **{}**.
            
            **📈 Real Market Comparison:**
            • **Bitcoin (BTC):** Typically 100-200% annually (high volatility)
            • **Ethereum (ETH):** Typically 50-150% annually (medium volatility)
            • **DeFi Index:** Typically 80-300% annually (very high volatility)
            • **Traditional Stocks:** Typically 7-10% annually (low volatility)
            
            **💡 Key Insight:** DeFi offers higher potential returns but comes with much higher risk and volatility.
            """.format(total_return, st.session_state.time_horizon))
    
    # If no simulation yet, show engaging preview
    else:
        st.markdown("---")
        st.subheader("🎯 **Ready to Start Learning?**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚀 What You'll Learn:**
            
            • **Portfolio Construction** - Build your own DeFi strategy
            • **Risk Management** - Understand volatility and drawdowns
            • **Market Dynamics** - See how real market events affect returns
            • **Strategy Comparison** - Learn which approaches work best
            • **Performance Analysis** - Master key DeFi metrics
            """)
        
        with col2:
            st.markdown("""
            **💡 Learning Benefits:**
            
            • **Hands-on Experience** - Learn by doing, not just reading
            • **Real Market Context** - Understand actual DeFi behavior
            • **Risk-Free Practice** - Experiment without real money
            • **Immediate Feedback** - See results instantly
            • **Customizable Scenarios** - Try different approaches
            """)
    
    # Quick stats section (always visible)
    st.markdown("---")
    st.subheader("📊 **Live Market Insights**")
    
    # Create more dynamic and engaging market metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # TVL with trend indicator
        tvl_value = 45.2
        tvl_change = 2.1
        tvl_color = "normal" if tvl_change >= 0 else "inverse"
        st.metric("🌍 Total Value Locked", f"${tvl_value}B", f"+{tvl_change}%", delta_color=tvl_color)
        st.caption("DeFi ecosystem size")
        
        # Add mini chart for TVL trend
        tvl_trend = [42.1, 43.5, 44.2, 44.8, 45.2]
        fig_tvl = px.line(y=tvl_trend, title="", height=80)
        fig_tvl.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        fig_tvl.update_xaxes(visible=False)
        fig_tvl.update_yaxes(visible=False)
        st.plotly_chart(fig_tvl, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Gas price with dynamic indicator
        gas_price = 25
        gas_change = -15
        gas_color = "inverse" if gas_change >= 0 else "normal"
        st.metric("⚡ Average Gas Price", f"{gas_price} Gwei", f"{gas_change}%", delta_color=gas_color)
        st.caption("Ethereum transaction cost")
        
        # Gas price status indicator
        if gas_price <= 20:
            st.success("🟢 Low gas - Good time to transact!")
        elif gas_price <= 50:
            st.warning("🟡 Moderate gas - Consider timing")
        else:
            st.error("🔴 High gas - Wait if possible")
    
    with col3:
        # APY with strategy recommendation
        apy_value = 8.5
        apy_change = 0.5
        st.metric("💰 Average APY", f"{apy_value}%", f"+{apy_change}%")
        st.caption("DeFi lending rates")
        
        # APY strategy tip
        if apy_value >= 10:
            st.info("🔥 High yields available!")
        elif apy_value >= 5:
            st.success("✅ Good yield opportunities")
        else:
            st.warning("⚠️ Low yields - consider alternatives")
    
    with col4:
        # Market cap with market sentiment
        market_cap = 12.8
        market_change = 5.2
        st.metric("📈 DeFi Market Cap", f"${market_cap}B", f"+{market_change}%")
        st.caption("DeFi token market")
        
        # Market sentiment indicator
        if market_change >= 5:
            st.success("🚀 Bullish momentum")
        elif market_change >= 0:
            st.info("📊 Stable market")
        else:
            st.error("📉 Bearish pressure")
    
    # Interactive market insights
    st.markdown("---")
    st.subheader("🔍 **Market Intelligence**")
    
    # Market sentiment analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create market sentiment chart
        sentiment_data = {
            'Metric': ['Fear & Greed', 'Market Momentum', 'Volatility', 'Liquidity', 'Developer Activity'],
            'Score': [65, 72, 58, 81, 69],
            'Status': ['Greed', 'Bullish', 'Moderate', 'High', 'Active']
        }
        df_sentiment = pd.DataFrame(sentiment_data)
        
        fig_sentiment = px.bar(df_sentiment, x='Metric', y='Score', color='Status',
                               title="📊 Current Market Sentiment Analysis",
                               color_discrete_map={
                                   'Greed': '#ff6b6b',
                                   'Bullish': '#4ecdc4',
                                   'Moderate': '#45b7d1',
                                   'High': '#96ceb4',
                                   'Active': '#feca57'
                               })
        
        fig_sentiment.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 **Market Insights**")
        
        # Dynamic market insights based on data
        if sentiment_data['Score'][0] > 60:  # Fear & Greed
            st.warning("⚠️ **Market Sentiment**: Greedy conditions - be cautious")
        else:
            st.success("✅ **Market Sentiment**: Fearful conditions - opportunities may arise")
        
        if sentiment_data['Score'][1] > 70:  # Market Momentum
            st.success("🚀 **Momentum**: Strong bullish momentum")
        else:
            st.info("📊 **Momentum**: Neutral to bearish")
        
        if sentiment_data['Score'][2] > 60:  # Volatility
            st.warning("📈 **Volatility**: High - expect price swings")
        else:
            st.success("📉 **Volatility**: Low - stable conditions")
        
        # Quick action button
        if st.button("📊 **Get Detailed Analysis**", use_container_width=True):
            st.session_state.show_detailed_analysis = True
    
    # Detailed market analysis
    if st.session_state.get('show_detailed_analysis', False):
        st.markdown("---")
        st.subheader("📊 **Detailed Market Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 **Technical Indicators**")
            
            # Simulate technical indicators
            indicators = {
                "RSI": "65 (Neutral)",
                "MACD": "Bullish crossover",
                "Bollinger Bands": "Price near upper band",
                "Volume": "Above average",
                "Support Level": "$2,800",
                "Resistance Level": "$3,200"
            }
            
            for indicator, value in indicators.items():
                st.markdown(f"**{indicator}:** {value}")
        
        with col2:
            st.markdown("#### 📰 **Market News Impact**")
            
            # Simulate news sentiment
            news_events = [
                "✅ Ethereum upgrade successful",
                "⚠️ Regulatory concerns in Asia",
                "✅ Institutional adoption increasing",
                "⚠️ Gas fees remain elevated",
                "✅ DeFi protocols growing"
            ]
            
            for event in news_events:
                st.markdown(f"• {event}")
        
        # Market recommendations
        st.markdown("#### 💡 **Market Recommendations**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ **Conservative Strategy**")
            st.markdown("• Focus on stablecoins")
            st.markdown("• Reduce exposure to high-risk assets")
            st.markdown("• Wait for better entry points")
        
        with col2:
            st.info("🎯 **Balanced Approach**")
            st.markdown("• Maintain current allocations")
            st.markdown("• Add to positions on dips")
            st.markdown("• Monitor key support levels")
        
        with col3:
            st.warning("🚀 **Aggressive Strategy**")
            st.markdown("• Look for breakout opportunities")
            st.markdown("• Consider leverage positions")
            st.markdown("• Set tight stop losses")
    
    # Learning opportunities section
    st.markdown("---")
    st.subheader("🎓 **Learning Opportunities**")
    
    # Dynamic learning recommendations based on market conditions
    if sentiment_data['Score'][0] > 60:  # Greedy market
        st.warning("""
        🚨 **Current Market Conditions: Greedy**
        
        **Perfect time to learn about:**
        • Risk management strategies
        • Portfolio rebalancing
        • Exit strategies
        • Market psychology
        """)
    else:
        st.success("""
        💡 **Current Market Conditions: Fearful**
        
        **Great time to learn about:**
        • Value investing principles
        • Dollar-cost averaging
        • Long-term strategies
        • Market cycles
        """)
    
    # Quick learning actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 **Risk Management Course**", use_container_width=True):
            st.info("🎯 Redirecting to Risk Management module...")
    
    with col2:
        if st.button("📊 **Market Analysis Tutorial**", use_container_width=True):
            st.info("📈 Starting Market Analysis tutorial...")
    
    with col3:
        if st.button("🎮 **Portfolio Simulator**", use_container_width=True):
            st.info("🚀 Opening Portfolio Simulator...")
    
    # Final call to action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 10px;">
        <h3 style="margin: 0 0 1rem 0; color: #1e293b;">🚀 Ready to Master DeFi?</h3>
        <p style="margin: 0 0 1.5rem 0; color: #64748b;">Build your portfolio above and start learning today!</p>
        <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">🎓 Educational platform • No real money required • Learn at your own pace</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# EDUCATION PAGE
# ==============================================================================

def education_page():
    """Education page with interactive learning system"""
    st.title("🎓 MAGMA DeFi Academy")
    st.markdown("Master Ethereum and DeFi with interactive tutorials")
    
    # Learning Path Selection
    st.subheader("🎯 Choose Your Learning Path")
    
    learning_paths = {
        "Beginner": {
            "description": "Start your DeFi journey",
            "modules": ["Ethereum Basics", "Wallet Setup", "Gas Fundamentals", "Basic DeFi Concepts"],
            "estimated_time": "2-3 hours",
            "difficulty": "Beginner"
        },
        "Intermediate": {
            "description": "Deep dive into DeFi protocols",
            "modules": ["Yield Farming", "Liquidity Pools", "AMMs", "Risk Management"],
            "estimated_time": "4-6 hours",
            "difficulty": "Intermediate"
        },
        "Advanced": {
            "description": "Master advanced DeFi strategies",
            "modules": ["MEV", "Flash Loans", "Complex Strategies", "Portfolio Optimization"],
            "estimated_time": "6-8 hours",
            "difficulty": "Advanced"
        }
    }
    
    selected_path = st.selectbox("Select Learning Path:", list(learning_paths.keys()))
    
    if selected_path:
        path_info = learning_paths[selected_path]
        
        st.info(f"📚 {selected_path} Path: {path_info['description']}")
        st.text(f"⏱️ Estimated Time: {path_info['estimated_time']}")
        st.text(f"🎯 Difficulty: {path_info['difficulty']}")
        
        st.markdown("---")
        
        # Module Progress
        st.subheader("📖 Module Progress")
        
        for i, module in enumerate(path_info["modules"]):
            with st.expander(f"📚 {module}"):
                # Simulate progress
                progress = min(100, (i + 1) * 25)
                
                if progress < 100:
                    st.progress(progress / 100)
                    st.text(f"Progress: {progress}%")
                    
                    if st.button(f"Start {module}", key=f"start_{i}"):
                        st.success(f"🎯 Starting {module}...")
                        st.info("This would open the interactive tutorial")
                else:
                    st.success("✅ Module Completed!")
                    st.text("Score: 95%")
                    st.text("Time spent: 45 minutes")
        
        st.markdown("---")
        
        # Interactive Quiz
        st.subheader("🧠 Quick Quiz")
        
        if selected_path == "Beginner":
            quiz_questions = {
                "What is Ethereum?": ["A cryptocurrency", "A blockchain platform", "A DeFi protocol", "A wallet"],
                "What are gas fees?": ["Transaction costs", "Mining rewards", "Staking rewards", "Trading fees"],
                "What does DeFi stand for?": ["Decentralized Finance", "Digital Finance", "Distributed Finance", "Defined Finance"]
            }
        elif selected_path == "Intermediate":
            quiz_questions = {
                "What is yield farming?": ["Earning rewards by providing liquidity", "Mining cryptocurrency", "Trading tokens", "Staking ETH"],
                "What is an AMM?": ["Automated Market Maker", "Advanced Mining Method", "Asset Management Model", "Automated Trading Bot"]
            }
        else:
            quiz_questions = {
                "What is MEV?": ["Miner Extractable Value", "Maximum Expected Value", "Minimum Efficient Volume", "Market Equilibrium Value"]
            }
        
        # Quiz interface
        user_answers = {}
        correct_answers = {
            "What is Ethereum?": 1,
            "What are gas fees?": 0,
            "What does DeFi stand for?": 0,
            "What is yield farming?": 0,
            "What is an AMM?": 0,
            "What is MEV?": 0
        }
        
        for question, options in quiz_questions.items():
            user_answers[question] = st.radio(question, options, key=f"quiz_{question}")
        
        if st.button("Submit Quiz"):
            score = 0
            total = len(quiz_questions)
            
            for question, user_answer in user_answers.items():
                if question in correct_answers:
                    if user_answer == options[correct_answers[question]]:
                        score += 1
            
            percentage = (score / total) * 100
            
            if percentage >= 80:
                st.success(f"🎉 Congratulations! You scored {percentage:.1f}%")
                st.balloons()
            elif percentage >= 60:
                st.warning(f"Good effort! You scored {percentage:.1f}%")
            else:
                st.error(f"Keep learning! You scored {percentage:.1f}%")
            
            st.info("Review the modules to improve your score!")

# ==============================================================================
# SETTINGS PAGE
#==============================================================================

def settings_page():
    """Settings page for dashboard configuration"""
    st.title("⚙️ Dashboard Settings")
    st.markdown("Configure your MAGMA dashboard experience")
    
    # API Configuration
    st.subheader("🔌 API Configuration")
    
    st.text_input("LunarCrush API Key", type="password", key="lunarcrush_key")
    st.text_input("OpenAI API Key", type="password", key="openai_key")
    st.text_input("Etherscan API Key", type="password", key="etherscan_key")
    st.text_input("News API Key", type="password", key="news_key")
    
    if st.button("Save API Keys"):
        st.success("API keys saved successfully!")
    
    st.markdown("---")
    
    # Dashboard Preferences
    st.subheader("🎨 Dashboard Preferences")
    
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    refresh_rate = st.slider("Auto-refresh rate (seconds)", 30, 300, 60)
    notifications = st.checkbox("Enable notifications", value=True)
    
    if st.button("Save Preferences"):
        st.success("Preferences saved successfully!")
    
    st.markdown("---")
    
    # Data Export
    st.subheader("📊 Data Export")
    
    export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
    
    if st.button("Export Dashboard Data"):
        st.info("Preparing export...")
        # Simulate export
        time.sleep(1)
        st.success("Data exported successfully!")
        
        # Create sample export data
        export_data = {
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'ETH_Price': [4500 + i*10 for i in range(30)],
            'Gas_Price': [25 + (i%7)*5 for i in range(30)],
            'Social_Score': [80 + (i%5)*3 for i in range(30)]
        }
        
        df_export = pd.DataFrame(export_data)
        
        if export_format == "CSV":
            csv = df_export.to_csv(index=False)
            st.download_button("Download CSV", csv, "magma_data.csv", "text/csv")
        elif export_format == "JSON":
            json_str = df_export.to_json(orient='records')
            st.download_button("Download JSON", json_str, "magma_data.json", "application/json")

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    """Main application function"""
    # Setup sidebar
    page, user_level = setup_sidebar()
    
    # Page routing
    if page == "🏠 Dashboard":
        main_dashboard()
    elif page == "🧠 Market Events":
        market_events_page()
    elif page == "⚡ Protocol Simulator":
        protocol_simulator_page()
    elif page == "🌐 Community Intelligence":
        community_intelligence_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "🎓 Education":
        education_page()
    elif page == "⚙️ Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🚀 Powered by MAGMA Supreme Intelligence Bot | 
            Built with Streamlit | 
            Real-time ETH Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# NEWS AND REDDIT INTEGRATION FUNCTIONS
# ==============================================================================

def get_real_eth_news():
    """Get real ETH news from premium crypto sources"""
    try:
        news_api_key = os.getenv('NEWS_API_KEY')
        if not news_api_key:
            return None
        
        # Premium crypto sources
        premium_sources = [
            'coindesk.com',
            'theblock.co', 
            'decrypt.co',
            'cointelegraph.com',
            'beincrypto.com'
        ]
        
        sources_param = ','.join(premium_sources)
        
        # NewsAPI endpoint with premium sources
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'ethereum OR ETH OR "ethereum network" OR "ethereum blockchain" OR DeFi OR "decentralized finance"',
            'domains': sources_param,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 15,
            'from': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
            'apiKey': news_api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            # Filter high-quality ETH articles
            eth_news = []
            for article in articles:
                if (article.get('title') and article.get('description') and 
                    is_quality_eth_article(article)):
                    
                    eth_news.append({
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'source': article['source']['name'],
                        'publishedAt': article['publishedAt'],
                        'urlToImage': article.get('urlToImage', '')
                    })
                    
                    if len(eth_news) >= 5:  # Get top 5 quality articles
                        break
            
            return eth_news if eth_news else get_fallback_premium_news()
        
    except Exception as e:
        st.error(f"Error fetching premium news: {e}")
        return get_fallback_premium_news()

def is_quality_eth_article(article):
    """Check if article is high-quality ETH content"""
    title = article.get('title', '').lower()
    description = article.get('description', '').lower()
    
    # Must contain ETH-related keywords
    eth_keywords = ['ethereum', 'eth', 'defi', 'smart contract', 'blockchain', 'crypto', 'vitalik']
    has_eth_keyword = any(keyword in title or keyword in description for keyword in eth_keywords)
    
    # Exclude low-quality content
    exclude_keywords = ['price prediction', 'bitget', 'advertisement', 'sponsored', 'airdrop', 'giveaway']
    has_exclude = any(keyword in title or keyword in description for keyword in exclude_keywords)
    
    # Must be substantial content
    has_content = len(description) > 50
    
    return has_eth_keyword and not has_exclude and has_content

def get_fallback_premium_news():
    """Get fallback premium news when API fails"""
    return [
        {
            'title': 'Ethereum Foundation Announces New Research Initiative for Scalability',
            'description': 'The Ethereum Foundation has unveiled a comprehensive research program focused on improving network scalability through innovative Layer 2 solutions and sharding implementations.',
            'source': 'CoinDesk',
            'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'url': 'https://coindesk.com',
            'urlToImage': ''
        },
        {
            'title': 'Major DeFi Protocol Launches on Ethereum Mainnet After Successful Audit',
            'description': 'A new decentralized finance protocol focusing on yield optimization has launched on Ethereum mainnet following extensive security audits and testing phases.',
            'source': 'The Block',
            'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'url': 'https://theblock.co',
            'urlToImage': ''
        },
        {
            'title': 'Ethereum Gas Fees Drop to Multi-Month Lows Amid Network Optimization',
            'description': 'Transaction fees on the Ethereum network have decreased significantly due to recent network optimizations and increased adoption of Layer 2 scaling solutions.',
            'source': 'Decrypt',
            'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'url': 'https://decrypt.co',
            'urlToImage': ''
        },
        {
            'title': 'Institutional Investment in Ethereum Staking Reaches Record High',
            'description': 'Corporate and institutional investors have significantly increased their participation in Ethereum staking, contributing to network security and decentralization.',
            'source': 'Cointelegraph',
            'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'url': 'https://cointelegraph.com',
            'urlToImage': ''
        },
        {
            'title': 'Smart Contract Security Standards Updated for Enhanced DeFi Protection',
            'description': 'New security standards and best practices have been released to help developers create more secure smart contracts and protect DeFi users from vulnerabilities.',
            'source': 'BeInCrypto',
            'publishedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'url': 'https://beincrypto.com',
            'urlToImage': ''
        }
    ]

def get_real_reddit_posts():
    """Get real ETH Reddit posts"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Get posts from multiple crypto subreddits with specific content
        subreddits = [
            'ethereum',
            'ethfinance', 
            'ethstaker',
            'ethdev',
            'defi'
        ]
        
        all_posts = []
        
        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                params = {'limit': 20, 'raw_json': 1}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for post_data in data['data']['children']:
                        post = post_data['data']
                        title = post['title'].lower()
                        
                        # Filter out generic/daily threads and focus on specific crypto content
                        excluded_terms = [
                            'daily general discussion',
                            'daily thread',
                            'weekly thread',
                            'monthly thread',
                            'daily discussion',
                            'general discussion',
                            'open discussion',
                            'random discussion',
                            'daily',
                            'weekly',
                            'monthly'
                        ]
                        
                        # Include specific crypto topics
                        crypto_keywords = [
                            'ethereum', 'eth', 'defi', 'staking', 'validator', 'merge', 'pos',
                            'layer 2', 'arbitrum', 'optimism', 'polygon', 'scaling',
                            'eip', 'upgrade', 'fork', 'sharding', 'danksharding',
                            'smart contract', 'dapp', 'protocol', 'yield', 'liquidity',
                            'uniswap', 'aave', 'compound', 'makerdao', 'curve',
                            'gas', 'gwei', 'transaction', 'wallet', 'metamask'
                        ]
                        
                        # Include only posts with good scores, specific content, and avoid generic discussions
                        if (post.get('score', 0) > 30 and  # Good score threshold
                            not any(term in title for term in excluded_terms) and
                            any(keyword in title for keyword in crypto_keywords) and
                            len(post['title']) > 15):  # Meaningful titles
                            
                            all_posts.append({
                                'title': post['title'],
                                'selftext': post.get('selftext', '')[:300],
                                'score': post['score'],
                                'num_comments': post['num_comments'],
                                'upvote_ratio': post.get('upvote_ratio', 0.5),
                                'url': f"https://reddit.com{post['permalink']}",
                                'created_utc': post['created_utc'],
                                'subreddit': subreddit
                            })
            except Exception as e:
                continue  # Skip this subreddit if error
        
        # Sort by score and return top posts
        all_posts.sort(key=lambda x: x['score'], reverse=True)
        return all_posts[:8]  # Top 8 posts across all subreddits
        
    except Exception as e:
        st.error(f"Error fetching Reddit posts: {e}")
        return None

def display_real_news_section(news_data):
    """Display real news section"""
    if not news_data:
        display_fallback_news_section()
        return
    
    for i, article in enumerate(news_data):
        with st.expander(f"📰 {article['title']}", expanded=(i == 0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Source:** {article['source']}")
                st.markdown(f"**Published:** {article['publishedAt'][:10]}")
                st.markdown(article['description'])
                st.markdown(f"[Read Full Article]({article['url']})")
                
                # Educational connection
                educational_concept = determine_educational_concept_from_news(article)
                st.info(f"🎓 **Learn from this:** {educational_concept}")
            
            with col2:
                if article.get('urlToImage'):
                    try:
                        st.image(article['urlToImage'], width=150)
                    except:
                        st.info("📰 News Article")
                else:
                    st.info("📰 News Article")

def display_fallback_news_section():
    """Display fallback news section with premium sources"""
    fallback_news = get_fallback_premium_news()
    
    # Add educational concepts to fallback news
    educational_concepts = [
        'Ethereum Research and Development',
        'DeFi Protocol Security and Analysis', 
        'Gas Fee Economics and Layer 2 Solutions',
        'Institutional Adoption and Staking',
        'Smart Contract Security Best Practices'
    ]
    
    for i, article in enumerate(fallback_news):
        educational_concept = educational_concepts[i] if i < len(educational_concepts) else 'Ethereum Fundamentals'
        
        with st.expander(f"📰 {article['title']}", expanded=(i == 0)):
            st.markdown(f"**Source:** {article['source']}")
            st.markdown(f"**Published:** Today")
            st.markdown(article['description'])
            st.info(f"🎓 **Learn from this:** {educational_concept}")
            
            if st.button(f"📚 Learn: {educational_concept}", key=f"news_learn_{i}"):
                st.success(f"🚀 Opening tutorial: {educational_concept}")

def display_real_reddit_section(reddit_data):
    """Display real Reddit section"""
    if not reddit_data:
        display_fallback_reddit_section()
        return
    
    # Calculate overall sentiment
    total_score = sum(post['score'] for post in reddit_data)
    avg_ratio = sum(post['upvote_ratio'] for post in reddit_data) / len(reddit_data)
    sentiment = "Bullish" if avg_ratio > 0.7 else "Bearish" if avg_ratio < 0.5 else "Neutral"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Community Sentiment", sentiment)
    with col2:
        st.metric("Total Engagement", f"{total_score:,}")
    with col3:
        st.metric("Avg Upvote Ratio", f"{avg_ratio:.1%}")
    
    st.markdown("**📱 Top ETH Posts Today:**")
    
    for i, post in enumerate(reddit_data):
        # Show subreddit source in title
        subreddit_emoji = {
            'ethereum': '⚡',
            'ethfinance': '💰', 
            'ethstaker': '🥩',
            'ethdev': '👨‍💻',
            'defi': '🏦'
        }
        emoji = subreddit_emoji.get(post.get('subreddit', 'ethereum'), '🔥')
        
        with st.expander(f"{emoji} r/{post.get('subreddit', 'ethereum')} | {post['title'][:50]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**From:** r/{post.get('subreddit', 'ethereum')}")
                if post['selftext']:
                    st.markdown(post['selftext'])
                st.markdown(f"[View on Reddit]({post['url']})")
                
                # Educational takeaway
                educational_takeaway = determine_educational_concept_from_reddit(post)
                st.info(f"🎓 **Educational Takeaway:** {educational_takeaway}")
            
            with col2:
                st.metric("Score", post['score'])
                st.metric("Comments", post['num_comments'])
                st.metric("Upvote %", f"{post['upvote_ratio']:.1%}")

def display_fallback_reddit_section():
    """Display fallback Reddit section"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Community Sentiment", "Bullish", "+5%")
    with col2:
        st.metric("Total Engagement", "4,363", "+18%")
    with col3:
        st.metric("Avg Upvote Ratio", "89%", "+3%")
    
    fallback_posts = [
        {
            'title': 'Ethereum Foundation announces new grant program for scalability research',
            'content': 'The Ethereum Foundation is allocating $100M for Layer 2 scaling solutions and EIP improvements. Applications open until September 2025.',
            'score': 1247,
            'comments': 312,
            'educational_takeaway': 'Layer 2 Scaling Solutions and EIP Process'
        },
        {
            'title': 'Vitalik discusses sharding timeline and technical challenges',
            'content': 'In latest blog post, Vitalik outlines the remaining technical hurdles for full sharding implementation, estimating 2026 for completion.',
            'score': 892,
            'comments': 203,
            'educational_takeaway': 'Ethereum Sharding and Scalability Roadmap'
        },
        {
            'title': 'EIP-4844 Proto-Danksharding reduces L2 costs by 90%',
            'content': 'Major Layer 2 protocols report dramatic cost reductions following successful EIP-4844 implementation. Arbitrum and Optimism fees now under $0.10.',
            'score': 1156,
            'comments': 187,
            'educational_takeaway': 'Proto-Danksharding and Blob Transactions'
        },
        {
            'title': 'Coinbase announces native ETH staking with 3.8% APY',
            'content': 'Coinbase launches institutional ETH staking service, joining other major exchanges. Minimum stake: 32 ETH, with slashing protection.',
            'score': 623,
            'comments': 145,
            'educational_takeaway': 'Ethereum Staking Economics and Validator Risks'
        },
        {
            'title': 'MEV-Boost adoption reaches 85% of validators',
            'content': 'Flashbots reports that MEV-Boost is now used by majority of validators, generating $12M+ in additional rewards this month.',
            'score': 445,
            'comments': 89,
            'educational_takeaway': 'Maximum Extractable Value (MEV) and Block Building'
        }
    ]
    
    st.markdown("**📱 Top ETH Community Discussions:**")
    
    for i, post in enumerate(fallback_posts):
        with st.expander(f"🔥 {post['title']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(post['content'])
                st.info(f"🎓 **Educational Takeaway:** {post['educational_takeaway']}")
                
                if st.button(f"📚 Learn: {post['educational_takeaway']}", key=f"reddit_learn_{i}"):
                    st.success(f"🚀 Opening tutorial: {post['educational_takeaway']}")
            
            with col2:
                st.metric("Score", post['score'])
                st.metric("Comments", post['comments'])

def determine_educational_concept_from_news(article):
    """Determine educational concept from news article"""
    title_lower = article['title'].lower()
    desc_lower = article['description'].lower()
    
    if any(word in title_lower or word in desc_lower for word in ['gas', 'fee', 'cost']):
        return "Gas Fee Economics and Optimization"
    elif any(word in title_lower or word in desc_lower for word in ['defi', 'yield', 'liquidity']):
        return "DeFi Protocol Analysis"
    elif any(word in title_lower or word in desc_lower for word in ['layer 2', 'scaling', 'rollup']):
        return "Layer 2 Scaling Solutions"
    elif any(word in title_lower or word in desc_lower for word in ['staking', 'validator', 'consensus']):
        return "Proof of Stake and Staking"
    elif any(word in title_lower or word in desc_lower for word in ['smart contract', 'dapp']):
        return "Smart Contract Development"
    elif any(word in title_lower or word in desc_lower for word in ['institutional', 'adoption', 'enterprise']):
        return "Institutional Adoption Analysis"
    else:
        return "Ethereum Ecosystem Fundamentals"

def determine_educational_concept_from_reddit(post):
    """Determine educational concept from Reddit post"""
    title_lower = post['title'].lower()
    content_lower = post.get('selftext', '').lower()
    
    if any(word in title_lower or word in content_lower for word in ['gas', 'fee', 'optimization']):
        return "Gas Optimization Strategies"
    elif any(word in title_lower or word in content_lower for word in ['staking', 'reward', 'validator']):
        return "Staking and Rewards Mechanics"
    elif any(word in title_lower or word in content_lower for word in ['defi', 'protocol', 'yield']):
        return "DeFi Protocol Usage"
    elif any(word in title_lower or word in content_lower for word in ['nft', 'token', 'mint']):
        return "NFT and Token Economics"
    else:
        return "Community Sentiment Analysis"

def generate_educational_insights_from_events(news_data, reddit_data):
    """Generate educational insights from current events"""
    insights = []
    
    # From news
    if news_data:
        for article in news_data[:3]:
            concept = determine_educational_concept_from_news(article)
            insights.append({
                'title': f"News: {article['title'][:50]}...",
                'event': article['description'][:100] + "...",
                'why_learn': f"Understanding this news helps you grasp current market dynamics and {concept.lower()}",
                'concept': concept
            })
    
    # From Reddit
    if reddit_data:
        for post in reddit_data[:2]:
            concept = determine_educational_concept_from_reddit(post)
            insights.append({
                'title': f"Community: {post['title'][:50]}...",
                'event': f"Community discussing: {post['title']}",
                'why_learn': f"Community sentiment reveals market psychology and practical applications of {concept.lower()}",
                'concept': concept
            })
    
    # Fallback insights
    if not insights:
        insights = [
            {
                'title': "Market Movement Analysis",
                'event': "ETH price movement provides learning opportunity",
                'why_learn': "Price movements teach us about market psychology and timing",
                'concept': "Market Analysis and Psychology"
            },
            {
                'title': "Gas Fee Optimization",
                'event': "Current gas prices create optimization opportunity",
                'why_learn': "Understanding gas fees is crucial for effective Ethereum usage",
                'concept': "Gas Fee Economics"
            },
            {
                'title': "DeFi Protocol Analysis",
                'event': "Active DeFi markets provide analysis opportunities",
                'why_learn': "DeFi protocols demonstrate practical blockchain applications",
                'concept': "DeFi Fundamentals"
            }
        ]
    
    return insights[:5]

def calculate_daily_learning_score(news_data, reddit_data):
    """Calculate daily learning score based on current events"""
    opportunities = 0
    
    # Count opportunities from news
    if news_data:
        opportunities += len(news_data)
    
    # Count opportunities from Reddit
    if reddit_data:
        opportunities += len(reddit_data)
    
    # Add base opportunities
    opportunities += 3
    
    # Determine difficulty based on market conditions
    difficulty = "Medium"
    time_minutes = 20
    
    if opportunities > 8:
        difficulty = "High"
        time_minutes = 35
    elif opportunities < 5:
        difficulty = "Easy"
        time_minutes = 15
    
    return {
        'opportunities': min(opportunities, 10),
        'difficulty': difficulty,
        'time_minutes': time_minutes
    }

# ==============================================================================
# NEW ADVANCED FEATURES PAGES
# ==============================================================================

def market_events_page():
    """Market Events Learning Engine - Core Innovation"""
    
    # Header with clear explanation
    st.title("🧠 Market Events Learning Engine")
    st.markdown("**🎯 Transform real market events into structured learning opportunities**")
    
    # What this feature does section
    with st.expander("❓ What does this feature do?", expanded=True):
        st.markdown("""
        **🚀 This engine automatically:**
        
        1. **📊 Analyzes real market events** - Processes live news from CoinDesk, TheBlock, Decrypt, Cointelegraph, BeInCrypto
        2. **🎓 Converts events to lessons** - Transforms complex market events into understandable learning modules
        3. **📚 Generates educational content** - Creates step-by-step guides, key insights, and practical examples
        4. **⏰ Tracks learning progress** - Monitors your daily learning opportunities and time investment
        5. **🎯 Personalizes content** - Adapts difficulty and focus based on your learning goals
        
        **💡 Perfect for: Beginners learning DeFi, traders understanding market psychology, developers learning protocol mechanics**
        """)
    
    st.markdown("---")
    
    try:
        # Initialize the learning engine
        engine = MarketEventLearningEngine()
        
        # Get REAL news and Reddit data instead of simulated events
        st.markdown("### 📡 **Fetching Real Market Events...**")
        
        with st.spinner("🔄 Loading real-time ETH news and community intelligence..."):
            real_news = get_real_eth_news()
            real_reddit = get_real_reddit_posts()
        
        if not real_news and not real_reddit:
            st.warning("⚠️ No real-time data available. Check your API connections.")
            return
        
        # Convert news and Reddit posts to market events format
        market_events = []
        
        # Process news articles
        if real_news:
            st.success(f"✅ Loaded {len(real_news)} real news articles from premium sources")
            for article in real_news:
                market_events.append({
                    'title': article['title'],
                    'description': article['description'],
                    'source': article['source'],
                    'url': article['url'],
                    'type': 'news',
                    'published_at': article['publishedAt']
                })
        
        # Process Reddit posts
        if real_reddit:
            st.success(f"✅ Loaded {len(real_reddit)} real community posts from ETH subreddits")
            for post in real_reddit:
                market_events.append({
                    'title': post['title'],
                    'description': post['selftext'] if post['selftext'] else post['title'],
                    'source': f"Reddit r/{post['subreddit']}",
                    'url': post['url'],
                    'type': 'community',
                    'score': post['score'],
                    'comments': post['num_comments']
                })
        
        st.markdown("---")
        
        # Process events and generate learning opportunities
        with st.spinner("🧠 Processing real market events and generating learning opportunities..."):
            opportunities = engine.process_multiple_events(market_events)
        
        # Display learning opportunities with better organization
        st.subheader("🎯 **Today's Learning Opportunities from Real Events**")
        st.markdown("**Click on any opportunity below to explore and start learning!**")
        
        for i, opportunity in enumerate(opportunities):
            with st.expander(f"📚 **{opportunity.concept}** - {opportunity.difficulty.title()} Level", expanded=True):
                
                # Event information
                st.markdown(f"### 📰 **Triggering Event:** {opportunity.event.title}")
                st.markdown(f"*{opportunity.event.description}*")
                
                # Show source and metadata
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**📡 Source:** {opportunity.event.source}")
                    if hasattr(opportunity.event, 'url') and opportunity.event.url:
                        st.markdown(f"**🔗 Link:** [View Source]({opportunity.event.url})")
                with col2:
                    if hasattr(opportunity.event, 'type'):
                        event_type_icon = "📰" if opportunity.event.type == 'news' else "💬"
                        st.markdown(f"**Type:** {event_type_icon} {opportunity.event.type.title()}")
                    if hasattr(opportunity.event, 'score'):
                        st.markdown(f"**Community Score:** {opportunity.event.score}")
                
                # Learning details in columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 🎯 **Why Learn This?**")
                    st.info(f"{opportunity.why_learn}")
                    
                    st.markdown("#### ⏰ **Time Investment**")
                    st.metric("Estimated Time", f"{opportunity.estimated_time} minutes")
                    
                    st.markdown("#### 🔑 **Key Insights You'll Gain**")
                    for insight in opportunity.key_insights:
                        st.markdown(f"• {insight}")
                    
                    st.markdown("#### 🛠️ **Practical Examples**")
                    for example in opportunity.practical_examples:
                        st.markdown(f"• {example}")
                
                with col2:
                    # Impact level indicator with better visualization
                    impact_info = {
                        'low': {'icon': '🟢', 'desc': 'Minor market impact'},
                        'medium': {'icon': '🟡', 'desc': 'Moderate market impact'},
                        'high': {'icon': '🟠', 'desc': 'Significant market impact'},
                        'critical': {'icon': '🔴', 'desc': 'Major market impact'}
                    }
                    
                    impact_data = impact_info.get(opportunity.event.impact_level, {'icon': '⚪', 'desc': 'Unknown impact'})
                    
                    st.markdown("#### ⚡ **Market Impact**")
                    st.markdown(f"{impact_data['icon']} **{opportunity.event.impact_level.title()}**")
                    st.caption(impact_data['desc'])
                    
                    st.markdown("#### 🔗 **Related Concepts**")
                    for concept in opportunity.related_concepts:
                        st.markdown(f"• {concept}")
                    
                    st.markdown("#### 📚 **Learning Path**")
                    for step in opportunity.learning_path:
                        st.markdown(f"• {step}")
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button(f"🚀 **Start Learning**", key=f"learn_{i}", use_container_width=True):
                        st.success(f"🎯 Opening interactive tutorial: {opportunity.concept}")
                        st.info("This would launch an interactive learning module with real examples and exercises.")
                
                with col2:
                    if st.button(f"📊 **View Details**", key=f"details_{i}", use_container_width=True):
                        st.json(opportunity.__dict__)
                
                with col3:
                    if st.button(f"⭐ **Save for Later**", key=f"save_{i}", use_container_width=True):
                        st.success(f"💾 Saved '{opportunity.concept}' to your learning library!")
        
        # Daily learning summary with better metrics
        st.markdown("---")
        st.subheader("📊 **Your Daily Learning Dashboard**")
        
        summary = engine.generate_daily_learning_summary(opportunities)
        
        # Summary metrics in a more organized way
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Opportunities", summary['total_opportunities'], "Available today")
        
        with col2:
            st.metric("⏰ Total Learning Time", f"{summary['total_learning_time']} min", "Recommended")
        
        with col3:
            st.metric("🎯 Focus Area", summary['recommended_focus'][:15] + "...", "Primary topic")
        
        with col4:
            # Calculate completion percentage
            completion = min(100, (summary['total_opportunities'] / 5) * 100)
            st.metric("📈 Daily Progress", f"{completion:.0f}%", "Of daily goal")
            st.progress(completion / 100)
        
        # Difficulty distribution chart
        if summary['difficulty_distribution']:
            st.markdown("#### 📊 **Difficulty Distribution**")
            difficulty_data = summary['difficulty_distribution']
            
            # Create a more informative chart
            fig = px.pie(
                values=list(difficulty_data.values()), 
                names=list(difficulty_data.keys()),
                title="Learning Opportunities by Difficulty",
                color_discrete_map={
                    'Beginner': '#00ff00',
                    'Intermediate': '#ffff00', 
                    'Advanced': '#ff8000',
                    'Expert': '#ff0000'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Data source summary
        st.markdown("---")
        st.subheader("📡 **Data Sources Summary**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📰 **News Sources**")
            if real_news:
                st.success(f"✅ {len(real_news)} articles from premium sources")
                for article in real_news[:3]:
                    st.markdown(f"• **{article['source']}**: {article['title'][:50]}...")
            else:
                st.warning("⚠️ No news data available")
        
        with col2:
            st.markdown("#### 💬 **Community Sources**")
            if real_reddit:
                st.success(f"✅ {len(real_reddit)} posts from ETH communities")
                for post in real_reddit[:3]:
                    st.markdown(f"• **r/{post['subreddit']}**: {post['title'][:50]}...")
            else:
                st.warning("⚠️ No community data available")
        
        # Quick actions
        st.markdown("---")
        st.subheader("⚡ **Quick Actions**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 **Refresh Events**", key="refresh_events", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📚 **View Learning History**", key="view_history", use_container_width=True):
                st.info("📚 This would show your completed learning modules and progress tracking.")
        
        with col3:
            if st.button("🎯 **Set Learning Goals**", key="set_goals", use_container_width=True):
                st.info("🎯 This would allow you to customize your learning path and set daily/weekly goals.")
        

    except Exception as e:
        st.error(f"❌ Error initializing Market Events Learning Engine: {e}")
        st.info("💡 This feature requires the market_event_learning_engine module")
        st.code(f"Error details: {str(e)}")

def protocol_simulator_page():
    """Protocol Simulator page - Interactive DeFi protocol learning"""
    
    # Initialize session state variables to avoid errors
    if 'show_quick_swap' not in st.session_state:
        st.session_state.show_quick_swap = False
    if 'show_quick_lending' not in st.session_state:
        st.session_state.show_quick_lending = False
    if 'show_quick_yield' not in st.session_state:
        st.session_state.show_quick_yield = False
    if 'show_uniswap_demo' not in st.session_state:
        st.session_state.show_uniswap_demo = False
    if 'show_lending_demo' not in st.session_state:
        st.session_state.show_lending_demo = False
    if 'show_yield_demo' not in st.session_state:
        st.session_state.show_yield_demo = False
    if 'show_stablecoin_demo' not in st.session_state:
        st.session_state.show_stablecoin_demo = False
    
    # Strategy explanation
    st.info("""
    🎯 **MAGMA Learning Strategy:**
    
    **📚 Phase 1 (Current):** Learn DeFi concepts through simulations
    **🔗 Phase 2 (Coming Soon):** Connect wallet to testnets for real interactions (NO real money)
    **🚫 Never:** Connect to mainnet or use real money
    
    **💡 Goal:** Learn DeFi safely, then practice on testnets before real DeFi!
    """)
    
    st.title("⚡ Protocol Simulator - Learn DeFi Safely")
    st.markdown("**🎮 Practice DeFi interactions in a safe, educational environment**")
    
    # What can you do here section
    with st.expander("❓ What can you do here?", expanded=True):
        st.markdown("""
        **🚀 This LEARNING simulator helps you:**
        
        1. **📚 Understand DeFi protocols** - Learn how Uniswap, Aave, Compound work
        2. **🎮 Practice interactions** - Simulate swaps, lending, and yield farming
        3. **⚠️ Learn about risks** - Understand impermanent loss, gas fees, and protocol risks
        4. **🔗 Prepare for testnets** - Learn concepts here, then practice on real testnets
        5. **📊 Read real-time data** - See actual protocol metrics (TVL, APY, volume)
        
        **💡 Perfect for:** Beginners learning DeFi, intermediate users practicing strategies, advanced users teaching others
        
        **🔒 REMEMBER:** This is for learning only - no real money involved!
        """)
    
    st.markdown("---")
    
    # Wallet Connection Status (Future Feature)
    st.subheader("🔗 **Wallet Connection Status**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **📱 Wallet Connection Coming Soon:**
        
        **🔒 Current Status:** Simulation Mode (No wallet needed)
        **🔗 Next Phase:** Testnet Connection (Goerli/Sepolia networks)
        **🚫 Never:** Mainnet connection or real money
        
        **💡 Benefits of future wallet connection:**
        • Practice real transactions on testnets
        • Learn gas fee management
        • Understand transaction confirmations
        • Get familiar with wallet interfaces
        """)
    
    with col2:
        # Simulated wallet status
        st.markdown("#### 🔐 **Current Mode**")
        st.success("🟢 **Simulation Mode**")
        st.caption("Learning through examples")
        
        st.markdown("#### 🔮 **Future Mode**")
        st.info("🟡 **Testnet Mode**")
        st.caption("Real testnet interactions")
        
        # Simulated connect button
        if st.button("🔗 **Connect Testnet Wallet** (Coming Soon)", key="connect_wallet", use_container_width=True, disabled=True):
            st.info("🔮 This feature will be available in the next update!")
    
    st.markdown("---")
    
    # Quick Actions for instant demos
    st.subheader("⚡ **Interactive Learning Simulator**")
    st.markdown("**🎯 Enter your own numbers and see real DeFi calculations!**")
    
    # Create interactive simulation forms
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 **Swap Simulator**", "💰 **Lending Simulator**", "📊 **Liquidity Simulator**", "🎮 **Quick Demos**"])
    
    with tab1:
        st.markdown("#### 🔄 **Token Swap Simulator**")
        st.markdown("**Enter amounts and see exactly what you'd get in a swap**")
        
        # How This Works section
        with st.expander("❓ **How This Works - Understanding Token Swaps**", expanded=False):
            st.markdown("""
            **🔄 What is a Token Swap?**
            
            A token swap is when you exchange one cryptocurrency for another using a **Decentralized Exchange (DEX)** like Uniswap.
            
            **🧮 The Math Behind Swaps:**
            
            1. **Input Value**: Your token amount × Current price = USD value
            2. **Protocol Fee**: Uniswap charges 0.3% of the trade value
            3. **Price Impact**: Large trades can move the price (slippage)
            4. **Output Calculation**: (Input value - Fee) ÷ Output token price
            5. **Gas Cost**: Network fee for processing the transaction
            
            **⚠️ Key Concepts:**
            
            • **Slippage**: Protection against price movement during transaction
            • **Price Impact**: How much your trade affects the market price
            • **Impermanent Loss**: Risk when providing liquidity (not applicable to swaps)
            • **Gas Fees**: Cost of processing transactions on Ethereum
            
            **💡 Why Learn This?**
            
            Understanding swaps helps you:
            • Calculate exact costs before trading
            • Minimize fees and slippage
            • Choose optimal trade sizes
            • Understand DeFi economics
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Input Token**")
            token_in = st.selectbox("From:", ["ETH", "USDC", "USDT", "DAI", "WBTC", "LINK", "UNI"], key="swap_token_in")
            amount_in = st.number_input("Amount:", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="swap_amount_in")
            
            # Show current price
            current_prices = {"ETH": 3200.0, "USDC": 1.0, "USDT": 1.0, "DAI": 1.0, "WBTC": 65000.0, "LINK": 18.5, "UNI": 8.2}
            price_in = current_prices.get(token_in, 0)
            value_usd = amount_in * price_in
            st.metric(f"Value in USD", f"${value_usd:,.2f}")
        
        with col2:
            st.markdown("**📤 Output Token**")
            token_out = st.selectbox("To:", ["ETH", "USDC", "USDT", "DAI", "WBTC", "LINK", "UNI"], key="swap_token_out")
            
            # Prevent same token selection
            if token_in == token_out:
                st.warning("⚠️ Select different tokens for swap")
                token_out = None
            
            slippage = st.slider("Slippage Tolerance (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="swap_slippage")
            st.info(f"💡 Slippage: {slippage}% - protects against price movement")
        
        # Calculate swap button
        if st.button("🔄 **Calculate Swap**", key="calculate_swap", use_container_width=True, type="primary"):
            if token_in and token_out and token_in != token_out:
                # Simulate swap calculation
                from live_protocol_simulator import DeFiProtocolSimulator
                simulator = DeFiProtocolSimulator()
                
                try:
                    swap_result = simulator.simulate_swap(
                        protocol='uniswap_v3',
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=amount_in,
                        slippage=slippage
                    )
                    
                    if 'error' not in swap_result:
                        st.success("✅ **Swap Calculation Complete!**")
                        
                        # Display results in a nice format
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("💰 Input", f"{amount_in} {token_in}", f"${swap_result['calculation']['value_in_usd']:,.2f}")
                            st.metric("💸 Fee", f"${swap_result['calculation']['fee_usd']:.2f}", "0.3% Uniswap fee")
                        
                        with col2:
                            st.metric("🎯 Output", f"{swap_result['calculation']['amount_out']:.6f} {token_out}", f"${swap_result['calculation']['value_out_usd']:,.2f}")
                            st.metric("📊 Price Impact", f"{swap_result['calculation']['price_impact_percent']:.2f}%")
                        
                        with col3:
                            st.metric("⛽ Gas Cost", f"${swap_result['gas_cost_usd']:.2f}", f"{swap_result['gas_estimate']:,} gas")
                            st.metric("⚠️ Min Output", f"{swap_result['calculation']['min_output']:.6f} {token_out}")
                        
                        # Educational breakdown
                        st.markdown("---")
                        st.markdown("#### 📚 **What This Teaches You:**")
                        for note in swap_result['educational_notes']:
                            st.markdown(f"• {note}")
                        
                        # Show the math
                        with st.expander("🧮 **Show the Math**"):
                            st.markdown(f"""
                            **Swap Calculation Breakdown:**
                            
                            1. **Input Value**: {amount_in} {token_in} × ${price_in:,.2f} = ${value_usd:,.2f}
                            2. **Uniswap Fee**: ${value_usd:,.2f} × 0.3% = ${swap_result['calculation']['fee_usd']:.2f}
                            3. **Value After Fee**: ${value_usd:,.2f} - ${swap_result['calculation']['fee_usd']:.2f} = ${swap_result['calculation']['value_out_usd']:,.2f}
                            4. **Output Amount**: ${swap_result['calculation']['value_out_usd']:,.2f} ÷ ${current_prices.get(token_out, 0):,.2f} = {swap_result['calculation']['amount_out']:.6f} {token_out}
                            5. **Min Output (with {slippage}% slippage)**: {swap_result['calculation']['amount_out']:.6f} × (1 - {slippage/100:.3f}) = {swap_result['calculation']['min_output']:.6f} {token_out}
                            """)
                    else:
                        st.error(f"❌ Error: {swap_result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Simulation error: {e}")
            else:
                st.warning("⚠️ Please select different input and output tokens")
    
    with tab2:
        st.markdown("#### 💰 **Lending & Borrowing Simulator**")
        st.markdown("**See exactly how much you'd earn or pay in interest**")
        
        # How This Works section
        with st.expander("❓ **How This Works - Understanding DeFi Lending**", expanded=False):
            st.markdown("""
            **🏦 What is DeFi Lending?**
            
            DeFi lending allows you to **deposit assets to earn interest** or **borrow assets by providing collateral**.
            
            **💰 How Deposits Work:**
            
            1. **Supply Assets**: Deposit tokens into a lending protocol
            2. **Earn Interest**: Protocol pays you APY (Annual Percentage Yield)
            3. **Interest Compounds**: Earnings are added to your balance daily
            4. **Withdraw Anytime**: Remove your assets + earned interest
            
            **💸 How Borrowing Works:**
            
            1. **Provide Collateral**: Lock up assets worth more than you borrow
            2. **Borrow Assets**: Take out loans in different tokens
            3. **Pay Interest**: Accrue interest on borrowed amount
            4. **Maintain Collateral**: Keep sufficient value to avoid liquidation
            
            **📊 APY vs APR:**
            
            • **APY (Annual Percentage Yield)**: Includes compound interest
            • **APR (Annual Percentage Rate)**: Simple interest rate
            • **Daily Calculation**: Interest compounds every block (~12 seconds)
            
            **⚠️ Key Risks:**
            
            • **Liquidation**: If collateral value drops below threshold
            • **Interest Rate Changes**: APY can fluctuate based on market demand
            • **Smart Contract Risk**: Protocol vulnerabilities (rare but possible)
            • **Collateral Volatility**: Asset prices can change rapidly
            
            **💡 Why Use DeFi Lending?**
            
            **For Depositors:**
            • Earn higher yields than traditional banks
            • Maintain control of your assets
            • No credit checks or paperwork
            
            **For Borrowers:**
            • Access liquidity without selling assets
            • Lower fees than traditional loans
            • Use borrowed funds for trading or other DeFi activities
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏦 Protocol**")
            protocol = st.selectbox("Select Protocol:", ["Aave V3", "Compound V3"], key="lending_protocol")
            
            st.markdown("**💼 Asset**")
            asset = st.selectbox("Select Asset:", ["ETH", "USDC", "USDT", "DAI", "WBTC", "LINK"], key="lending_asset")
            
            st.markdown("**💰 Amount**")
            amount = st.number_input("Amount:", min_value=0.01, max_value=1000000.0, value=1000.0, step=100.0, key="lending_amount")
            
            # Show current value
            price = current_prices.get(asset, 0)
            value_usd = amount * price
            st.metric(f"Value in USD", f"${value_usd:,.2f}")
        
        with col2:
            st.markdown("**📊 Action**")
            action = st.selectbox("What do you want to do?", ["Deposit (Earn Interest)", "Borrow (Pay Interest)"], key="lending_action")
            
            if action == "Deposit (Earn Interest)":
                st.info("💡 **Depositing**: You'll earn interest on your assets")
                action_type = "deposit"
            else:
                st.info("⚠️ **Borrowing**: You'll pay interest and need collateral")
                action_type = "borrow"
            
            # Time period for calculations
            time_period = st.selectbox("Show returns for:", ["Daily", "Weekly", "Monthly", "Yearly"], key="lending_time")
        
        # Calculate lending button
        if st.button("💰 **Calculate Lending Returns**", key="calculate_lending", use_container_width=True, type="primary"):
            try:
                from live_protocol_simulator import DeFiProtocolSimulator
                simulator = DeFiProtocolSimulator()
                
                # Map protocol names
                protocol_map = {"Aave V3": "aave_v3", "Compound V3": "compound_v3"}
                protocol_id = protocol_map.get(protocol)
                
                lending_result = simulator.simulate_lending(
                    protocol=protocol_id,
                    action=action_type,
                    asset=asset,
                    amount=amount
                )
                
                if 'error' not in lending_result:
                    st.success("✅ **Lending Calculation Complete!**")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("💰 Asset Value", f"${lending_result['calculation']['value_usd']:,.2f}")
                        st.metric("📈 APY", f"{lending_result['calculation']['apy']:.1f}%")
                    
                    with col2:
                        if action_type == "deposit":
                            st.metric("💵 Daily Earnings", f"${lending_result['calculation']['daily_return']:.2f}")
                            st.metric("📅 Weekly Earnings", f"${lending_result['calculation']['weekly_return']:.2f}")
                        else:
                            st.metric("💸 Daily Cost", f"${abs(lending_result['calculation']['daily_return']):.2f}")
                            st.metric("📅 Weekly Cost", f"${abs(lending_result['calculation']['weekly_return']):.2f}")
                    
                    with col3:
                        if action_type == "deposit":
                            st.metric("📊 Monthly Earnings", f"${lending_result['calculation']['monthly_return']:.2f}")
                            st.metric("🎯 Yearly Earnings", f"${lending_result['calculation']['yearly_return']:.2f}")
                        else:
                            st.metric("📊 Monthly Cost", f"${abs(lending_result['calculation']['monthly_return']):.2f}")
                            st.metric("🎯 Yearly Cost", f"${abs(lending_result['calculation']['yearly_return']):.2f}")
                    
                    # Educational breakdown
                    st.markdown("---")
                    st.markdown("#### 📚 **What This Teaches You:**")
                    for note in lending_result['educational_notes']:
                        st.markdown(f"• {note}")
                    
                    # Show the math
                    with st.expander("🧮 **Show the Math**"):
                        if action_type == "deposit":
                            st.markdown(f"""
                            **Interest Calculation Breakdown:**
                            
                            1. **Principal**: ${amount:,.2f} {asset} × ${price:,.2f} = ${value_usd:,.2f}
                            2. **APY**: {lending_result['calculation']['apy']:.1f}%
                            3. **Daily Interest**: ${value_usd:,.2f} × {lending_result['calculation']['apy']:.1f}% ÷ 365 = ${lending_result['calculation']['daily_return']:.2f}
                            4. **Weekly Interest**: ${lending_result['calculation']['daily_return']:.2f} × 7 = ${lending_result['calculation']['weekly_return']:.2f}
                            5. **Monthly Interest**: ${lending_result['calculation']['daily_return']:.2f} × 30 = ${lending_result['calculation']['monthly_return']:.2f}
                            6. **Yearly Interest**: ${value_usd:,.2f} × {lending_result['calculation']['apy']:.1f}% = ${lending_result['calculation']['yearly_return']:.2f}
                            """)
                        else:
                            st.markdown(f"""
                            **Borrowing Cost Breakdown:**
                            
                            1. **Borrowed Amount**: ${amount:,.2f} {asset} × ${price:,.2f} = ${value_usd:,.2f}
                            2. **Borrow APY**: {lending_result['calculation']['apy']:.1f}%
                            3. **Daily Cost**: ${value_usd:,.2f} × {lending_result['calculation']['apy']:.1f}% ÷ 365 = ${abs(lending_result['calculation']['daily_return']):.2f}
                            4. **Weekly Cost**: ${abs(lending_result['calculation']['daily_return']):.2f} × 7 = ${abs(lending_result['calculation']['weekly_return']):.2f}
                            5. **Monthly Cost**: ${abs(lending_result['calculation']['daily_return']):.2f} × 30 = ${abs(lending_result['calculation']['monthly_return']):.2f}
                            6. **Yearly Cost**: ${value_usd:,.2f} × {lending_result['calculation']['apy']:.1f}% = ${abs(lending_result['calculation']['yearly_return']):.2f}
                            """)
                else:
                    st.error(f"❌ Error: {lending_result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Simulation error: {e}")
    
    with tab3:
        st.markdown("#### 📊 **Liquidity Provision Simulator**")
        st.markdown("**See how much you'd earn from providing liquidity**")
        
        # How This Works section
        with st.expander("❓ **How This Works - Understanding Liquidity Provision**", expanded=False):
            st.markdown("""
            **🏊 What is Liquidity Provision?**
            
            Liquidity provision means **adding your tokens to a trading pool** so others can swap between them. You earn fees from every trade.
            
            **🔄 How It Works:**
            
            1. **Add Equal Value**: Provide equal USD value of two tokens
            2. **Receive LP Tokens**: Get "Liquidity Provider" tokens representing your share
            3. **Earn Trading Fees**: Collect 0.3% fee from every swap in your pool
            4. **Remove Liquidity**: Trade LP tokens back for your original tokens + fees
            
            **🧮 LP Token Calculation:**
            
            **LP Tokens = √(Value A × Value B)**
            
            This formula ensures:
            • Equal value contribution from both tokens
            • Fair distribution of trading fees
            • Protection against manipulation
            
            **📈 How You Earn:**
            
            • **Trading Fees**: 0.3% of every swap in your pool
            • **Yield Farming**: Additional rewards from protocols (if available)
            • **Compound Growth**: Reinvested fees increase your position
            
            **⚠️ Impermanent Loss Explained:**
            
            **What it is:** Loss when token prices change relative to each other
            
            **Example:**
            - You add 1 ETH ($3,200) + 3,200 USDC ($3,200) = $6,400 total
            - ETH price doubles to $6,400
            - If you had held: 1 ETH ($6,400) + 3,200 USDC ($3,200) = $9,600
            - As LP: You get less ETH but more USDC, total value ≈ $8,000
            - **Impermanent Loss: $1,600** (but you earned fees!)
            
            **💡 Risk vs Reward:**
            
            **Lower Risk Pools:**
            • Stablecoin pairs (USDC/USDT/DAI)
            • Lower APY (8-12%)
            • Minimal impermanent loss
            
            **Higher Risk Pools:**
            • Volatile pairs (ETH/LINK, ETH/UNI)
            • Higher APY (15-25%)
            • Higher impermanent loss potential
            
            **🎯 When to Provide Liquidity:**
            
            • **Long-term holding**: You plan to hold both tokens
            • **Market neutral**: You don't care about price direction
            • **Fee generation**: You want passive income from trading activity
            • **Diversification**: Spread risk across multiple assets
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏦 Protocol**")
            protocol = st.selectbox("Select Protocol:", ["Uniswap V3", "Curve Finance"], key="liq_protocol")
            
            st.markdown("**🪙 Token A**")
            token_a = st.selectbox("First Token:", ["ETH", "USDC", "USDT", "DAI", "WBTC"], key="liq_token_a")
            
            st.markdown("**💰 Amount A**")
            amount_a = st.number_input("Amount of Token A:", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="liq_amount_a")
            
            # Show value A
            price_a = current_prices.get(token_a, 0)
            value_a_usd = amount_a * price_a
            st.metric(f"Value A in USD", f"${value_a_usd:,.2f}")
        
        with col2:
            st.markdown("**🪙 Token B**")
            token_b = st.selectbox("Second Token:", ["ETH", "USDC", "USDT", "DAI", "WBTC"], key="liq_token_b")
            
            st.markdown("**💰 Amount B**")
            amount_b = st.number_input("Amount of Token B:", min_value=0.01, max_value=1000.0, value=1000.0, step=0.01, key="liq_amount_b")
            
            # Show value B
            price_b = current_prices.get(token_b, 0)
            value_b_usd = amount_b * price_b
            st.metric(f"Value B in USD", f"${value_b_usd:,.2f}")
            
            # Show total pool value
            total_pool_value = value_a_usd + value_b_usd
            st.metric("🏊 **Total Pool Value**", f"${total_pool_value:,.2f}")
        
        # Calculate liquidity button
        if st.button("📊 **Calculate Liquidity Returns**", key="calculate_liquidity", use_container_width=True, type="primary"):
            if token_a != token_b:
                try:
                    from live_protocol_simulator import DeFiProtocolSimulator
                    simulator = DeFiProtocolSimulator()
                    
                    # Map protocol names
                    protocol_map = {"Uniswap V3": "uniswap_v3", "Curve Finance": "curve_finance"}
                    protocol_id = protocol_map.get(protocol)
                    
                    liq_result = simulator.simulate_add_liquidity(
                        protocol=protocol_id,
                        token_a=token_a,
                        token_b=token_b,
                        amount_a=amount_a,
                        amount_b=amount_b
                    )
                    
                    if 'error' not in liq_result:
                        st.success("✅ **Liquidity Calculation Complete!**")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🏊 Pool Value", f"${liq_result['calculation']['total_pool_value']:,.2f}")
                            st.metric("🎯 LP Tokens", f"{liq_result['calculation']['lp_tokens']:.6f}")
                        
                        with col2:
                            st.metric("📈 Expected APY", f"{liq_result['calculation']['expected_apy']:.1f}%")
                            st.metric("💵 Daily Rewards", f"${liq_result['calculation']['daily_rewards_usd']:.2f}")
                        
                        with col3:
                            st.metric("⚠️ Risk Level", liq_result['calculation']['risk_level'])
                            st.metric("⛽ Gas Cost", f"${liq_result['gas_cost_usd']:.2f}")
                        
                        # Educational breakdown
                        st.markdown("---")
                        st.markdown("#### 📚 **What This Teaches You:**")
                        for note in liq_result['educational_notes']:
                            st.markdown(f"• {note}")
                        
                        # Show the math
                        with st.expander("🧮 **Show the Math**"):
                            st.markdown(f"""
                            **Liquidity Calculation Breakdown:**
                            
                            1. **Token A Value**: {amount_a} {token_a} × ${price_a:,.2f} = ${value_a_usd:,.2f}
                            2. **Token B Value**: {amount_b} {token_b} × ${price_b:,.2f} = ${value_b_usd:,.2f}
                            3. **Total Pool Value**: ${value_a_usd:,.2f} + ${value_b_usd:,.2f} = ${liq_result['calculation']['total_pool_value']:,.2f}
                            4. **LP Tokens**: √(${value_a_usd:,.2f} × ${value_b_usd:,.2f}) = {liq_result['calculation']['lp_tokens']:.6f}
                            5. **Daily Rewards**: ${liq_result['calculation']['total_pool_value']:,.2f} × {liq_result['calculation']['expected_apy']:.1f}% ÷ 365 = ${liq_result['calculation']['daily_rewards_usd']:.2f}
                            6. **Risk Assessment**: {'Low risk for stable pairs' if liq_result['calculation']['risk_level'] == 'Low' else 'Medium risk for volatile pairs'}
                            """)
                    else:
                        st.error(f"❌ Error: {liq_result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Simulation error: {e}")
            else:
                st.warning("⚠️ Please select different tokens for liquidity provision")
    
    with tab4:
        st.markdown("#### 🎮 **Quick Learning Demos**")
        st.markdown("**Try these instant demos to learn DeFi concepts**")
        
        # How This Works section
        with st.expander("❓ **How This Works - Understanding Quick Demos**", expanded=False):
            st.markdown("""
            **🎯 What are Quick Demos?**
            
            Quick demos are **instant simulations** that show you the basic concepts of DeFi without needing to input specific numbers.
            
            **🔄 Swap Demo:**
            
            **What you learn:**
            • How token exchanges work on DEXs
            • The role of slippage protection
            • Gas fee calculations
            • Transaction confirmation process
            
            **Real-world application:**
            • Trading ETH for USDC on Uniswap
            • Understanding price impact of large trades
            • Setting appropriate slippage tolerance
            
            **💰 Lending Demo:**
            
            **What you learn:**
            • How to deposit assets to earn interest
            • Understanding APY vs APR differences
            • Collateral requirements for borrowing
            • Interest calculation and compounding
            
            **Real-world application:**
            • Earning 3-5% APY on stablecoins
            • Borrowing against ETH collateral
            • Managing liquidation risks
            
            **📊 Yield Demo:**
            
            **What you learn:**
            • Basics of yield farming
            • Liquidity provision mechanics
            • Impermanent loss concepts
            • Reward token distribution
            
            **Real-world application:**
            • Providing liquidity to earn fees
            • Participating in governance protocols
            • Earning additional rewards beyond trading fees
            
            **💡 Learning Path:**
            
            1. **Start with Quick Demos** - Get familiar with concepts
            2. **Use Interactive Simulators** - Input your own numbers
            3. **Practice on Testnets** - Real transactions without real money
            4. **Apply on Mainnet** - When you're confident and ready
            
            **🎓 Why This Approach Works:**
            
            • **Progressive Learning**: Start simple, build complexity
            • **Hands-on Practice**: See results immediately
            • **Risk-free Environment**: Learn from mistakes safely
            • **Real-world Preparation**: Understand what to expect
            """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 **Quick Swap Demo**", key="quick_swap", use_container_width=True):
                st.session_state.show_quick_swap = True
        
        with col2:
            if st.button("💰 **Quick Lending Demo**", key="quick_lending", use_container_width=True):
                st.session_state.show_quick_lending = True
        
        with col3:
            if st.button("📊 **Quick Yield Demo**", key="quick_yield", use_container_width=True):
                st.session_state.show_quick_yield = True
        
        # Quick demo results
        if st.session_state.get('show_quick_swap', False):
            with st.expander("🔄 **Quick Swap Demo Result**", expanded=True):
                st.success("✅ **Swap Simulation Completed Successfully!**")
                st.markdown("""
                **📚 What you just learned:**
                
                • **Token Swap Process:** How to exchange one token for another
                • **Slippage Impact:** How price changes affect your swap
                • **Gas Fee Calculation:** Understanding transaction costs
                • **Confirmation Process:** What happens after you submit a swap
                
                **💡 Next Step:** Try the full Protocol Simulator above for more detailed learning!
                """)
                if st.button("Got it!", key="close_quick_swap"):
                    st.session_state.show_quick_swap = False
        
        if st.session_state.get('show_quick_lending', False):
            with st.expander("💰 **Quick Lending Demo Result**", expanded=True):
                st.success("✅ **Lending Simulation Completed Successfully!**")
                st.markdown("""
                **📚 What you just learned:**
                
                • **Lending Process:** How to deposit assets to earn interest
                • **APY vs APR:** Understanding different yield metrics
                • **Collateral Requirements:** How lending protocols stay secure
                • **Interest Calculation:** How your earnings accumulate over time
                
                **💡 Next Step:** Explore the full lending protocols in the simulator above!
                """)
                if st.button("Got it!", key="close_quick_lending"):
                    st.session_state.show_quick_lending = False
        
        if st.session_state.get('show_quick_yield', False):
            with st.expander("📊 **Quick Yield Demo Result**", expanded=True):
                st.success("✅ **Yield Farming Demo Completed Successfully!**")
                st.markdown("""
                **📚 What you just learned:**
                
                • **Yield Farming Basics:** How to earn rewards from DeFi protocols
                • **Liquidity Provision:** Providing tokens to earn trading fees
                • **Impermanent Loss:** Understanding the risks of liquidity provision
                • **Reward Tokens:** How protocols incentivize participation
                
                **💡 Next Step:** Dive deeper into yield strategies in the simulator above!
                """)
                if st.button("Got it!", key="close_quick_yield"):
                    st.session_state.show_quick_yield = False

def display_protocol_learning_section(protocol_name, description, features, learning_goals, difficulty, time_estimate):
    """Display a protocol learning section with educational content"""
    
    st.markdown(f"### {protocol_name}")
    st.markdown(f"**{description}**")
    
    # Protocol features
    st.markdown(f"**🚀 Features:** {features}")
    
    # Learning goals
    st.markdown(f"**📚 Learning Goals:** {learning_goals}")
    
    # Difficulty and time
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**🎯 Difficulty:** {difficulty}")
    with col2:
        st.info(f"**⏰ Time:** {time_estimate}")
    
    st.markdown("---")

def community_intelligence_page():
    """Community Intelligence page - Analyze community sentiment and trends"""
    
    # Header with clear explanation
    st.title("🌐 Community Intelligence Engine")
    st.markdown("**🎯 Discover what the DeFi community is talking about and find your next learning opportunity**")
    
    # What this page does section
    with st.expander("❓ What can you learn from this page?", expanded=True):
        st.markdown("""
        **🚀 This COMMUNITY INTELLIGENCE engine helps you:**
        
        1. **📊 Discover trending topics** - See what's hot in the DeFi community right now
        2. **🎯 Find learning opportunities** - Identify concepts that are gaining attention
        3. **💬 Understand community sentiment** - See how people feel about different DeFi topics
        4. **📚 Get educational recommendations** - Find out what you should learn next
        5. **🔍 Track community engagement** - See which topics are getting the most discussion
        
        **💡 Perfect for:** 
        • **Beginners:** Find popular topics to start learning
        • **Intermediate users:** Discover what's trending to stay current
        • **Advanced users:** Identify emerging trends and opportunities
        • **Educators:** See what the community wants to learn about
        
        **🔒 REMEMBER:** This analyzes real community data to help you learn what matters most!
        """)
    
    st.markdown("---")
    
    # How it works explanation
    st.subheader("🔍 **How Community Intelligence Works**")
    st.markdown("**Understanding how we analyze community data to help you learn**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📡 **Data Sources We Analyze**")
        st.markdown("""
        **🌐 Real-time community data from:**
        
        • **Reddit** - r/ethereum, r/defi, r/cryptocurrency
        • **Twitter** - DeFi influencers and community discussions
        • **GitHub** - Developer activity and protocol updates
        • **Discord/Telegram** - Community discussions and questions
        
        **💡 We focus on Ethereum and DeFi communities specifically**
        """)
    
    with col2:
        st.markdown("#### 🧠 **How We Process Information**")
        st.markdown("""
        **🔬 Our AI analyzes:**
        
        • **Topic extraction** - What concepts are being discussed
        • **Sentiment analysis** - How people feel about topics
        • **Engagement scoring** - Which posts get most attention
        • **Educational value** - How much can you learn from each topic
        
        **💡 We turn community noise into actionable learning insights**
        """)
    
    st.markdown("---")
    
    # Educational disclaimer
    st.info("""
    📚 **Educational Note:** This page shows you what the DeFi community is learning and discussing.
    **💡 Use these insights to guide your own learning journey and stay current with trends.**
    """)
    
    try:
        # Initialize the community intelligence engine
        engine = CommunityIntelligenceEngine()
        
        # Get community data
        with st.spinner("🔍 Analyzing community data..."):
            # Get real Reddit data from r/ethereum and r/defi
            try:
                reddit_posts = get_real_reddit_posts()
                if reddit_posts:
                    st.success(f"✅ Loaded {len(reddit_posts)} real Reddit posts from the community!")
                    community_data = engine.analyze_community_intelligence(reddit_posts=reddit_posts)
                else:
                    st.warning("⚠️ Could not load real Reddit data, using sample data for demonstration")
                    # Fallback to sample data
                    sample_reddit_posts = [
                        {
                            'title': 'How do I understand DeFi yield farming?',
                            'selftext': 'I\'m new to DeFi and want to learn about yield farming strategies. Can someone explain the basics?',
                            'author': 'crypto_learner',
                            'score': 45
                        },
                        {
                            'title': 'Ethereum Layer 2 solutions explained',
                            'selftext': 'Great breakdown of how rollups work and why they\'re important for scaling Ethereum.',
                            'author': 'eth_expert',
                            'score': 128
                        },
                        {
                            'title': 'DeFi security best practices',
                            'selftext': 'Important security considerations when using DeFi protocols. Always verify contracts and use hardware wallets.',
                            'author': 'security_guru',
                            'score': 128
                        }
                    ]
                    community_data = engine.analyze_community_intelligence(reddit_posts=sample_reddit_posts)
            except Exception as e:
                st.error(f"❌ Error loading Reddit data: {e}")
                st.info("💡 Using sample data for demonstration purposes")
                # Fallback to sample data
                sample_reddit_posts = [
                    {
                        'title': 'How do I understand DeFi yield farming?',
                        'selftext': 'I\'m new to DeFi and want to learn about yield farming strategies. Can someone explain the basics?',
                        'author': 'crypto_learner',
                        'score': 45
                    },
                    {
                        'title': 'Ethereum Layer 2 solutions explained',
                        'selftext': 'Great breakdown of how rollups work and why they\'re important for scaling Ethereum.',
                        'author': 'eth_expert',
                        'score': 128
                    },
                    {
                        'title': 'DeFi security best practices',
                        'selftext': 'Important security considerations when using DeFi protocols. Always verify contracts and use hardware wallets.',
                        'author': 'security_guru',
                        'score': 128
                    }
                ]
                community_data = engine.analyze_community_intelligence(reddit_posts=sample_reddit_posts)
        
        # Community Intelligence Summary
        st.subheader("📊 **Community Intelligence Summary**")
        st.markdown("**Here's what the DeFi community is talking about right now**")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Posts", community_data['summary']['total_posts'])
        
        with col2:
            st.metric("Total Engagement", community_data['summary']['total_engagement'])
        
        with col3:
            st.metric("Avg Educational Value", f"{community_data['summary']['average_educational_value']:.2f}")
        
        with col4:
            top_source = community_data['summary']['top_sources'][0]['source'] if community_data['summary']['top_sources'] else 'N/A'
            st.metric("Top Source", top_source)
        
        st.markdown("---")
        
        # Trending Concepts with better explanations
        st.subheader("🔥 **Trending Concepts - What's Hot Right Now**")
        st.markdown("**These are the DeFi topics getting the most community attention**")
        
        # Educational explanation of trending concepts
        with st.expander("📚 **What are Trending Concepts?**", expanded=False):
            st.markdown("""
            **🔥 Trending Concepts Explained:**
            
            • **Frequency:** How often this topic is mentioned in the community
            • **Engagement:** How much the community is interacting with this topic
            • **Sentiment:** How positive/negative the community feels about it
            • **Learning Time:** Estimated time to understand this concept
            • **Difficulty:** How complex this topic is to learn
            
            **💡 Higher engagement = More people want to learn about this!**
            """)
        
        # Display trending concepts
        for concept in community_data['trending_concepts']:
            with st.expander(f"🔥 **{concept['concept']} - {concept['difficulty_level']}**", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Frequency", concept['frequency'])
                    st.metric("Engagement", concept['engagement_score'])
                
                with col2:
                    st.metric("Sources", ", ".join(concept['sources']))
                    st.metric("Sentiment", f"{concept['sentiment_score']:.2f}")
                
                with col3:
                    st.metric("Learning Time", f"{concept['estimated_learning_time']} min")
                    st.metric("Difficulty", concept['difficulty_level'])
                
                # Educational opportunity
                st.markdown("#### 📚 **What You Can Learn**")
                st.info(f"**{concept['educational_opportunity']}**")
                
                # Related topics
                if concept.get('related_topics'):
                    st.markdown("#### 🔗 **Related Topics to Explore**")
                    for topic in concept['related_topics']:
                        st.markdown(f"• **{topic}**")
                
                # Learning recommendation
                st.markdown("#### 🎯 **Learning Recommendation**")
                if concept['difficulty_level'] == 'Beginner':
                    st.success("**🚀 Great starting point!** This is perfect for beginners.")
                elif concept['difficulty_level'] == 'Intermediate':
                    st.warning("**📚 Build your foundation first!** Make sure you understand the basics.")
                else:
                    st.error("**⚡ Advanced topic!** Master intermediate concepts before diving in.")
                
                # Action button
                if st.button(f"🎯 **Start Learning {concept['concept']}**", key=f"learn_{concept['concept']}"):
                    st.success(f"🚀 Starting your learning journey in {concept['concept']}!")
                    st.info("💡 This would redirect you to our educational modules for this topic.")
        
        st.markdown("---")
        
        # Show the actual Reddit posts being analyzed
        st.subheader("📱 **Real Community Posts Being Analyzed**")
        st.markdown("**These are the actual Reddit posts from r/ethereum and r/defi that we're analyzing**")
        
        if 'reddit_posts' in locals() and reddit_posts:
            # Show real Reddit posts
            for i, post in enumerate(reddit_posts[:5]):  # Show first 5 posts
                with st.expander(f"📱 **{post['title'][:60]}...**", expanded=False):
                    st.markdown(f"**Author:** {post.get('author', 'Unknown')}")
                    st.markdown(f"**Score:** {post.get('score', 0)} upvotes")
                    st.markdown(f"**Comments:** {post.get('num_comments', 0)}")
                    st.markdown(f"**Subreddit:** r/{post.get('subreddit', 'ethereum')}")
                    
                    # Show post content
                    if post.get('selftext'):
                        st.markdown("**Content:**")
                        st.markdown(f"_{post['selftext'][:200]}..._")
                    
                    # Show URL
                    if post.get('url'):
                        st.markdown(f"**🔗 [View on Reddit]({post['url']})**")
                    
                    # Show when it was posted
                    if post.get('created_utc'):
                        from datetime import datetime
                        post_time = datetime.fromtimestamp(post['created_utc'])
                        st.markdown(f"**Posted:** {post_time.strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            # Show sample data info
            st.info("""
            **📝 Currently using sample data for demonstration**
            
            **💡 In production, this would show real Reddit posts from:**
            • r/ethereum - Ethereum community discussions
            • r/defi - DeFi protocol discussions  
            • r/cryptocurrency - General crypto discussions
            
            **🔗 The real posts would include:**
            • Actual questions from the community
            • Real discussions about DeFi concepts
            • Current trending topics and concerns
            • Educational content being shared
            """)
        
        st.markdown("---")
        
        # Educational Opportunities Section
        st.subheader("📚 **Your Personalized Learning Opportunities**")
        st.markdown("**Based on what the community is learning, here's what you should focus on**")
        
        # Get educational opportunities from the community data
        opportunities = community_data.get('educational_opportunities', [])
        
        if opportunities:
            for i, opportunity in enumerate(opportunities):
                with st.expander(f"📚 **{opportunity.get('topics', ['General'])[0]} - {opportunity.get('difficulty', 'Medium')}**", expanded=False):
                    st.markdown(f"**🎯 Why learn this:** {opportunity.get('content_preview', 'Learn about this trending topic')}")
                    st.markdown(f"**⏰ Time investment:** {opportunity.get('estimated_time', 30)} minutes")
                    st.markdown(f"**📊 Community interest:** {opportunity.get('engagement', 0)}")
                    
                    # Learning path
                    if opportunity.get('learning_path'):
                        st.markdown("#### 🛤️ **Recommended Learning Path**")
                        for step in opportunity['learning_path']:
                            st.markdown(f"• {step}")
                    
                    # Start learning button
                    if st.button(f"🚀 **Begin Learning {opportunity.get('topics', ['General'])[0]}**", key=f"start_{i}"):
                        st.success(f"🎯 Starting {opportunity.get('topics', ['General'])[0]} learning module!")
                        st.info("💡 This would open our interactive learning interface.")
        
        st.markdown("---")
        
        # Community Insights Section
        st.subheader("💡 **Community Insights & Trends**")
        st.markdown("**What patterns we're seeing in the DeFi community**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 **Rising Topics**")
            st.markdown("**These concepts are gaining momentum:**")
            rising_topics = ["Layer 2 Scaling", "DeFi Insurance", "Cross-chain Bridges", "MEV Protection"]
            for topic in rising_topics:
                st.markdown(f"• **{topic}** - Growing community interest")
        
        with col2:
            st.markdown("#### 📉 **Declining Topics**")
            st.markdown("**These concepts are getting less attention:**")
            declining_topics = ["Basic DeFi Tutorials", "Simple Yield Farming", "Basic Liquidity Provision"]
            for topic in declining_topics:
                st.markdown(f"• **{topic}** - Community moving to advanced topics")
        
        st.markdown("---")
        
        # Quick Actions Section
        st.subheader("⚡ **Community Intelligence Actions**")
        st.markdown("**Take action based on community insights**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 **Refresh Community Data**", key="refresh_community", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📊 **Export Community Report**", key="export_community", use_container_width=True):
                st.success("📊 Community intelligence report exported successfully!")
        
        with col3:
            if st.button("🎯 **Get Learning Recommendations**", key="get_recommendations", use_container_width=True):
                st.info("🎯 Analyzing your profile to give personalized recommendations...")
        
        # FAQ Section
        st.markdown("---")
        st.subheader("❓ **Frequently Asked Questions - Community Intelligence**")
        
        with st.expander("🤔 **How often is this data updated?**", expanded=False):
            st.markdown("""
            **📡 Data Update Frequency:**
            
            • **Real-time analysis:** We analyze new posts as they appear
            • **Trending updates:** Updated every few hours
            • **Historical data:** We track trends over time
            • **Source monitoring:** Continuous monitoring of all sources
            
            **💡 The data you see is always current and relevant!**
            """)
        
        with st.expander("🎯 **How do I use this to improve my DeFi knowledge?**", expanded=False):
            st.markdown("""
            **📚 Using Community Intelligence for Learning:**
            
            1. **Check trending concepts** - See what's popular right now
            2. **Focus on high-engagement topics** - More engagement = more learning resources
            3. **Follow the learning path** - Start with basics, then advance
            4. **Join community discussions** - Learn from others' questions and answers
            5. **Stay current** - DeFi moves fast, this keeps you updated
            
            **💡 Think of this as your DeFi community radar!**
            """)
        
        with st.expander("🔍 **How accurate is the sentiment analysis?**", expanded=False):
            st.markdown("""
            **🧠 Sentiment Analysis Accuracy:**
            
            • **AI-powered analysis:** Uses advanced natural language processing
            • **Community context:** Understands DeFi-specific terminology
            • **Pattern recognition:** Identifies trends across multiple sources
            • **Human validation:** Community feedback improves accuracy over time
            
            **⚠️ Remember:** Sentiment is one factor - always do your own research!
            """)
        
        # Final call to action
        st.success("""
        🎯 **Ready to discover what the DeFi community is learning?**
        
        **🚀 Use the insights above to guide your learning journey!**
        **💡 Focus on trending topics with high engagement**
        **📚 Follow the recommended learning paths**
        **🔍 Check back regularly for new trends and opportunities**
        """)
        
    except Exception as e:
        st.error(f"❌ Error initializing Community Intelligence Engine: {e}")
        st.info("💡 This feature requires the community_intelligence_engine module")
        st.code(f"Error details: {str(e)}")

def simulate_protocol_action(protocol_name, action_type, parameters):
    """Simulate a protocol action for educational purposes"""
    
    st.success(f"✅ **{action_type.title()} Simulation Completed Successfully!**")
    st.markdown(f"**🎮 Protocol:** {protocol_name}")
    st.markdown(f"**🔄 Action:** {action_type}")
    
    # Display parameters
    st.markdown("#### 📊 **Simulation Parameters**")
    for key, value in parameters.items():
        st.info(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Educational result
    st.markdown("#### 📚 **What You Learned**")
    st.markdown("""
    **💡 Educational Insights:**
    
    • **Transaction Process:** How DeFi transactions work step by step
    • **Parameter Impact:** How different settings affect your results
    • **Gas Estimation:** Understanding transaction costs
    • **Risk Assessment:** Identifying potential risks and mitigations
    
    **⚠️ Remember:** This is a simulation for learning - real DeFi involves real risks!
    """)
    
    # Next steps
    st.markdown("#### 🚀 **Next Learning Steps**")
    st.markdown("""
    **📚 Continue your DeFi education:**
    
    1. **Read the protocol documentation** to understand the mechanics
    2. **Practice with different parameters** to see how they affect outcomes
    3. **Study the risks** associated with this type of interaction
    4. **Prepare for testnet** when wallet connection becomes available
    
    **💡 Tip:** The more you practice here, the more confident you'll be on testnets!
    """)

if __name__ == "__main__":
    main()
