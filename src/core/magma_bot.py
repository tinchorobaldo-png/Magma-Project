#!/usr/bin/env python3
"""
MAGMA BOT - Educational ETH Intelligence System
Transform market intelligence into learning opportunities with GPT-powered educational insights
Focus: Educational analysis, market context, and progressive learning for Ethereum mastery
"""

import os
import asyncio
import logging
import sys
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from dataclasses import dataclass

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all advanced modules
from lunarcrush_analyzer import LunarCrushAnalyzer
from gpt_whale_analyzer import GPTWhalePatternAnalyzer
from alpha_signals_generator import AlphaSignalsGenerator, AlphaSignal
from content_generator import SuperiorContentGenerator
from crypto_events_calendar import CryptoEventsCalendar
# from reddit_analyzer import AdvancedRedditAnalyzer  # Using internal RedditAnalyzer instead
from news_analyzer import NewsAnalyzer
from magma_gas_optimizer_fixed import MagmaGasOptimizerFixed, GPTRecommendation, TRANSACTION_TYPES
from educational_gas_wizard import EducationalGasWizard, EducationalAlert
from educational_context_connector import EducationalContextConnector, LearningLevel

# Import existing modules
from pycoingecko import CoinGeckoAPI

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/magma_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# REDDIT POST STRUCTURE
# ==============================================================================

@dataclass
class RedditPost:
    """Estructura para posts de Reddit"""
    title: str
    content: str
    author: str
    url: str
    subreddit: str
    score: int
    comments: int
    upvote_ratio: float
    created_utc: int
    sentiment_score: float = 0.0
    gpt_analysis: Dict[str, Any] = None
    engagement_score: float = 0.0
    virality_potential: float = 0.0

# ==============================================================================
# REDDIT ANALYZER CLASS
# ==============================================================================

class RedditAnalyzer:
    """Analizador avanzado de Reddit con GPT integration"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Subreddits ETH-CENTRIC para análisis
        self.eth_subreddits = [
            'ethereum', 'ethtrader', 'defi', 'ethfinance', 'ethdev',
            'ethereumclassic', 'ethstaker', 'eth2', 'layer2'
        ]
        
        # Keywords ETH-CENTRIC para análisis de tendencias
        self.eth_trend_keywords = {
            'ethereum_core': ['ethereum', 'eth', 'ether', 'vitalik', 'foundation'],
            'defi_ecosystem': ['defi', 'yield', 'liquidity', 'pool', 'swap', 'farm', 'uniswap', 'aave'],
            'scaling_solutions': ['layer2', 'rollup', 'sharding', 'sidechain', 'polygon', 'arbitrum', 'optimism'],
            'smart_contracts': ['smart contract', 'solidity', 'vyper', 'dapp', 'web3'],
            'gas_optimization': ['gas', 'gas fees', 'eip1559', 'base fee', 'priority fee'],
            'staking': ['staking', 'validator', 'beacon chain', 'consensus', 'pos'],
            'institutional_eth': ['etf', 'institutional', 'adoption', 'enterprise', 'corporate']
        }
    
    async def get_comprehensive_reddit_analysis(self, cryptos_list: List[str] = None) -> Dict[str, Any]:
        """Análisis ETH-CENTRIC completo de Reddit con GPT"""
        # Solo ETH y temas relacionados
        eth_topics = ['ethereum', 'defi', 'layer2', 'smart contracts', 'gas fees', 'staking']
        
        logger.info("Starting ETH-CENTRIC Reddit analysis with GPT...")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": {
                "score": 0.0,
                "classification": "NEUTRAL",
                "confidence": 0.0
            },
            "eth_specific": {},
            "trending_topics": [],
            "viral_posts": [],
            "market_sentiment": {
                "bullish_signals": [],
                "bearish_signals": [],
                "neutral_signals": []
            },
            "gpt_insights": {},
            "engagement_metrics": {
                "total_posts": 0,
                "total_comments": 0,
                "avg_engagement": 0.0
            }
        }
        
        try:
            # 1. Obtener posts de subreddits ETH-CENTRIC
            eth_subreddit_posts = self._scrape_eth_subreddits()
            
            # 2. Obtener posts específicos por tema ETH
            eth_topic_posts = {}
            for topic in eth_topics:
                eth_topic_posts[topic] = self._scrape_eth_topic_posts(topic)
            
            # 3. Análisis GPT de posts virales ETH
            viral_posts = self._identify_viral_posts(eth_subreddit_posts)
            gpt_viral_analysis = await self._analyze_viral_posts_with_gpt(viral_posts)
            
            # 4. Análisis de tendencias ETH
            trending_topics = self._analyze_eth_trending_topics(eth_subreddit_posts)
            
            # 5. Análisis GPT por tema ETH
            eth_topic_analysis = {}
            for topic, posts in eth_topic_posts.items():
                if posts:
                    eth_topic_analysis[topic] = await self._analyze_eth_topic_posts_with_gpt(topic, posts)
            
            # 6. Calcular sentiment general ETH
            overall_sentiment = self._calculate_eth_overall_sentiment(eth_subreddit_posts, eth_topic_posts)
            
            # 7. Construir análisis final ETH-CENTRIC
            analysis.update({
                "overall_sentiment": overall_sentiment,
                "eth_specific": eth_topic_analysis,
                "trending_topics": trending_topics,
                "viral_posts": gpt_viral_analysis,
                "engagement_metrics": self._calculate_engagement_metrics(eth_subreddit_posts, eth_topic_posts)
            })
            
            logger.info(f"ETH-CENTRIC Reddit analysis completed: {len(eth_subreddit_posts)} posts analyzed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive Reddit analysis: {e}")
            return self._generate_fallback_reddit_data()
    
    def _scrape_eth_subreddits(self) -> List[RedditPost]:
        """Scraping ETH-CENTRIC de subreddits de Ethereum"""
        posts = []
        
        try:
            # Obtener posts de subreddits ETH-CENTRIC
            for subreddit in self.eth_subreddits[:3]:  # Top 3 subreddits ETH
                try:
                    # Obtener posts de diferentes categorías
                    categories = ['hot', 'top']
                    
                    for category in categories:
                        url = f"https://www.reddit.com/r/{subreddit}/{category}/.json"
                        params = {
                            'limit': 20,
                            't': 'day',
                            'raw_json': 1
                        }
                        
                        response = requests.get(url, headers=self.headers, params=params, timeout=15)
                        if response.status_code == 200:
                            data = response.json()
                            
                            for post in data['data']['children']:
                                post_data = post['data']
                                
                                # Filtrar posts ETH-relevantes
                                if self._is_eth_relevant_post(post_data):
                                    reddit_post = RedditPost(
                                        title=post_data.get('title', ''),
                                        content=post_data.get('selftext', ''),
                                        author=post_data.get('author', '[deleted]'),
                                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                                        subreddit=subreddit,
                                        score=post_data.get('score', 0),
                                        comments=post_data.get('num_comments', 0),
                                        upvote_ratio=post_data.get('upvote_ratio', 0.5),
                                        created_utc=post_data.get('created_utc', 0)
                                    )
                                    
                                    # Calcular engagement score
                                    reddit_post.engagement_score = self._calculate_engagement_score(reddit_post)
                                    reddit_post.virality_potential = self._calculate_virality_potential(reddit_post)
                                    
                                    posts.append(reddit_post)
                        
                        time.sleep(1)  # Rate limiting
                        
                except Exception as e:
                    logger.warning(f"Error scraping r/{subreddit}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error scraping ETH subreddits: {e}")
        
        return posts
    
    def _scrape_eth_topic_posts(self, topic: str) -> List[RedditPost]:
        """Scraping de posts específicos por tema ETH"""
        posts = []
        
        try:
            # Buscar en subreddits ETH-CENTRIC por tema específico
            for subreddit in self.eth_subreddits[:2]:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search/.json"
                    params = {
                        'q': topic,
                        'restrict_sr': 'on',
                        'limit': 10,
                        't': 'day',
                        'raw_json': 1
                    }
                    
                    response = requests.get(url, headers=self.headers, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data['data']['children']:
                            post_data = post['data']
                            
                            if self._is_eth_relevant_post(post_data):
                                reddit_post = RedditPost(
                                    title=post_data.get('title', ''),
                                    content=post_data.get('selftext', ''),
                                    author=post_data.get('author', '[deleted]'),
                                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                                    subreddit=subreddit,
                                    score=post_data.get('score', 0),
                                    comments=post_data.get('num_comments', 0),
                                    upvote_ratio=post_data.get('upvote_ratio', 0.5),
                                    created_utc=post_data.get('created_utc', 0)
                                )
                                
                                reddit_post.engagement_score = self._calculate_engagement_score(reddit_post)
                                reddit_post.virality_potential = self._calculate_virality_potential(reddit_post)
                                
                                posts.append(reddit_post)
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error searching {topic} in r/{subreddit}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error scraping ETH topic posts for {topic}: {e}")
        
        return posts
    
    def _scrape_crypto_specific_posts(self, crypto: str) -> List[RedditPost]:
        """Scraping de posts específicos por crypto"""
        posts = []
        
        # Mapeo de cryptos a subreddits
        crypto_subreddits = {
            'bitcoin': ['Bitcoin', 'btc'],
            'ethereum': ['ethereum', 'ethtrader'],
            'solana': ['solana'],
            'cardano': ['cardano'],
            'defi': ['defi', 'CryptoCurrency'],
            'binance': ['binance']
        }
        
        subreddits = crypto_subreddits.get(crypto.lower(), [])
        
        for subreddit in subreddits[:2]:  # Máximo 2 subreddits por crypto
            try:
                url = f"https://www.reddit.com/r/{subreddit}/top/.json"
                params = {
                    'limit': 15,
                    't': 'day',
                    'raw_json': 1
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data['data']['children']:
                        post_data = post['data']
                        
                        if self._is_relevant_post(post_data):
                            reddit_post = RedditPost(
                                title=post_data.get('title', ''),
                                content=post_data.get('selftext', ''),
                                author=post_data.get('author', '[deleted]'),
                                url=f"https://reddit.com{post_data.get('permalink', '')}",
                                subreddit=subreddit,
                                score=post_data.get('score', 0),
                                comments=post_data.get('num_comments', 0),
                                upvote_ratio=post_data.get('upvote_ratio', 0.5),
                                created_utc=post_data.get('created_utc', 0)
                            )
                            
                            reddit_post.engagement_score = self._calculate_engagement_score(reddit_post)
                            reddit_post.virality_potential = self._calculate_virality_potential(reddit_post)
                            
                            posts.append(reddit_post)
                
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error scraping r/{subreddit}: {e}")
                continue
        
        return posts
    
    def _is_eth_relevant_post(self, post_data: Dict) -> bool:
        """Determina si un post es ETH-relevante para análisis"""
        title = post_data.get('title', '').lower()
        content = post_data.get('selftext', '').lower()
        
        # Filtrar posts irrelevantes
        irrelevant_keywords = ['meme', 'shitpost', 'moon', 'lambo', 'wen', 'bitcoin', 'btc', 'solana', 'cardano']
        if any(keyword in title for keyword in irrelevant_keywords):
            return False
        
        # Debe contener keywords ETH
        eth_keywords = ['ethereum', 'eth', 'defi', 'layer2', 'smart contract', 'gas', 'staking', 'rollup', 'sharding']
        if not any(keyword in title or keyword in content for keyword in eth_keywords):
            return False
        
        # Debe tener engagement mínimo
        score = post_data.get('score', 0)
        comments = post_data.get('num_comments', 0)
        
        return score > 5 or comments > 3
    
    def _is_relevant_post(self, post_data: Dict) -> bool:
        """Determina si un post es relevante para análisis (legacy method)"""
        return self._is_eth_relevant_post(post_data)
    
    def _calculate_engagement_score(self, post: RedditPost) -> float:
        """Calcula score de engagement"""
        # Fórmula: (score * 0.4) + (comments * 0.6) + (upvote_ratio * 100)
        engagement = (post.score * 0.4) + (post.comments * 0.6) + (post.upvote_ratio * 100)
        return round(engagement, 2)
    
    def _calculate_virality_potential(self, post: RedditPost) -> float:
        """Calcula potencial de viralidad"""
        # Fórmula basada en engagement y ratio de upvotes
        virality = (post.engagement_score * post.upvote_ratio) / 100
        return round(virality, 2)
    
    def _identify_viral_posts(self, posts: List[RedditPost]) -> List[RedditPost]:
        """Identifica posts virales"""
        # Ordenar por engagement score y virality potential
        viral_posts = sorted(posts, key=lambda x: (x.engagement_score, x.virality_potential), reverse=True)
        return viral_posts[:10]  # Top 10 posts virales
    
    async def _analyze_viral_posts_with_gpt(self, viral_posts: List[RedditPost]) -> Dict[str, Any]:
        """Análisis GPT de posts virales"""
        if not viral_posts or not self.openai_api_key:
            return {"insights": "No viral posts found or GPT not available", "trends": []}
        
        try:
            # Preparar datos para GPT
            posts_data = []
            for post in viral_posts[:5]:  # Top 5 para análisis
                posts_data.append({
                    "title": post.title,
                    "content": post.content[:500],  # Limitar contenido
                    "score": post.score,
                    "comments": post.comments,
                    "engagement": post.engagement_score
                })
            
            prompt = self._create_viral_analysis_prompt(posts_data)
            
            # Aquí iría la llamada a OpenAI API
            # Por ahora, retornamos análisis simulado
            return {
                "insights": "Viral posts analysis completed. High engagement detected in DeFi and scaling discussions.",
                "trends": ["DeFi", "Scaling", "Institutional Adoption"],
                "posts_analyzed": len(posts_data)
            }
            
        except Exception as e:
            logger.error(f"Error in GPT analysis of viral posts: {e}")
            return {"insights": "GPT analysis failed", "trends": []}
    
    async def _analyze_crypto_posts_with_gpt(self, crypto: str, posts: List[RedditPost]) -> Dict[str, Any]:
        """Análisis GPT específico por crypto"""
        if not posts:
            return {"sentiment": "NEUTRAL", "score": 0.0, "insights": "No posts found"}
        
        try:
            # Preparar datos para GPT
            posts_summary = []
            for post in posts[:3]:  # Top 3 posts
                posts_summary.append(f"Title: {post.title}\nEngagement: {post.engagement_score}")
            
            # Aquí iría la llamada a OpenAI API
            # Por ahora, retornamos análisis simulado
            return {
                "sentiment": "SLIGHTLY_BULLISH",
                "score": 0.3,
                "insights": f"Positive sentiment detected in {crypto} discussions. High engagement in technical topics.",
                "posts_analyzed": len(posts)
            }
            
        except Exception as e:
            logger.error(f"Error in GPT analysis for {crypto}: {e}")
            return {"sentiment": "NEUTRAL", "score": 0.0, "insights": "GPT analysis failed"}
    
    def _analyze_trending_topics(self, posts: List[RedditPost]) -> List[Dict[str, Any]]:
        """Análisis de temas trending"""
        topic_counts = {}
        
        for post in posts:
            title_content = f"{post.title} {post.content}".lower()
            
            for category, keywords in self.trend_keywords.items():
                for keyword in keywords:
                    if keyword in title_content:
                        if category not in topic_counts:
                            topic_counts[category] = 0
                        topic_counts[category] += 1
        
        # Ordenar por frecuencia
        trending_topics = []
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            trending_topics.append({
                "topic": topic,
                "frequency": count,
                "trending_score": count / len(posts) if posts else 0
            })
        
        return trending_topics[:5]  # Top 5 trending topics
    
    def _analyze_eth_trending_topics(self, posts: List[RedditPost]) -> List[Dict[str, Any]]:
        """Análisis de temas trending ETH-CENTRIC"""
        topic_counts = {}
        
        for post in posts:
            title_content = f"{post.title} {post.content}".lower()
            
            for category, keywords in self.eth_trend_keywords.items():
                for keyword in keywords:
                    if keyword in title_content:
                        if category not in topic_counts:
                            topic_counts[category] = 0
                        topic_counts[category] += 1
        
        # Ordenar por frecuencia
        trending_topics = []
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            trending_topics.append({
                "topic": topic,
                "frequency": count,
                "trending_score": count / len(posts) if posts else 0
            })
        
        return trending_topics[:5]  # Top 5 trending topics ETH
    
    def _calculate_overall_sentiment(self, crypto_currency_posts: List[RedditPost], crypto_posts: Dict[str, List[RedditPost]]) -> Dict[str, Any]:
        """Calcula sentiment general"""
        all_scores = []
        
        # Scores de posts de r/CryptoCurrency
        for post in crypto_currency_posts:
            # Calcular sentiment básico basado en engagement y keywords
            sentiment = self._calculate_basic_sentiment(f"{post.title} {post.content}")
            all_scores.append(sentiment)
        
        # Scores de posts específicos por crypto
        for posts in crypto_posts.values():
            for post in posts:
                sentiment = self._calculate_basic_sentiment(f"{post.title} {post.content}")
                all_scores.append(sentiment)
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            classification = self._classify_sentiment(avg_score)
            confidence = min(len(all_scores) / 50, 1.0)  # Confidence basado en cantidad de posts
        else:
            avg_score = 0.0
            classification = "NEUTRAL"
            confidence = 0.0
        
        return {
            "score": round(avg_score, 3),
            "classification": classification,
            "confidence": round(confidence, 2)
        }
    
    def _calculate_eth_overall_sentiment(self, eth_subreddit_posts: List[RedditPost], eth_topic_posts: Dict[str, List[RedditPost]]) -> Dict[str, Any]:
        """Calcula sentiment general ETH-CENTRIC"""
        all_scores = []
        
        # Scores de posts de subreddits ETH
        for post in eth_subreddit_posts:
            # Calcular sentiment básico basado en engagement y keywords ETH
            sentiment = self._calculate_eth_basic_sentiment(f"{post.title} {post.content}")
            all_scores.append(sentiment)
        
        # Scores de posts específicos por tema ETH
        for posts in eth_topic_posts.values():
            for post in posts:
                sentiment = self._calculate_eth_basic_sentiment(f"{post.title} {post.content}")
                all_scores.append(sentiment)
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            classification = self._classify_eth_sentiment(avg_score)
            confidence = min(len(all_scores) / 50, 1.0)  # Confidence basado en cantidad de posts
        else:
            avg_score = 0.0
            classification = "NEUTRAL"
            confidence = 0.0
        
        return {
            "score": round(avg_score, 3),
            "classification": classification,
            "confidence": round(confidence, 2)
        }
    
    def _calculate_basic_sentiment(self, text: str) -> float:
        """Calcula sentiment básico"""
        positive_words = ['bullish', 'moon', 'pump', 'buy', 'hodl', 'good', 'great', 'amazing', 'up', 'rise', 'gain', 'adoption']
        negative_words = ['bearish', 'dump', 'sell', 'crash', 'down', 'fall', 'drop', 'bad', 'terrible', 'loss', 'scam']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalizar score entre -1 y 1
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_eth_basic_sentiment(self, text: str) -> float:
        """Calcula sentiment básico ETH-CENTRIC"""
        # Keywords positivas ETH
        positive_words = ['bullish', 'moon', 'pump', 'buy', 'hodl', 'good', 'great', 'amazing', 'up', 'rise', 'gain', 'adoption', 'ethereum', 'defi', 'layer2', 'scaling', 'pos', 'staking']
        # Keywords negativas ETH
        negative_words = ['bearish', 'dump', 'sell', 'crash', 'down', 'fall', 'drop', 'bad', 'terrible', 'loss', 'scam', 'gas fees', 'congestion', 'slow', 'expensive']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalizar score entre -1 y 1
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment))
    
    def _classify_eth_sentiment(self, score: float) -> str:
        """Clasifica sentiment ETH basado en score"""
        if score >= 0.1:
            return "BULLISH"
        elif score <= -0.1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _classify_sentiment(self, score: float) -> str:
        """Clasifica sentiment basado en score"""
        if score >= 0.1:
            return "BULLISH"
        elif score <= -0.1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_engagement_metrics(self, crypto_currency_posts: List[RedditPost], crypto_posts: Dict[str, List[RedditPost]]) -> Dict[str, Any]:
        """Calcula métricas de engagement"""
        all_posts = crypto_currency_posts + [post for posts in crypto_posts.values() for post in posts]
        
        if not all_posts:
            return {"total_posts": 0, "total_comments": 0, "avg_engagement": 0.0}
        
        total_posts = len(all_posts)
        total_comments = sum(post.comments for post in all_posts)
        avg_engagement = sum(post.engagement_score for post in all_posts) / total_posts
        
        return {
            "total_posts": total_posts,
            "total_comments": total_comments,
            "avg_engagement": round(avg_engagement, 2)
        }
    
    def _create_viral_analysis_prompt(self, posts_data: List[Dict[str, Any]]) -> str:
        """Crea prompt para análisis GPT de posts virales"""
        prompt = "Analyze these viral Reddit posts for crypto market sentiment and trends:\n\n"
        
        for i, post in enumerate(posts_data, 1):
            prompt += f"Post {i}:\n"
            prompt += f"Title: {post['title']}\n"
            prompt += f"Content: {post['content'][:200]}...\n"
            prompt += f"Score: {post['score']}, Comments: {post['comments']}\n\n"
        
        prompt += "Provide insights on:\n1. Overall market sentiment\n2. Key trends identified\n3. Potential market impact\n4. Community mood"
        
        return prompt
    
    def _generate_fallback_reddit_data(self) -> Dict[str, Any]:
        """Genera datos de fallback para Reddit"""
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": {
                "score": 0.0,
                "classification": "NEUTRAL",
                "confidence": 0.0
            },
            "crypto_specific": {},
            "trending_topics": [
                {"topic": "defi", "frequency": 15, "trending_score": 0.3},
                {"topic": "scaling", "frequency": 12, "trending_score": 0.24},
                {"topic": "institutional", "frequency": 8, "trending_score": 0.16}
            ],
            "viral_posts": {
                "insights": "Reddit analysis temporarily unavailable. Using fallback data.",
                "trends": ["DeFi", "Scaling", "Institutional"],
                "posts_analyzed": 0
            },
            "engagement_metrics": {
                "total_posts": 0,
                "total_comments": 0,
                "avg_engagement": 0.0
            }
        }
    
    async def _analyze_eth_topic_posts_with_gpt(self, topic: str, posts: List[RedditPost]) -> Dict[str, Any]:
        """Análisis GPT de posts por tema ETH"""
        try:
            if not posts:
                return {"error": "No posts to analyze"}
            
            # Preparar prompt para GPT
            prompt = self._create_eth_topic_analysis_prompt(topic, posts)
            
            # Llamar a GPT
            response = await self._call_gpt(prompt)
            
            if response:
                return {
                    "topic": topic,
                    "gpt_analysis": response,
                    "posts_analyzed": len(posts),
                    "avg_engagement": sum(post.engagement_score for post in posts) / len(posts) if posts else 0
                }
            else:
                return {"error": "GPT analysis failed"}
                
        except Exception as e:
            logger.error(f"Error in ETH topic GPT analysis for {topic}: {e}")
            return {"error": str(e)}
    
    def _create_eth_topic_analysis_prompt(self, topic: str, posts: List[RedditPost]) -> str:
        """Crea prompt para análisis GPT de tema ETH"""
        sample_posts = posts[:3]  # Top 3 posts
        
        prompt = f"""Analyze the following Ethereum-related Reddit posts about {topic}:

Posts to analyze:
"""
        
        for i, post in enumerate(sample_posts, 1):
            prompt += f"""
Post {i}:
Title: {post.title}
Content: {post.content[:200]}...
Score: {post.score}, Comments: {post.comments}
"""
        
        prompt += f"""
Based on these posts about {topic}, provide:
1. Overall sentiment (BULLISH/BEARISH/NEUTRAL)
2. Key themes and discussions
3. Community mood and engagement
4. Potential impact on Ethereum ecosystem
5. Trading/investment insights

Format as JSON with these keys: sentiment, themes, mood, impact, insights
"""
        
        return prompt
    
    async def _call_gpt(self, prompt: str) -> str:
        """Llama a GPT para análisis"""
        try:
            if not self.openai_api_key:
                return "GPT API key not available"
            
            import openai
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert Ethereum analyst. Provide concise, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling GPT: {e}")
            return f"GPT analysis failed: {str(e)}"

class MagmaSupremeBot:
    """Educational MAGMA bot transforming market intelligence into learning opportunities"""
    
    def __init__(self):
        # API Keys
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.lunarcrush_api_key = os.getenv('LUNARCRUSH_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        # Initialize APIs
        self.coingecko = CoinGeckoAPI()
        
        # Initialize advanced modules
        self.lunarcrush_analyzer = LunarCrushAnalyzer(self.lunarcrush_api_key) if self.lunarcrush_api_key else None
        self.gpt_whale_analyzer = GPTWhalePatternAnalyzer(self.openai_api_key) if self.openai_api_key else None
        self.alpha_signals_generator = AlphaSignalsGenerator()
        self.content_generator = SuperiorContentGenerator(self.openai_api_key) if self.openai_api_key else None
        self.crypto_events_calendar = CryptoEventsCalendar()
        self.reddit_analyzer = RedditAnalyzer(self.openai_api_key)  # Use internal RedditAnalyzer
        self.news_analyzer = NewsAnalyzer(self.news_api_key, self.openai_api_key) if self.news_api_key else None
        
        # Initialize existing modules
        self.gas_monitor = GasMonitor(self.etherscan_api_key)
        self.whale_tracker = WhaleTracker(self.etherscan_api_key)
        self.fear_greed_calculator = FearGreedCalculator()
        
        # Initialize GAS OPTIMIZER GPT
        self.gas_optimizer = MagmaGasOptimizerFixed()
        
        # Initialize EDUCATIONAL GAS WIZARD
        self.educational_gas_wizard = EducationalGasWizard()
        
        # Initialize EDUCATIONAL CONTEXT CONNECTOR
        self.educational_connector = EducationalContextConnector(self.openai_api_key)
        
        # Initialize YIELD OPPORTUNITY SCANNER with GPT
        try:
            from yield_opportunity_scanner import YieldOpportunityScanner
            self.yield_scanner = YieldOpportunityScanner()
            logger.info("✅ Yield Opportunity Scanner initialized with GPT intelligence!")
        except ImportError as e:
            logger.warning(f"⚠️ Yield Scanner not available: {e}")
            self.yield_scanner = None
        
        logger.info("MAGMA Educational Bot initialized with all advanced modules + EDUCATIONAL CONTEXT CONNECTOR!")
    
    async def run_educational_intelligence_cycle(self) -> Dict[str, Any]:
        """Run complete educational intelligence cycle transforming market data into learning opportunities"""
        try:
            logger.info("Starting Educational Intelligence Cycle...")
            
            # 1. Market Intelligence
            market_data = await self._get_market_intelligence()
            
            # 2. Gas Intelligence
            gas_data = await self._get_gas_intelligence()
            
            # 3. Whale Intelligence
            whale_data = await self._get_whale_intelligence()
            
            # 4. Social Intelligence (LunarCrush)
            social_data = await self._get_social_intelligence()
            
            # 5. Fear & Greed Intelligence
            fear_greed_data = await self._get_fear_greed_intelligence(market_data, social_data)
            
            # 6. Reddit Intelligence
            reddit_data = await self._get_reddit_intelligence()
            
            # 7. News Intelligence
            news_data = await self._get_news_intelligence()
            
            # 8. Crypto Events Intelligence
            events_data = await self._get_events_intelligence()
            
            # 9. GPT Whale Pattern Analysis
            gpt_whale_analysis = await self._get_gpt_whale_analysis(whale_data, social_data)
            
            # 10. Alpha Signals Generation
            alpha_signals = await self._get_alpha_signals(
                market_data, gas_data, whale_data, social_data, fear_greed_data
            )
            
            # 11. YIELD OPPORTUNITY SCANNER (Educational Focus)
            yield_data = await self._get_yield_intelligence()
            
            # 12. EDUCATIONAL CONTEXT GENERATION
            educational_context = await self._generate_educational_context(
                market_data, gas_data, whale_data, social_data, fear_greed_data, news_data, yield_data
            )
            
            # 13. Educational Newsletter Generation
            newsletter = await self._generate_educational_newsletter(
                market_data, gas_data, whale_data, social_data, fear_greed_data, alpha_signals, news_data, yield_data, educational_context
            )
            
            # Compile comprehensive educational report
            report = self._compile_educational_report(
                market_data, gas_data, whale_data, social_data, fear_greed_data,
                reddit_data, news_data, events_data, gpt_whale_analysis, alpha_signals, newsletter, yield_data, educational_context
            )
            
            logger.info("Educational Intelligence Cycle completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error in Educational Intelligence Cycle: {e}")
            return self._generate_error_report(str(e))
    
    async def _generate_educational_context(self, market_data: Dict[str, Any], gas_data: Dict[str, Any],
                                          whale_data: Dict[str, Any], social_data: Dict[str, Any],
                                          fear_greed_data: Dict[str, Any], news_data: Dict[str, Any],
                                          yield_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive educational context from all intelligence data"""
        try:
            # Compile all intelligence data
            intelligence_data = {
                'market_intelligence': market_data,
                'gas_intelligence': gas_data,
                'whale_intelligence': whale_data,
                'social_intelligence': social_data,
                'fear_greed_intelligence': fear_greed_data,
                'news_intelligence': news_data,
                'yield_intelligence': yield_data
            }
            
            # Generate educational context for intermediate level (default)
            educational_context = await self.educational_connector.generate_comprehensive_learning_context(
                intelligence_data, LearningLevel.INTERMEDIATE
            )
            
            logger.info("Educational context generated successfully!")
            return educational_context
            
        except Exception as e:
            logger.error(f"Error generating educational context: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'educational_insights': [],
                'daily_learning_summary': 'Educational context temporarily unavailable.'
            }
    
    async def _generate_educational_newsletter(self, market_data: Dict[str, Any], gas_data: Dict[str, Any],
                                             whale_data: Dict[str, Any], social_data: Dict[str, Any],
                                             fear_greed: Dict[str, Any], alpha_signals: List[AlphaSignal],
                                             news_data: Dict[str, Any] = None, yield_data: Dict[str, Any] = None,
                                             educational_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate educational newsletter with learning opportunities"""
        try:
            if self.content_generator and educational_context:
                try:
                    # Create educational newsletter with learning focus
                    educational_newsletter = {
                        'title': 'Today\'s Market Learning Opportunities',
                        'educational_summary': educational_context.get('daily_learning_summary', ''),
                        'key_learning_insights': educational_context.get('educational_insights', []),
                        'recommended_actions': educational_context.get('recommended_actions', []),
                        'progress_opportunities': educational_context.get('progress_opportunities', []),
                        'market_context': self._create_educational_market_context(market_data, gas_data),
                        'learning_focus': self._determine_today_learning_focus(educational_context),
                        'next_steps': self._generate_next_learning_steps(educational_context)
                    }
                    
                    return educational_newsletter
                    
                except Exception as e:
                    logger.warning(f"Educational Content Generator error, using fallback: {e}")
                    return self._generate_fallback_educational_newsletter(educational_context)
            else:
                logger.warning("Educational Content Generator not available")
                return self._generate_fallback_educational_newsletter(educational_context)
                
        except Exception as e:
            logger.error(f"Error generating educational newsletter: {e}")
            return self._generate_fallback_educational_newsletter(educational_context)
    
    def _create_educational_market_context(self, market_data: Dict[str, Any], gas_data: Dict[str, Any]) -> str:
        """Create educational market context explanation"""
        try:
            price_change = market_data.get('price_change_24h', 0)
            current_price = market_data.get('current_price', 0)
            current_gas = gas_data.get('current_gas', 25)
            
            context_parts = []
            
            # Price context
            if abs(price_change) > 3:
                if price_change > 0:
                    context_parts.append(f"ETH surged {price_change:.1f}% to ${current_price:,.2f}")
                    context_parts.append("This demonstrates market confidence and increased demand.")
                else:
                    context_parts.append(f"ETH declined {abs(price_change):.1f}% to ${current_price:,.2f}")
                    context_parts.append("This could present a learning opportunity about market corrections.")
            
            # Gas context
            if current_gas < 20:
                context_parts.append(f"Gas fees are optimal at {current_gas} gwei - perfect for learning DeFi interactions.")
            elif current_gas > 50:
                context_parts.append(f"High gas fees ({current_gas} gwei) create a learning opportunity about Layer 2 solutions.")
            
            return " ".join(context_parts) if context_parts else "Stable market conditions provide a calm environment for learning."
            
        except Exception as e:
            logger.error(f"Error creating educational market context: {e}")
            return "Market analysis available for educational purposes."
    
    def _determine_today_learning_focus(self, educational_context: Dict[str, Any]) -> str:
        """Determine today's primary learning focus"""
        try:
            insights = educational_context.get('educational_insights', [])
            
            if not insights:
                return "General Ethereum and DeFi concepts"
            
            # Count insights by type
            focus_areas = {}
            for insight in insights:
                concept = insight.get('concept_to_learn', 'General')
                focus_areas[concept] = focus_areas.get(concept, 0) + 1
            
            # Return most frequent focus area
            if focus_areas:
                primary_focus = max(focus_areas, key=focus_areas.get)
                return primary_focus
            else:
                return "Market Analysis and DeFi Fundamentals"
                
        except Exception as e:
            logger.error(f"Error determining learning focus: {e}")
            return "Ethereum Ecosystem Understanding"
    
    def _generate_next_learning_steps(self, educational_context: Dict[str, Any]) -> List[str]:
        """Generate next learning steps based on insights"""
        try:
            recommended_actions = educational_context.get('recommended_actions', [])
            progress_opportunities = educational_context.get('progress_opportunities', [])
            
            next_steps = []
            
            # Add top recommended actions
            for action in recommended_actions[:2]:
                next_steps.append(action)
            
            # Add top progress opportunities
            for opportunity in progress_opportunities[:2]:
                next_steps.append(opportunity)
            
            # Default steps if none available
            if not next_steps:
                next_steps = [
                    "Complete today's market analysis tutorial",
                    "Practice gas optimization strategies",
                    "Study DeFi protocol fundamentals",
                    "Review social sentiment analysis"
                ]
            
            return next_steps[:4]  # Limit to 4 steps
            
        except Exception as e:
            logger.error(f"Error generating next learning steps: {e}")
            return ["Continue learning Ethereum fundamentals"]
    
    def _generate_fallback_educational_newsletter(self, educational_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback educational newsletter"""
        return {
            'title': 'Today\'s Market Learning Opportunities',
            'educational_summary': 'Educational analysis temporarily unavailable. Continue with basic Ethereum learning.',
            'key_learning_insights': [],
            'recommended_actions': ['Study Ethereum basics', 'Learn about gas fees', 'Understand DeFi concepts'],
            'progress_opportunities': ['Complete beginner tutorials', 'Practice with testnets'],
            'market_context': 'Market data available for educational analysis.',
            'learning_focus': 'Ethereum Fundamentals',
            'next_steps': ['Start with Ethereum basics', 'Understand wallet management', 'Learn about smart contracts']
        }
    
    async def _get_yield_intelligence(self) -> Dict[str, Any]:
        """Get comprehensive yield optimization intelligence"""
        try:
            if not self.yield_scanner:
                return {
                    'status': 'unavailable',
                    'message': 'Yield scanner not initialized',
                    'opportunities': []
                }
            
            logger.info("🔍 Scanning for yield opportunities...")
            
            # Get current gas price for analysis
            try:
                current_gas = await self.gas_monitor.get_current_gas_price()
                gas_price_gwei = current_gas.get('standard', 25.0)  # Default fallback
            except:
                # Fallback to default gas price
                gas_price_gwei = 25.0
            
            # Get ETH price for calculations
            try:
                eth_price = await self._get_eth_price()
            except:
                # Fallback to default ETH price
                eth_price = 4500.0
            
            # Find yield opportunities
            opportunities = await self.yield_scanner.find_combined_opportunities(gas_price_gwei, eth_price)
            
            # Get GPT market sentiment
            if opportunities:
                market_sentiment = await self.yield_scanner.gpt_analyzer.get_market_sentiment_analysis(
                    [opp.yield_opportunity for opp in opportunities]
                )
            else:
                market_sentiment = "No high-ROI opportunities detected at current gas prices."
            
            yield_intelligence = {
                'status': 'success',
                'gas_price_gwei': gas_price_gwei,
                'eth_price_usd': eth_price,
                'opportunities_count': len(opportunities),
                'market_sentiment': market_sentiment,
                'opportunities': opportunities,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'yield_optimization'
            }
            
            logger.info(f"✅ Yield intelligence gathered: {len(opportunities)} opportunities found")
            return yield_intelligence
            
        except Exception as e:
            logger.error(f"❌ Error in yield intelligence: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'opportunities': [],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_market_intelligence(self) -> Dict[str, Any]:
        """Get comprehensive market intelligence"""
        try:
            # Get ETH data from CoinGecko
            eth_data = self.coingecko.get_coin_by_id('ethereum')
            
            if eth_data:
                return {
                    'current_price': eth_data['market_data']['current_price']['usd'],
                    'price_change_24h': eth_data['market_data']['price_change_percentage_24h'],
                    'volume_24h': eth_data['market_data']['total_volume']['usd'],
                    'market_cap': eth_data['market_data']['market_cap']['usd'],
                    'dominance': eth_data['market_data'].get('market_cap_percentage', 18) / 100,
                    'volatility': eth_data['market_data'].get('price_change_percentage_24h', 0) / 100,
                    'ath': eth_data['market_data']['ath']['usd'],
                    'ath_change_percentage': eth_data['market_data']['ath_change_percentage']['usd']
                }
            
        except Exception as e:
            logger.error(f"Error getting market intelligence: {e}")
        
        # Fallback data
        return {
            'current_price': 4750.0,
            'price_change_24h': -2.5,
            'volume_24h': 25000000000,
            'market_cap': 500000000000,
            'dominance': 0.18,
            'volatility': 0.25,
            'ath': 4868.0,
            'ath_change_percentage': -2.4
        }
    
    async def _get_gas_intelligence(self) -> Dict[str, Any]:
        """Get enhanced gas intelligence with GPT optimizer"""
        try:
            # Get basic gas intelligence
            basic_gas_data = await self.gas_monitor.get_gas_intelligence()
            
            # Get GPT-enhanced gas optimization
            gas_optimization = await self._get_gas_optimization_intelligence()
            
            # Combine both
            enhanced_gas_data = {
                **basic_gas_data,
                'gpt_optimization': gas_optimization,
                'enhanced': True
            }
            
            return enhanced_gas_data
            
        except Exception as e:
            logger.error(f"Error getting gas intelligence: {e}")
            return {'current_gas': 25, 'trend': 'stable', 'status': 'Low', 'prediction': 'stable', 'enhanced': False}
    
    async def _get_gas_optimization_intelligence(self) -> Dict[str, Any]:
        """Get GPT-enhanced gas optimization intelligence"""
        try:
            # Get current gas prices
            current_gas = await self.gas_optimizer.get_current_gas_prices()
            if not current_gas:
                return {'status': 'unavailable', 'recommendations': {}}
            
            # Get market context
            market_context = self.gas_optimizer.get_market_context(current_gas)
            
            # Get GPT recommendations for key transaction types
            key_transactions = ['uniswap_swap', 'eth_transfer', 'add_liquidity', 'claim_rewards']
            gpt_recommendations = {}
            
            for tx_type in key_transactions:
                try:
                    # Calculate potential savings
                    gas_units = TRANSACTION_TYPES.get(tx_type, 21000)
                    current_cost_usd = (current_gas.standard_gas * gas_units / 1e9) * 2500  # Assume ETH = $2500
                    optimal_cost_usd = current_cost_usd * 0.75  # Assume 25% savings potential
                    savings_usd = current_cost_usd - optimal_cost_usd
                    
                    # Get GPT recommendation
                    recommendation = await self.gas_optimizer.get_gpt_recommendation(
                        current_gas, tx_type, savings_usd, market_context
                    )
                    gpt_recommendations[tx_type] = {
                        'action': recommendation.action,
                        'confidence': recommendation.confidence,
                        'reasoning': recommendation.reasoning,
                        'estimated_savings_usd': savings_usd
                    }
                    
                except Exception as e:
                    logger.error(f"Error getting GPT recommendation for {tx_type}: {e}")
                    continue
            
            # Determine overall recommendation
            wait_count = sum(1 for rec in gpt_recommendations.values() if rec['action'] == 'WAIT')
            overall_action = 'WAIT' if wait_count > len(gpt_recommendations) / 2 else 'EXECUTE'
            
            return {
                'status': 'active',
                'current_gas_gwei': current_gas.standard_gas,
                'market_context': market_context,
                'recommendations': gpt_recommendations,
                'overall_recommendation': overall_action,
                'total_transaction_types': len(gpt_recommendations),
                'wait_recommendations': wait_count
            }
            
        except Exception as e:
            logger.error(f"Error getting gas optimization intelligence: {e}")
            return {'status': 'error', 'recommendations': {}}
    
    async def _get_whale_intelligence(self) -> Dict[str, Any]:
        """Get whale intelligence"""
        try:
            return await self.whale_tracker.get_whale_intelligence()
        except Exception as e:
            logger.error(f"Error getting whale intelligence: {e}")
            return {'trend': 'accumulation', 'whale_count': 3, 'total_volume': '15500 ETH', 'network_impact': 'Medium'}
    
    async def _get_social_intelligence(self) -> Dict[str, Any]:
        """Get social intelligence from LunarCrush"""
        try:
            if self.lunarcrush_analyzer:
                try:
                    return await self.lunarcrush_analyzer.get_ethereum_social_metrics()
                except Exception as e:
                    logger.warning(f"LunarCrush API error, using fallback: {e}")
                    return self._generate_fallback_social_data()
            else:
                logger.warning("LunarCrush analyzer not available")
                return self._generate_fallback_social_data()
        except Exception as e:
            logger.error(f"Error getting social intelligence: {e}")
            return self._generate_fallback_social_data()
    
    async def _get_fear_greed_intelligence(self, market_data: Dict[str, Any], social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get Fear & Greed intelligence"""
        try:
            return await self.fear_greed_calculator.calculate_fear_greed_index(market_data, social_data)
        except Exception as e:
            logger.warning(f"Fear & Greed calculation error, using fallback: {e}")
            return {'score': 35, 'phase': 'Fear', 'interpretation': 'Fear - Potential buying opportunity'}
    
    async def _get_reddit_intelligence(self) -> Dict[str, Any]:
        """Get Reddit intelligence"""
        try:
            if self.reddit_analyzer:
                # Use the new RedditAnalyzer for comprehensive analysis
                reddit_analysis = await self.reddit_analyzer.get_comprehensive_reddit_analysis(['ethereum', 'bitcoin', 'defi'])
                
                # Format the data for the report
                return {
                    'overall_sentiment': reddit_analysis.get('overall_sentiment', {}).get('classification', 'NEUTRAL'),
                    'sentiment_score': reddit_analysis.get('overall_sentiment', {}).get('score', 0.0),
                    'trending_topics': [topic['topic'] for topic in reddit_analysis.get('trending_topics', [])],
                    'community_mood': self._determine_community_mood(reddit_analysis.get('overall_sentiment', {}).get('score', 0.0)),
                    'key_insights': self._extract_reddit_insights(reddit_analysis),
                    'engagement_metrics': reddit_analysis.get('engagement_metrics', {}),
                    'viral_posts': reddit_analysis.get('viral_posts', {}),
                    'crypto_specific': reddit_analysis.get('crypto_specific', {})
                }
            else:
                logger.warning("Reddit analyzer not available")
                return self._generate_fallback_reddit_data()
        except Exception as e:
            logger.error(f"Error getting Reddit intelligence: {e}")
            return self._generate_fallback_reddit_data()
    
    def _determine_community_mood(self, sentiment_score: float) -> str:
        """Determine community mood based on sentiment score"""
        if sentiment_score >= 0.3:
            return 'Very Optimistic'
        elif sentiment_score >= 0.1:
            return 'Optimistic'
        elif sentiment_score <= -0.3:
            return 'Very Pessimistic'
        elif sentiment_score <= -0.1:
            return 'Pessimistic'
        else:
            return 'Neutral'
    
    def _extract_reddit_insights(self, reddit_analysis: Dict[str, Any]) -> List[str]:
        """Extract key insights from Reddit analysis"""
        insights = []
        
        # Overall sentiment insight
        overall_sentiment = reddit_analysis.get('overall_sentiment', {})
        if overall_sentiment.get('classification') != 'NEUTRAL':
            insights.append(f"Community sentiment is {overall_sentiment.get('classification', 'NEUTRAL').lower()}")
        
        # Trending topics insight
        trending_topics = reddit_analysis.get('trending_topics', [])
        if trending_topics:
            top_topic = trending_topics[0]['topic']
            insights.append(f"'{top_topic.title()}' is the most discussed topic")
        
        # Viral posts insight
        viral_posts = reddit_analysis.get('viral_posts', {})
        if viral_posts.get('posts_analyzed', 0) > 0:
            insights.append(f"Analyzed {viral_posts['posts_analyzed']} viral posts for market sentiment")
        
        # Engagement insight
        engagement = reddit_analysis.get('engagement_metrics', {})
        if engagement.get('total_posts', 0) > 0:
            insights.append(f"High community engagement with {engagement['total_posts']} posts analyzed")
        
        return insights[:5]  # Limit to 5 insights
    
    async def _get_news_intelligence(self) -> Dict[str, Any]:
        """Get ETH-focused news intelligence"""
        try:
            if not self.news_analyzer:
                logger.warning("News analyzer not available")
                return self._get_fallback_news_data()
            
            logger.info("Starting ETH-focused news analysis...")
            news_analysis = await self.news_analyzer.get_comprehensive_news_analysis()
            
            # Extract key metrics
            daily_overview = news_analysis.get('daily_overview', {})
            recent_developments = news_analysis.get('recent_developments', {})
            
            return {
                'total_articles_24h': daily_overview.get('total_articles', 0),
                'recent_articles_6h': recent_developments.get('total_articles', 0),
                'overall_sentiment': news_analysis.get('market_sentiment', 'neutral'),
                'sentiment_breakdown': daily_overview.get('sentiment_breakdown', {}),
                'key_themes': daily_overview.get('key_themes', []),
                'breaking_news': news_analysis.get('breaking_news', []),
                'impact_analysis': daily_overview.get('impact_analysis', {}),
                'news_summary': daily_overview.get('summary', ''),
                'trend_analysis': news_analysis.get('trend_analysis', {}),
                'timestamp': news_analysis.get('timestamp', datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error getting news intelligence: {e}")
            return self._get_fallback_news_data()
    
    def _get_fallback_news_data(self) -> Dict[str, Any]:
        """Fallback news data when news API fails"""
        return {
            'total_articles_24h': 0,
            'recent_articles_6h': 0,
            'overall_sentiment': 'neutral',
            'sentiment_breakdown': {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0},
            'key_themes': [],
            'breaking_news': [],
            'impact_analysis': {'average_impact': 0, 'high_impact_count': 0, 'max_impact': 0},
            'news_summary': 'News analysis unavailable. Check NewsAPI configuration.',
            'trend_analysis': {},
            'timestamp': datetime.now().isoformat(),
            'error': True
        }
    
    async def _get_events_intelligence(self) -> Dict[str, Any]:
        """Get crypto events intelligence"""
        try:
            try:
                upcoming_events = await self.crypto_events_calendar.get_upcoming_events(30)
                market_catalysts = await self.crypto_events_calendar.get_market_catalysts(30)
                
                return {
                    'upcoming_events': len(upcoming_events),
                    'market_catalysts': market_catalysts,
                    'critical_events': len(market_catalysts.get('critical_events', [])),
                    'bullish_events': len(market_catalysts.get('bullish_events', []))
                }
            except Exception as e:
                logger.warning(f"Events Calendar API error, using fallback: {e}")
                return {'upcoming_events': 0, 'critical_events': 0, 'bullish_events': 0}
        except Exception as e:
            logger.error(f"Error getting events intelligence: {e}")
            return {'upcoming_events': 0, 'critical_events': 0, 'bullish_events': 0}
    
    async def _get_gpt_whale_analysis(self, whale_data: Dict[str, Any], social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get GPT-powered whale pattern analysis"""
        try:
            if self.gpt_whale_analyzer:
                try:
                    return await self.gpt_whale_analyzer.analyze_whale_patterns_with_gpt(whale_data, social_data)
                except Exception as e:
                    logger.warning(f"GPT API error, using fallback: {e}")
                    return self._generate_fallback_gpt_analysis(whale_data)
            else:
                logger.warning("GPT Whale Analyzer not available")
                return self._generate_fallback_gpt_analysis(whale_data)
        except Exception as e:
            logger.error(f"Error getting GPT whale analysis: {e}")
            return self._generate_fallback_gpt_analysis(whale_data)
    
    async def _get_alpha_signals(self, market_data: Dict[str, Any], gas_data: Dict[str, Any], 
                                whale_data: Dict[str, Any], social_data: Dict[str, Any], 
                                fear_greed: Dict[str, Any]) -> List[AlphaSignal]:
        """Get alpha signals"""
        try:
            try:
                return await self.alpha_signals_generator.generate_alpha_signals(
                    market_data, gas_data, whale_data, social_data, fear_greed
                )
            except Exception as e:
                logger.warning(f"Alpha Signals API error, using fallback: {e}")
                return []
        except Exception as e:
            logger.error(f"Error getting alpha signals: {e}")
            return []
    
    async def _generate_market_newsletter(self, market_data: Dict[str, Any], gas_data: Dict[str, Any],
                                        whale_data: Dict[str, Any], social_data: Dict[str, Any],
                                        fear_greed: Dict[str, Any], alpha_signals: List[AlphaSignal],
                                        news_data: Dict[str, Any] = None, yield_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate market newsletter"""
        try:
            if self.content_generator:
                try:
                    # Convert AlphaSignal objects to dictionaries for compatibility
                    alpha_signals_dict = [signal.__dict__ for signal in alpha_signals] if alpha_signals else []
                    return await self.content_generator.generate_market_newsletter(
                        market_data, gas_data, whale_data, social_data, fear_greed, alpha_signals_dict
                    )
                except Exception as e:
                    logger.warning(f"Content Generator API error, using fallback: {e}")
                    return self._generate_fallback_newsletter()
            else:
                logger.warning("Content Generator not available")
                return self._generate_fallback_newsletter()
        except Exception as e:
            logger.error(f"Error generating newsletter: {e}")
            return self._generate_fallback_newsletter()
    
    def _compile_educational_report(self, market_data: Dict[str, Any], gas_data: Dict[str, Any],
                                   whale_data: Dict[str, Any], social_data: Dict[str, Any],
                                   fear_greed: Dict[str, Any], reddit_data: Dict[str, Any],
                                   news_data: Dict[str, Any], events_data: Dict[str, Any], 
                                   gpt_whale_analysis: Dict[str, Any], alpha_signals: List[AlphaSignal], 
                                   newsletter: Dict[str, Any], yield_data: Dict[str, Any] = None,
                                   educational_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compile comprehensive educational intelligence report"""
        
        # Calculate overall confidence
        confidence_scores = []
        if market_data: confidence_scores.append(85)
        if gas_data: confidence_scores.append(80)
        if whale_data: confidence_scores.append(75)
        if social_data: confidence_scores.append(80)
        if fear_greed: confidence_scores.append(85)
        if reddit_data: confidence_scores.append(70)
        if news_data: confidence_scores.append(85)
        if events_data: confidence_scores.append(90)
        if gpt_whale_analysis: confidence_scores.append(75)
        if alpha_signals: confidence_scores.append(80)
        if newsletter: confidence_scores.append(85)
        if yield_data and yield_data.get('status') == 'success': confidence_scores.append(90)  # Yield scanner is high confidence
        if educational_context and educational_context.get('total_insights', 0) > 0: confidence_scores.append(95)  # Educational context is highest confidence
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_confidence': round(overall_confidence, 1),
            'intelligence_summary': {
                'market_intelligence': market_data,
                'gas_intelligence': gas_data,
                'whale_intelligence': whale_data,
                'social_intelligence': social_data,
                'fear_greed_intelligence': fear_greed,
                'reddit_intelligence': reddit_data,
                'news_intelligence': news_data,
                'events_intelligence': events_data,
                'gpt_whale_analysis': gpt_whale_analysis,
                'alpha_signals': [signal.__dict__ for signal in alpha_signals] if alpha_signals else [],
                'newsletter': newsletter,
                'yield_intelligence': yield_data,
                'educational_context': educational_context
            },
            'educational_insights': self._extract_educational_insights(
                market_data, gas_data, whale_data, social_data, fear_greed, alpha_signals, yield_data, educational_context
            ),
            'learning_opportunities': self._generate_learning_opportunities(educational_context),
            'educational_recommendations': self._generate_educational_recommendations(educational_context),
            'progress_tracking': self._generate_progress_tracking(educational_context),
            'market_context_for_learning': self._generate_market_context_for_learning(
                market_data, social_data, fear_greed, events_data, educational_context
            )
        }
    
    def _extract_key_insights(self, market_data: Dict[str, Any], gas_data: Dict[str, Any],
                             whale_data: Dict[str, Any], social_data: Dict[str, Any],
                             fear_greed: Dict[str, Any], alpha_signals: List[AlphaSignal], 
                             yield_data: Dict[str, Any] = None) -> List[str]:
        """Extract key insights from all data sources"""
        insights = []
        
        # Market insights
        if market_data.get('price_change_24h', 0) < -5:
            insights.append("Significant price decline suggests potential buying opportunity")
        elif market_data.get('price_change_24h', 0) > 5:
            insights.append("Strong price momentum indicates bullish sentiment")
        
        # Gas insights
        if gas_data.get('current_gas', 0) < 20:
            insights.append("Low gas prices create excellent conditions for DeFi transactions")
        elif gas_data.get('current_gas', 0) > 50:
            insights.append("High gas prices suggest waiting for better transaction conditions")
        
        # Whale insights
        if whale_data.get('trend') == 'accumulation':
            insights.append("Whale accumulation pattern suggests institutional confidence")
        elif whale_data.get('trend') == 'distribution':
            insights.append("Whale distribution pattern indicates potential selling pressure")
        
        # Social insights
        if social_data.get('ethereum_metrics', {}).get('social_score', 0) > 70:
            insights.append("High social sentiment indicates strong community confidence")
        
        # Fear & Greed insights
        fear_greed_score = fear_greed.get('score', 50)
        if fear_greed_score <= 25:
            insights.append("Extreme fear suggests contrarian buying opportunity")
        elif fear_greed_score >= 75:
            insights.append("Extreme greed indicates potential profit-taking opportunity")
        
        # Alpha signals insights
        if alpha_signals:
            top_signal = alpha_signals[0]
            insights.append(f"Top alpha signal: {top_signal.signal_type} with {top_signal.confidence:.1f}% confidence")
        
        # Yield optimization insights
        if yield_data and yield_data.get('status') == 'success':
            opportunities_count = yield_data.get('opportunities_count', 0)
            if opportunities_count > 0:
                top_opportunity = yield_data.get('opportunities', [{}])[0]
                roi = top_opportunity.get('yield_opportunity', {}).get('roi_on_gas', 0)
                insights.append(f"🔥 {opportunities_count} yield opportunities detected with top ROI: {roi:.0f}%")
            else:
                insights.append("⛽ Current gas prices not optimal for yield optimization")
        
        return insights[:10]  # Increased limit to include yield insights
    
    def _assess_overall_risk(self, market_data: Dict[str, Any], whale_data: Dict[str, Any],
                            fear_greed: Dict[str, Any], alpha_signals: List[AlphaSignal]) -> Dict[str, Any]:
        """Assess overall market risk"""
        risk_factors = []
        risk_level = "LOW"
        
        # Market volatility risk
        if abs(market_data.get('price_change_24h', 0)) > 10:
            risk_factors.append("High price volatility")
            risk_level = "MEDIUM"
        
        # Whale activity risk
        if whale_data.get('network_impact') == 'High':
            risk_factors.append("High whale network impact")
            risk_level = "MEDIUM"
        
        # Fear & Greed risk
        fear_greed_score = fear_greed.get('score', 50)
        if fear_greed_score <= 20 or fear_greed_score >= 80:
            risk_factors.append("Extreme market sentiment")
            risk_level = "MEDIUM"
        
        # Alpha signals risk
        if alpha_signals:
            high_risk_signals = [s for s in alpha_signals if s.risk_level in ['HIGH', 'EXTREME']]
            if high_risk_signals:
                risk_factors.append(f"{len(high_risk_signals)} high-risk alpha signals")
                risk_level = "HIGH"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get risk management recommendation"""
        recommendations = {
            'LOW': 'Standard monitoring and risk management',
            'MEDIUM': 'Increased monitoring and position review',
            'HIGH': 'Active risk management and position adjustments'
        }
        return recommendations.get(risk_level, 'Standard monitoring')
    
    def _generate_trading_recommendations(self, alpha_signals: List[AlphaSignal]) -> List[Dict[str, Any]]:
        """Generate trading recommendations from alpha signals"""
        recommendations = []
        
        if not alpha_signals:
            return [{'action': 'HOLD', 'reasoning': 'No clear signals available', 'confidence': 50}]
        
        # Group signals by type
        signal_groups = {}
        for signal in alpha_signals:
            signal_type = signal.signal_type
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # Generate recommendations for each signal type
        for signal_type, signals in signal_groups.items():
            if signals:
                top_signal = max(signals, key=lambda x: x.confidence)
                recommendations.append({
                    'action': signal_type,
                    'reasoning': top_signal.reasoning[:100] + "..." if len(top_signal.reasoning) > 100 else top_signal.reasoning,
                    'confidence': top_signal.confidence,
                    'timeframe': top_signal.timeframe,
                    'risk_level': top_signal.risk_level
                })
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _generate_market_outlook(self, market_data: Dict[str, Any], social_data: Dict[str, Any],
                                fear_greed: Dict[str, Any], events_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market outlook"""
        outlook = {
            'short_term': 'Neutral',
            'medium_term': 'Neutral',
            'key_factors': [],
            'catalysts': []
        }
        
        # Short-term outlook based on current data
        price_change = market_data.get('price_change_24h', 0)
        if price_change > 3:
            outlook['short_term'] = 'Bullish'
        elif price_change < -3:
            outlook['short_term'] = 'Bearish'
        
        # Medium-term outlook based on sentiment and events
        fear_greed_score = fear_greed.get('score', 50)
        if fear_greed_score <= 30:
            outlook['medium_term'] = 'Bullish'
            outlook['key_factors'].append('Extreme fear suggests contrarian opportunity')
        elif fear_greed_score >= 70:
            outlook['medium_term'] = 'Bearish'
            outlook['key_factors'].append('Extreme greed suggests potential reversal')
        
        # Add social sentiment factor
        social_score = social_data.get('ethereum_metrics', {}).get('social_score', 50)
        if social_score > 60:
            outlook['key_factors'].append('Positive social sentiment')
        elif social_score < 40:
            outlook['key_factors'].append('Negative social sentiment')
        
        # Add upcoming events as catalysts
        if events_data.get('critical_events', 0) > 0:
            outlook['catalysts'].append(f"{events_data['critical_events']} critical events upcoming")
        if events_data.get('bullish_events', 0) > 0:
            outlook['catalysts'].append(f"{events_data['bullish_events']} bullish events upcoming")
        
        return outlook
    
    def _generate_fallback_social_data(self) -> Dict[str, Any]:
        """Generate fallback social data"""
        return {
            'ethereum_metrics': {
                'social_score': 75,
                'social_volume': 1500000,
                'social_contributors': 50000,
                'social_engagement': 85,
                'altrank': 1,
                'galaxy_score': 85
            },
            'global_sentiment': {
                'market_sentiment': 'Bullish - Positive social sentiment with good volume'
            },
            'trending_coins': []
        }
    
    def _generate_fallback_reddit_data(self) -> Dict[str, Any]:
        """Generate fallback Reddit data"""
        return {
            'overall_sentiment': 'Neutral',
            'sentiment_score': 0.0,
            'trending_topics': ['Ethereum', 'DeFi', 'Scaling'],
            'community_mood': 'Mixed',
            'key_insights': ['Analysis completed using available data']
        }
    
    def _generate_fallback_gpt_analysis(self, whale_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback GPT analysis"""
        return {
            'whale_patterns': f"Analysis based on {whale_data.get('whale_count', 0)} whale transactions",
            'risk_assessment': "Standard risk assessment recommended",
            'trading_recommendations': "Monitor whale activity for signals",
            'confidence_score': 60.0,
            'risk_level': "MEDIUM"
        }
    
    def _generate_fallback_newsletter(self) -> Dict[str, Any]:
        """Generate fallback newsletter"""
        return {
            'title': 'ETH Market Intelligence Report',
            'summary': 'Analysis completed using available data sources',
            'key_takeaways': ['Market analysis completed', 'Data integrated', 'Insights generated']
        }
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'status': 'ERROR',
            'message': 'Intelligence cycle failed. Check logs for details.'
        }
    
    def print_supreme_report(self, report: Dict[str, Any]):
        """Print comprehensive intelligence report"""
        print("\n" + "="*80)
        print("🚀 MAGMA SUPREME INTELLIGENCE REPORT")
        print("="*80)
        print(f"📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")
        print(f"🎯 Overall Confidence: {report.get('overall_confidence', 0)}%")
        
        if 'error' in report:
            print(f"❌ ERROR: {report['error']}")
            return
        
        # Market Intelligence
        market = report['intelligence_summary']['market_intelligence']
        print(f"\n📊 MARKET INTELLIGENCE")
        print(f"   Price: ${market.get('current_price', 'N/A'):,.2f}")
        print(f"   24h Change: {market.get('price_change_24h', 'N/A'):+.2f}%")
        print(f"   Volume: ${market.get('volume_24h', 'N/A'):,.0f}")
        print(f"   Market Cap: ${market.get('market_cap', 'N/A'):,.0f}")
        
        # Gas Intelligence
        gas = report['intelligence_summary']['gas_intelligence']
        print(f"\n⛽ GAS INTELLIGENCE")
        print(f"   Current: {gas.get('current_gas', 'N/A')} gwei")
        print(f"   Status: {gas.get('status', 'N/A')}")
        print(f"   Trend: {gas.get('trend', 'N/A')}")
        
        # GPT Gas Optimization
        if gas.get('enhanced') and gas.get('gpt_optimization'):
            gpt_opt = gas['gpt_optimization']
            if gpt_opt.get('status') == 'active':
                print(f"\n🤖 GPT GAS OPTIMIZER")
                print(f"   Overall Action: {gpt_opt.get('overall_recommendation', 'N/A')}")
                print(f"   Analyzed Transactions: {gpt_opt.get('total_transaction_types', 0)}")
                print(f"   'WAIT' Recommendations: {gpt_opt.get('wait_recommendations', 0)}")
                
                # Show top 2 recommendations
                recommendations = gpt_opt.get('recommendations', {})
                if recommendations:
                    print(f"   🎯 Top GPT Recommendations:")
                    for i, (tx_type, rec) in enumerate(list(recommendations.items())[:2], 1):
                        tx_name = tx_type.replace('_', ' ').title()
                        print(f"      {i}. {tx_name}: {rec.get('action', 'N/A')} ({rec.get('confidence', 0):.0%})")
                        if rec.get('estimated_savings_usd', 0) > 0:
                            print(f"         💰 Potential Savings: ${rec.get('estimated_savings_usd', 0):.2f}")
            else:
                print(f"   🤖 GPT Gas Optimizer: {gpt_opt.get('status', 'inactive')}")
        
        # Whale Intelligence
        whale = report['intelligence_summary']['whale_intelligence']
        print(f"\n🐋 WHALE INTELLIGENCE")
        print(f"   Trend: {whale.get('trend', 'N/A')}")
        print(f"   Count: {whale.get('whale_count', 'N/A')}")
        print(f"   Volume: {whale.get('total_volume', 'N/A')}")
        
        # Social Intelligence
        social = report['intelligence_summary']['social_intelligence']
        if social.get('ethereum_metrics'):
            eth_metrics = social['ethereum_metrics']
            print(f"\n🌙 SOCIAL INTELLIGENCE")
            print(f"   Social Score: {eth_metrics.get('social_score', 'N/A')}")
            print(f"   Social Volume: {eth_metrics.get('social_volume', 'N/A'):,}")
            print(f"   Galaxy Score: {eth_metrics.get('galaxy_score', 'N/A')}")
        
        # Reddit Intelligence
        reddit = report['intelligence_summary']['reddit_intelligence']
        if reddit:
            print(f"\n📱 REDDIT INTELLIGENCE")
            print(f"   Overall Sentiment: {reddit.get('overall_sentiment', 'N/A')}")
            print(f"   Sentiment Score: {reddit.get('sentiment_score', 'N/A'):+.3f}")
            print(f"   Community Mood: {reddit.get('community_mood', 'N/A')}")
            if reddit.get('engagement_metrics'):
                engagement = reddit['engagement_metrics']
                print(f"   Posts Analyzed: {engagement.get('total_posts', 'N/A')}")
                print(f"   Total Comments: {engagement.get('total_comments', 'N/A')}")
        
        # Fear & Greed
        fg = report['intelligence_summary']['fear_greed_intelligence']
        print(f"\n😨 FEAR & GREED INDEX")
        print(f"   Score: {fg.get('score', 'N/A')}")
        print(f"   Phase: {fg.get('phase', 'N/A')}")
        
        # Alpha Signals
        alpha_signals = report['intelligence_summary']['alpha_signals']
        if alpha_signals:
            print(f"\n🎯 ALPHA SIGNALS ({len(alpha_signals)})")
            for i, signal in enumerate(alpha_signals[:3], 1):
                print(f"   {i}. {signal.get('signal_type', 'N/A')} - {signal.get('strength', 'N/A')} ({signal.get('confidence', 'N/A')}%)")
        
        # Key Insights
        insights = report.get('key_insights', [])
        if insights:
            print(f"\n💡 KEY INSIGHTS")
            for insight in insights[:5]:
                print(f"   • {insight}")
        
        # Risk Assessment
        risk = report.get('risk_assessment', {})
        if risk:
            print(f"\n⚠️ RISK ASSESSMENT")
            print(f"   Level: {risk.get('risk_level', 'N/A')}")
            print(f"   Recommendation: {risk.get('recommendation', 'N/A')}")
        
        # Trading Recommendations
        trading = report.get('trading_recommendations', [])
        if trading:
            print(f"\n📈 TRADING RECOMMENDATIONS")
            for rec in trading[:3]:
                print(f"   • {rec.get('action', 'N/A')}: {rec.get('reasoning', 'N/A')[:60]}...")
        
        # Market Outlook
        outlook = report.get('market_outlook', {})
        if outlook:
            print(f"\n🔮 MARKET OUTLOOK")
            print(f"   Short-term: {outlook.get('short_term', 'N/A')}")
            print(f"   Medium-term: {outlook.get('medium_term', 'N/A')}")
        
        print("\n" + "="*80)
        print("🤖 Powered by MAGMA Supreme Intelligence Bot")
        print("="*80)
    
    async def show_educational_gas_alert(self, report: Dict[str, Any]):
        """Show Educational Gas Wizard alert based on current report"""
        try:
            # Extract gas data from report
            gas_intelligence = report['intelligence_summary']['gas_intelligence']
            
            # Create simplified gas data for wizard
            gas_data = {
                'current_gas': gas_intelligence.get('current_gas', 25),
                'status': gas_intelligence.get('status', 'Unknown'),
                'trend': gas_intelligence.get('trend', 'stable')
            }
            
            # Generate educational alert
            educational_alert = await self.educational_gas_wizard.generate_educational_alert(
                gas_data, user_id="magma_user"
            )
            
            # Display the educational alert
            self.educational_gas_wizard.display_educational_alert(educational_alert)
            
        except Exception as e:
            logger.error(f"Error showing educational gas alert: {e}")
            print(f"\n🎓 Educational Gas Wizard temporarily unavailable: {e}")

# Existing classes (GasMonitor, WhaleTracker, FearGreedCalculator)
class GasMonitor:
    """Monitor Ethereum gas prices and trends"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
    
    async def get_gas_intelligence(self) -> Dict[str, Any]:
        """Get comprehensive gas intelligence"""
        try:
            # Get current gas prices
            gas_prices = await self._get_current_gas_prices()
            
            # Analyze trends
            trend = self._analyze_gas_trend(gas_prices)
            
            # Determine status
            status = self._determine_gas_status(gas_prices)
            
            # Make prediction
            prediction = self._predict_gas_trend(trend)
            
            return {
                'current_gas': gas_prices.get('SafeLow', 25),
                'trend': trend,
                'status': status,
                'prediction': prediction,
                'gas_prices': gas_prices
            }
            
        except Exception as e:
            logger.error(f'Error getting gas intelligence: {e}')
            return {'current_gas': 25, 'trend': 'stable', 'status': 'Low', 'prediction': 'stable'}
    
    async def _get_current_gas_prices(self) -> Dict[str, Any]:
        """Get current gas prices from Etherscan"""
        try:
            if not self.api_key:
                return {'SafeLow': 25, 'Standard': 30, 'Fast': 35, 'Fastest': 40}
            
            params = {
                'module': 'gastracker',
                'action': 'gasoracle',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    result = data['result']
                    return {
                        'SafeLow': int(float(result.get('SafeLow', 25))),
                        'Standard': int(float(result.get('ProposeGasPrice', 30))),
                        'Fast': int(float(result.get('FastGasPrice', 35))),
                        'Fastest': int(float(result.get('FastGasPrice', 40)))
                    }
            
        except Exception as e:
            logger.error(f'Error getting gas prices: {e}')
        
        return {'SafeLow': 25, 'Standard': 30, 'Fast': 35, 'Fastest': 40}
    
    def _analyze_gas_trend(self, gas_prices: Dict[str, Any]) -> str:
        """Analyze gas price trend"""
        try:
            safe_low = gas_prices.get('SafeLow', 25)
            
            if safe_low < 20:
                return 'decreasing'
            elif safe_low > 50:
                return 'increasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f'Error analyzing gas trend: {e}')
            return 'stable'
    
    def _determine_gas_status(self, gas_prices: Dict[str, Any]) -> str:
        """Determine gas price status"""
        try:
            safe_low = gas_prices.get('SafeLow', 25)
            
            if safe_low < 20:
                return 'Low'
            elif safe_low < 35:
                return 'Normal'
            elif safe_low < 50:
                return 'High'
            else:
                return 'Extreme'
                
        except Exception as e:
            logger.error(f'Error determining gas status: {e}')
            return 'Normal'
    
    def _predict_gas_trend(self, trend: str) -> str:
        """Predict gas price trend"""
        try:
            if trend == 'decreasing':
                return 'decreasing'
            elif trend == 'increasing':
                return 'increasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f'Error predicting gas trend: {e}')
            return 'stable'

class WhaleTracker:
    """Track large Ethereum transactions and whale activity"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
    
    async def get_whale_intelligence(self) -> Dict[str, Any]:
        """Get comprehensive whale intelligence"""
        try:
            # Get recent large transactions
            large_transfers = await self._get_large_transfers()
            
            # Analyze whale activity
            trend = self._analyze_whale_trend(large_transfers)
            whale_count = self._count_whales(large_transfers)
            total_volume = self._calculate_total_volume(large_transfers)
            network_impact = self._assess_network_impact(large_transfers)
            
            return {
                'trend': trend,
                'whale_count': whale_count,
                'total_volume': total_volume,
                'network_impact': network_impact,
                'large_transfers': large_transfers
            }
            
        except Exception as e:
            logger.error(f'Error getting whale intelligence: {e}')
            return {'trend': 'unknown', 'whale_count': 0, 'total_volume': '0 ETH', 'network_impact': 'Low'}
    
    async def _get_large_transfers(self) -> List[Dict[str, Any]]:
        """Get recent large ETH transfers"""
        try:
            if not self.api_key:
                return self._generate_mock_transfers()
            
            # For now, return mock data
            # In real implementation, fetch from Etherscan API
            return self._generate_mock_transfers()
            
        except Exception as e:
            logger.error(f'Error getting large transfers: {e}')
            return self._generate_mock_transfers()
    
    def _generate_mock_transfers(self) -> List[Dict[str, Any]]:
        """Generate mock transfer data"""
        return [
            {'from': '0x1234...', 'to': '0x5678...', 'value': '5000 ETH', 'type': 'accumulation'},
            {'from': '0x9abc...', 'to': '0xdef0...', 'value': '3000 ETH', 'type': 'distribution'},
            {'from': '0x1111...', 'to': '0x2222...', 'value': '7500 ETH', 'type': 'accumulation'}
        ]
    
    def _analyze_whale_trend(self, transfers: List[Dict[str, Any]]) -> str:
        """Analyze whale activity trend"""
        try:
            if not transfers:
                return 'unknown'
            
            accumulation_count = sum(1 for t in transfers if t.get('type') == 'accumulation')
            distribution_count = sum(1 for t in transfers if t.get('type') == 'distribution')
            
            if accumulation_count > distribution_count:
                return 'accumulation'
            elif distribution_count > accumulation_count:
                return 'distribution'
            else:
                return 'mixed'
                
        except Exception as e:
            logger.error(f'Error analyzing whale trend: {e}')
            return 'unknown'
    
    def _count_whales(self, transfers: List[Dict[str, Any]]) -> int:
        """Count unique whale addresses"""
        try:
            unique_addresses = set()
            for transfer in transfers:
                unique_addresses.add(transfer.get('from', ''))
                unique_addresses.add(transfer.get('to', ''))
            
            return len(unique_addresses)
            
        except Exception as e:
            logger.error(f'Error counting whales: {e}')
            return 0
    
    def _calculate_total_volume(self, transfers: List[Dict[str, Any]]) -> str:
        """Calculate total volume of transfers"""
        try:
            total_eth = 0
            for transfer in transfers:
                value_str = transfer.get('value', '0 ETH')
                try:
                    eth_amount = float(value_str.replace(' ETH', ''))
                    total_eth += eth_amount
                except ValueError:
                    continue
            
            return f"{total_eth:,.0f} ETH"
            
        except Exception as e:
            logger.error(f'Error calculating total volume: {e}')
            return '0 ETH'
    
    def _assess_network_impact(self, transfers: List[Dict[str, Any]]) -> str:
        """Assess network impact of whale activity"""
        try:
            if not transfers:
                return 'Low'
            
            total_volume = 0
            for transfer in transfers:
                value_str = transfer.get('value', '0 ETH')
                try:
                    eth_amount = float(value_str.replace(' ETH', ''))
                    total_volume += eth_amount
                except ValueError:
                    continue
            
            if total_volume > 10000:
                return 'High'
            elif total_volume > 5000:
                return 'Medium'
            else:
                return 'Low'
                
        except Exception as e:
            logger.error(f'Error assessing network impact: {e}')
            return 'Low'

class FearGreedCalculator:
    """Calculate Fear & Greed Index"""
    
    async def calculate_fear_greed_index(self, market_data: Dict[str, Any], social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive Fear & Greed Index"""
        try:
            # Calculate base score from market data
            base_score = self._calculate_base_score(market_data)
            
            # Adjust for social sentiment
            social_adjustment = self._calculate_social_adjustment(social_data)
            
            # Calculate final score
            final_score = max(0, min(100, base_score + social_adjustment))
            
            # Determine phase
            phase = self._determine_phase(final_score)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(final_score, phase)
            
            return {
                'score': round(final_score, 1),
                'phase': phase,
                'interpretation': interpretation,
                'base_score': base_score,
                'social_adjustment': social_adjustment
            }
            
        except Exception as e:
            logger.error(f'Error calculating Fear & Greed Index: {e}')
            return {'score': 50, 'phase': 'Neutral', 'interpretation': 'Unable to calculate'}
    
    def _calculate_base_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate base score from market data"""
        try:
            score = 50.0  # Start neutral
            
            # Price change adjustment
            price_change = market_data.get('price_change_24h', 0)
            if price_change > 10:
                score += 20  # Extreme greed
            elif price_change > 5:
                score += 15  # Greed
            elif price_change > 2:
                score += 10  # Slight greed
            elif price_change < -10:
                score -= 20  # Extreme fear
            elif price_change < -5:
                score -= 15  # Fear
            elif price_change < -2:
                score -= 10  # Slight fear
            
            # Volume adjustment
            volume_24h = market_data.get('volume_24h', 0)
            market_cap = market_data.get('market_cap', 1)
            volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
            
            if volume_ratio > 0.1:
                score += 10  # High volume = greed
            elif volume_ratio < 0.05:
                score -= 10  # Low volume = fear
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f'Error calculating base score: {e}')
            return 50.0
    
    def _calculate_social_adjustment(self, social_data: Dict[str, Any]) -> float:
        """Calculate social sentiment adjustment"""
        try:
            adjustment = 0.0
            
            # ETH social score adjustment
            eth_social_score = social_data.get('ethereum_metrics', {}).get('social_score', 50)
            if eth_social_score > 70:
                adjustment += 10  # Positive social sentiment
            elif eth_social_score < 30:
                adjustment -= 10  # Negative social sentiment
            
            # Global sentiment adjustment
            global_sentiment = social_data.get('global_sentiment', {}).get('market_sentiment', '')
            if 'Bullish' in global_sentiment:
                adjustment += 5
            elif 'Bearish' in global_sentiment:
                adjustment -= 5
            
            return adjustment
            
        except Exception as e:
            logger.error(f'Error calculating social adjustment: {e}')
            return 0.0
    
    def _determine_phase(self, score: float) -> str:
        """Determine Fear & Greed phase"""
        if score >= 80:
            return 'Extreme Greed'
        elif score >= 60:
            return 'Greed'
        elif score >= 40:
            return 'Neutral'
        elif score >= 20:
            return 'Fear'
        else:
            return 'Extreme Fear'
    
    def _generate_interpretation(self, score: float, phase: str) -> str:
        """Generate interpretation of the index"""
        if phase == 'Extreme Greed':
            return 'Extreme Greed - Market sentiment overly optimistic. Consider taking profits.'
        elif phase == 'Greed':
            return 'Greed - Positive sentiment. Monitor for potential reversal signals.'
        elif phase == 'Neutral':
            return 'Neutral - Balanced market sentiment. Standard monitoring recommended.'
        elif phase == 'Fear':
            return 'Fear - Negative sentiment. Potential accumulation opportunity.'
        else:  # Extreme Fear
            return 'Extreme Fear - Market sentiment overly pessimistic. Historical buying opportunity.'
    
    async def _get_eth_price(self) -> float:
        """Get current ETH price from market data"""
        try:
            # Try to get from market intelligence first
            market_data = await self._get_market_intelligence()
            if market_data and 'current_price' in market_data:
                return float(market_data['current_price'])
            
            # Fallback to default price
            return 4500.0
            
        except Exception as e:
            logger.error(f"Error getting ETH price: {e}")
            return 4500.0

async def main():
    """Main function to run the MAGMA Supreme Bot"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize the bot
        bot = MagmaSupremeBot()
        
        # Run the educational intelligence cycle
        report = await bot.run_educational_intelligence_cycle()
        
        # Print the comprehensive report
        bot.print_supreme_report(report)
        
        # 🎓 EDUCATIONAL GAS WIZARD ALERT
        await bot.show_educational_gas_alert(report)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'logs/magma_report_{timestamp}.json'
        
        try:
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {report_file}")
        except Exception as e:
            logger.error(f'Error saving report: {e}')
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
