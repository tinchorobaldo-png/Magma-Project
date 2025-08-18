#!/usr/bin/env python3
"""
COMMUNITY INTELLIGENCE ENGINE - Advanced Technical Features for MAGMA
Analyzes multiple community sources to identify trending concepts and learning opportunities
"""

import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import os
from collections import Counter

@dataclass
class CommunityPost:
    """Represents a post from any community source"""
    source: str  # reddit, discord, telegram, twitter, github
    content: str
    author: str
    timestamp: datetime
    engagement: int  # upvotes, likes, reactions
    url: str
    sentiment: str  # positive, negative, neutral
    topics: List[str]
    educational_value: int  # 1-100

@dataclass
class TrendingConcept:
    """Represents a concept that's trending in the community"""
    concept: str
    frequency: int
    sources: List[str]
    sentiment_score: float
    engagement_score: int
    educational_opportunity: str
    related_topics: List[str]
    difficulty_level: str
    estimated_learning_time: int

class CommunitySourceAnalyzer:
    """Analyzes content from different community sources"""
    
    def __init__(self):
        self.ethereum_keywords = [
            'ethereum', 'eth', 'defi', 'nft', 'dao', 'governance',
            'smart contract', 'layer 2', 'rollup', 'staking', 'yield',
            'liquidity', 'amm', 'flash loan', 'oracle', 'bridge',
            'metamask', 'uniswap', 'aave', 'compound', 'curve'
        ]
        
        self.educational_patterns = {
            'question': [
                r'how\s+do\s+i',
                r'what\s+is',
                r'can\s+someone\s+explain',
                r'i\s+don\'t\s+understand',
                r'help\s+me\s+understand'
            ],
            'tutorial_request': [
                r'tutorial',
                r'guide',
                r'step\s+by\s+step',
                r'walkthrough',
                r'learn'
            ],
            'concept_explanation': [
                r'explanation',
                r'breakdown',
                r'overview',
                r'introduction',
                r'basics'
            ]
        }
    
    def analyze_reddit_content(self, posts: List[Dict[str, Any]]) -> List[CommunityPost]:
        """Analyze Reddit posts for educational content"""
        analyzed_posts = []
        
        for post in posts:
            content = f"{post.get('title', '')} {post.get('selftext', '')}"
            
            # Extract topics
            topics = self._extract_topics(content)
            
            # Determine sentiment
            sentiment = self._analyze_sentiment(content)
            
            # Calculate educational value
            educational_value = self._calculate_educational_value(content, post)
            
            # Extract topics
            topics = self._extract_topics(content)
            
            community_post = CommunityPost(
                source='reddit',
                content=content[:500],  # Limit content length
                author=post.get('author', 'Unknown'),
                timestamp=datetime.fromtimestamp(post.get('created_utc', time.time())),
                engagement=post.get('score', 0),
                url=f"https://reddit.com{post.get('permalink', '')}",
                sentiment=sentiment,
                topics=topics,
                educational_value=educational_value
            )
            
            analyzed_posts.append(community_post)
        
        return analyzed_posts
    
    def analyze_twitter_content(self, tweets: List[Dict[str, Any]]) -> List[CommunityPost]:
        """Analyze Twitter content for educational value"""
        analyzed_posts = []
        
        for tweet in tweets:
            content = tweet.get('text', '')
            
            # Extract topics
            topics = self._extract_topics(content)
            
            # Determine sentiment
            sentiment = self._analyze_sentiment(content)
            
            # Calculate educational value
            educational_value = self._calculate_educational_value(content, tweet)
            
            community_post = CommunityPost(
                source='twitter',
                content=content,
                author=tweet.get('author_id', 'Unknown'),
                timestamp=datetime.now(),  # Twitter API timestamp handling
                engagement=tweet.get('public_metrics', {}).get('like_count', 0),
                url=f"https://twitter.com/user/status/{tweet.get('id', '')}",
                sentiment=sentiment,
                topics=topics,
                educational_value=educational_value
            )
            
            analyzed_posts.append(community_post)
        
        return analyzed_posts
    
    def analyze_github_content(self, repos: List[Dict[str, Any]]) -> List[CommunityPost]:
        """Analyze GitHub repositories for educational content"""
        analyzed_posts = []
        
        for repo in repos:
            content = f"{repo.get('name', '')} {repo.get('description', '')}"
            
            # Extract topics
            topics = self._extract_topics(content)
            
            # Determine sentiment (GitHub repos are usually positive)
            sentiment = 'positive'
            
            # Calculate educational value
            educational_value = self._calculate_educational_value(content, repo)
            
            community_post = CommunityPost(
                source='github',
                content=content,
                author=repo.get('owner', {}).get('login', 'Unknown'),
                timestamp=datetime.now(),  # GitHub API timestamp handling
                engagement=repo.get('stargazers_count', 0),
                url=repo.get('html_url', ''),
                sentiment=sentiment,
                topics=topics,
                educational_value=educational_value
            )
            
            analyzed_posts.append(community_post)
        
        return analyzed_posts
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract relevant topics from content"""
        content_lower = content.lower()
        found_topics = []
        
        for keyword in self.ethereum_keywords:
            if keyword in content_lower:
                found_topics.append(keyword)
        
        # Add educational patterns
        for pattern_type, patterns in self.educational_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    found_topics.append(pattern_type)
        
        return list(set(found_topics))  # Remove duplicates
    
    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'awesome', 'excellent', 'amazing', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'problem', 'issue']
        
        content_lower = content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_educational_value(self, content: str, metadata: Dict[str, Any]) -> int:
        """Calculate educational value score (1-100)"""
        score = 0
        
        # Content length
        if len(content) > 100:
            score += 20
        
        # Engagement
        engagement = metadata.get('score', metadata.get('like_count', metadata.get('stargazers_count', 0)))
        if engagement > 100:
            score += 25
        elif engagement > 50:
            score += 15
        elif engagement > 10:
            score += 10
        
        # Educational patterns
        for pattern_type, patterns in self.educational_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content.lower()):
                    score += 15
                    break
        
        # Ethereum keywords
        keyword_count = sum(1 for keyword in self.ethereum_keywords if keyword in content.lower())
        score += min(keyword_count * 5, 20)
        
        return min(score, 100)

class TrendingConceptAnalyzer:
    """Analyzes community content to identify trending concepts"""
    
    def __init__(self):
        self.concept_mapping = {
            'defi': ['defi', 'yield', 'liquidity', 'amm', 'flash loan'],
            'nft': ['nft', 'non-fungible', 'opensea', 'minting'],
            'governance': ['dao', 'governance', 'proposal', 'voting'],
            'layer2': ['layer 2', 'rollup', 'optimism', 'arbitrum', 'polygon'],
            'staking': ['staking', 'validator', 'consensus', 'proof of stake'],
            'smart_contracts': ['smart contract', 'solidity', 'vyper', 'development'],
            'security': ['security', 'audit', 'hack', 'vulnerability', 'exploit'],
            'bridges': ['bridge', 'cross-chain', 'multichain', 'interoperability']
        }
    
    def identify_trending_concepts(self, posts: List[CommunityPost], 
                                 time_window_hours: int = 24) -> List[TrendingConcept]:
        """Identify trending concepts from community posts"""
        
        # Filter posts by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_posts = [post for post in posts if post.timestamp > cutoff_time]
        
        # Count concept frequencies
        concept_counts = Counter()
        concept_sources = {}
        concept_sentiments = {}
        concept_engagements = {}
        
        for post in recent_posts:
            for concept, keywords in self.concept_mapping.items():
                if any(keyword in post.content.lower() for keyword in keywords):
                    concept_counts[concept] += 1
                    
                    # Track sources
                    if concept not in concept_sources:
                        concept_sources[concept] = set()
                    concept_sources[concept].add(post.source)
                    
                    # Track sentiment
                    if concept not in concept_sentiments:
                        concept_sentiments[concept] = []
                    concept_sentiments[concept].append(post.sentiment)
                    
                    # Track engagement
                    if concept not in concept_engagements:
                        concept_engagements[concept] = []
                    concept_engagements[concept].append(post.educational_value)
        
        # Create trending concepts
        trending_concepts = []
        
        for concept, frequency in concept_counts.most_common(10):  # Top 10
            if frequency >= 2:  # Minimum frequency threshold
                
                # Calculate sentiment score
                sentiments = concept_sentiments.get(concept, [])
                sentiment_score = self._calculate_sentiment_score(sentiments)
                
                # Calculate engagement score
                engagements = concept_engagements.get(concept, [])
                engagement_score = sum(engagements) if engagements else 0
                
                # Generate educational opportunity
                educational_opportunity = self._generate_educational_opportunity(concept)
                
                # Determine difficulty level
                difficulty_level = self._determine_difficulty_level(concept)
                
                # Estimate learning time
                estimated_learning_time = self._estimate_learning_time(concept, difficulty_level)
                
                trending_concept = TrendingConcept(
                    concept=concept.replace('_', ' ').title(),
                    frequency=frequency,
                    sources=list(concept_sources.get(concept, [])),
                    sentiment_score=sentiment_score,
                    engagement_score=engagement_score,
                    educational_opportunity=educational_opportunity,
                    related_topics=self._get_related_topics(concept),
                    difficulty_level=difficulty_level,
                    estimated_learning_time=estimated_learning_time
                )
                
                trending_concepts.append(trending_concept)
        
        return trending_concepts
    
    def _calculate_sentiment_score(self, sentiments: List[str]) -> float:
        """Calculate sentiment score from list of sentiments"""
        if not sentiments:
            return 0.0
        
        sentiment_values = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        
        total_score = sum(sentiment_values.get(sentiment, 0) for sentiment in sentiments)
        return total_score / len(sentiments)
    
    def _generate_educational_opportunity(self, concept: str) -> str:
        """Generate educational opportunity description"""
        
        opportunities = {
            'defi': 'Learn about decentralized finance protocols and yield strategies',
            'nft': 'Understand non-fungible tokens and digital asset ownership',
            'governance': 'Explore DAO governance and community decision-making',
            'layer2': 'Master Layer 2 scaling solutions and rollup technology',
            'staking': 'Learn about proof-of-stake consensus and validator economics',
            'smart_contracts': 'Develop smart contracts and understand blockchain programming',
            'security': 'Master DeFi security best practices and risk management',
            'bridges': 'Understand cross-chain interoperability and bridge mechanics'
        }
        
        return opportunities.get(concept, 'Explore this emerging blockchain concept')
    
    def _determine_difficulty_level(self, concept: str) -> str:
        """Determine difficulty level for a concept"""
        
        difficulty_mapping = {
            'defi': 'intermediate',
            'nft': 'beginner',
            'governance': 'intermediate',
            'layer2': 'advanced',
            'staking': 'intermediate',
            'smart_contracts': 'expert',
            'security': 'advanced',
            'bridges': 'advanced'
        }
        
        return difficulty_mapping.get(concept, 'intermediate')
    
    def _estimate_learning_time(self, concept: str, difficulty: str) -> int:
        """Estimate learning time in minutes"""
        
        base_times = {
            'beginner': 30,
            'intermediate': 60,
            'advanced': 120,
            'expert': 240
        }
        
        # Adjust based on concept complexity
        complexity_multipliers = {
            'defi': 1.5,
            'nft': 0.8,
            'governance': 1.2,
            'layer2': 2.0,
            'staking': 1.0,
            'smart_contracts': 2.5,
            'security': 1.8,
            'bridges': 1.6
        }
        
        base_time = base_times.get(difficulty, 60)
        multiplier = complexity_multipliers.get(concept, 1.0)
        
        return int(base_time * multiplier)
    
    def _get_related_topics(self, concept: str) -> List[str]:
        """Get related topics for a concept"""
        
        related_topics = {
            'defi': ['Yield Farming', 'Liquidity Mining', 'AMMs', 'Flash Loans'],
            'nft': ['Digital Art', 'Gaming', 'Collectibles', 'Metaverse'],
            'governance': ['Voting Mechanisms', 'Proposal Systems', 'Token Economics'],
            'layer2': ['Rollups', 'State Channels', 'Sidechains', 'Plasma'],
            'staking': ['Validator Economics', 'Consensus Mechanisms', 'Rewards'],
            'smart_contracts': ['Solidity', 'Vyper', 'Development Tools', 'Testing'],
            'security': ['Auditing', 'Bug Bounties', 'Risk Assessment', 'Incident Response'],
            'bridges': ['Cross-chain Communication', 'Interoperability', 'Relay Networks']
        }
        
        return related_topics.get(concept, ['Blockchain', 'Cryptocurrency', 'Technology'])

class CommunityIntelligenceEngine:
    """Main engine that orchestrates community intelligence analysis"""
    
    def __init__(self):
        self.source_analyzer = CommunitySourceAnalyzer()
        self.trending_analyzer = TrendingConceptAnalyzer()
    
    def analyze_community_intelligence(self, 
                                     reddit_posts: List[Dict[str, Any]] = None,
                                     twitter_tweets: List[Dict[str, Any]] = None,
                                     github_repos: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze community intelligence from multiple sources"""
        
        all_posts = []
        
        # Analyze Reddit content
        if reddit_posts:
            reddit_analyzed = self.source_analyzer.analyze_reddit_content(reddit_posts)
            all_posts.extend(reddit_analyzed)
        
        # Analyze Twitter content
        if twitter_tweets:
            twitter_analyzed = self.source_analyzer.analyze_twitter_content(twitter_tweets)
            all_posts.extend(twitter_analyzed)
        
        # Analyze GitHub content
        if github_repos:
            github_analyzed = self.source_analyzer.analyze_github_content(github_repos)
            all_posts.extend(github_analyzed)
        
        # Identify trending concepts
        trending_concepts = self.trending_analyzer.identify_trending_concepts(all_posts)
        
        # Generate summary statistics
        summary = self._generate_summary(all_posts, trending_concepts)
        
        return {
            'summary': summary,
            'trending_concepts': [self._trending_concept_to_dict(tc) for tc in trending_concepts],
            'source_breakdown': self._analyze_source_breakdown(all_posts),
            'educational_opportunities': self._identify_educational_opportunities(all_posts),
            'last_updated': datetime.now().isoformat()
        }
    
    def _generate_summary(self, posts: List[CommunityPost], 
                         trending_concepts: List[TrendingConcept]) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        if not posts:
            return {
                'total_posts': 0,
                'total_engagement': 0,
                'average_educational_value': 0,
                'top_sources': [],
                'sentiment_distribution': {}
            }
        
        total_engagement = sum(post.engagement for post in posts)
        avg_educational_value = sum(post.educational_value for post in posts) / len(posts)
        
        # Source distribution
        source_counts = Counter(post.source for post in posts)
        top_sources = source_counts.most_common(3)
        
        # Sentiment distribution
        sentiment_counts = Counter(post.sentiment for post in posts)
        
        return {
            'total_posts': len(posts),
            'total_engagement': total_engagement,
            'average_educational_value': round(avg_educational_value, 2),
            'top_sources': [{'source': source, 'count': count} for source, count in top_sources],
            'sentiment_distribution': dict(sentiment_counts)
        }
    
    def _analyze_source_breakdown(self, posts: List[CommunityPost]) -> Dict[str, Any]:
        """Analyze breakdown by source"""
        
        source_data = {}
        
        for post in posts:
            if post.source not in source_data:
                source_data[post.source] = {
                    'count': 0,
                    'total_engagement': 0,
                    'avg_educational_value': 0,
                    'top_topics': Counter()
                }
            
            source_data[post.source]['count'] += 1
            source_data[post.source]['total_engagement'] += post.engagement
            source_data[post.source]['top_topics'].update(post.topics)
        
        # Calculate averages
        for source in source_data:
            posts_for_source = [p for p in posts if p.source == source]
            if posts_for_source:
                source_data[source]['avg_educational_value'] = sum(p.educational_value for p in posts_for_source) / len(posts_for_source)
                source_data[source]['top_topics'] = dict(source_data[source]['top_topics'].most_common(5))
        
        return source_data
    
    def _identify_educational_opportunities(self, posts: List[CommunityPost]) -> List[Dict[str, Any]]:
        """Identify high-value educational opportunities"""
        
        # Sort by educational value and engagement
        high_value_posts = sorted(posts, 
                                key=lambda x: (x.educational_value, x.engagement), 
                                reverse=True)[:10]
        
        opportunities = []
        
        for post in high_value_posts:
            opportunity = {
                'source': post.source,
                'content_preview': post.content[:200] + '...' if len(post.content) > 200 else post.content,
                'author': post.author,
                'educational_value': post.educational_value,
                'engagement': post.engagement,
                'topics': post.topics,
                'url': post.url,
                'sentiment': post.sentiment
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def _trending_concept_to_dict(self, concept: TrendingConcept) -> Dict[str, Any]:
        """Convert TrendingConcept to dictionary for JSON serialization"""
        return {
            'concept': concept.concept,
            'frequency': concept.frequency,
            'sources': concept.sources,
            'sentiment_score': concept.sentiment_score,
            'engagement_score': concept.engagement_score,
            'educational_opportunity': concept.educational_opportunity,
            'related_topics': concept.related_topics,
            'difficulty_level': concept.difficulty_level,
            'estimated_learning_time': concept.estimated_learning_time
        }

# Example usage
if __name__ == "__main__":
    engine = CommunityIntelligenceEngine()
    
    # Sample data
    sample_reddit_posts = [
        {
            'title': 'How do I understand DeFi yield farming?',
            'selftext': 'I\'m new to DeFi and want to learn about yield farming strategies. Can someone explain the basics?',
            'author': 'crypto_learner',
            'created_utc': time.time(),
            'score': 45
        },
        {
            'title': 'Ethereum Layer 2 solutions explained',
            'selftext': 'Great breakdown of how rollups work and why they\'re important for scaling Ethereum.',
            'author': 'eth_expert',
            'created_utc': time.time(),
            'score': 128
        }
    ]
    
    # Analyze community intelligence
    results = engine.analyze_community_intelligence(reddit_posts=sample_reddit_posts)
    
    print("Community Intelligence Analysis Results:")
    print(json.dumps(results, indent=2, default=str))
