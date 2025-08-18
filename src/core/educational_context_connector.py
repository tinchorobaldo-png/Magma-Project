#!/usr/bin/env python3
"""
EDUCATIONAL CONTEXT CONNECTOR - Connect market events to learning opportunities
Transforms market intelligence into educational experiences for progressive learning
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"

class ConceptType(Enum):
    PRICE_MOVEMENT = "price_movement"
    GAS_OPTIMIZATION = "gas_optimization"
    DEFI_PROTOCOLS = "defi_protocols"
    MARKET_SENTIMENT = "market_sentiment"
    YIELD_FARMING = "yield_farming"
    SOCIAL_SIGNALS = "social_signals"
    WHALE_ACTIVITY = "whale_activity"
    NEWS_IMPACT = "news_impact"

@dataclass
class LearningContext:
    """Educational context for a market event"""
    event_description: str
    concept_type: ConceptType
    learning_level: LearningLevel
    why_this_matters: str
    key_concepts: List[str]
    tutorial_content: str
    practical_example: str
    next_steps: List[str]
    related_concepts: List[str]
    difficulty_score: int  # 1-10
    estimated_time_minutes: int

@dataclass
class EducationalInsight:
    """Educational insight from market intelligence"""
    title: str
    context: str
    learning_opportunity: str
    concept_to_learn: str
    tutorial_link: str
    difficulty: LearningLevel
    confidence: float

class EducationalContextConnector:
    """Connect market events to educational opportunities"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Educational content database
        self.concept_explanations = {
            ConceptType.PRICE_MOVEMENT: {
                LearningLevel.BEGINNER: {
                    "title": "Understanding ETH Price Changes",
                    "explanation": "ETH price moves based on supply and demand. When more people want to buy ETH than sell it, the price goes up.",
                    "key_concepts": ["Supply and Demand", "Market Cap", "Trading Volume"],
                    "tutorial": "Price movements happen when market sentiment changes. Learn to read price charts and understand market psychology.",
                    "practical_example": "If ETH drops 5%, it could be a buying opportunity if fundamentals remain strong.",
                    "next_steps": ["Learn about market cap", "Understand trading volume", "Study price charts"]
                },
                LearningLevel.INTERMEDIATE: {
                    "title": "Price Movement Analysis", 
                    "explanation": "ETH price movements are influenced by multiple factors including network activity, DeFi usage, and macro economic conditions.",
                    "key_concepts": ["Technical Analysis", "On-chain Metrics", "Market Correlation"],
                    "tutorial": "Advanced price analysis involves studying on-chain metrics, DeFi TVL, and correlation with traditional markets.",
                    "practical_example": "Rising DeFi TVL often correlates with ETH price increases due to increased utility demand.",
                    "next_steps": ["Study on-chain analytics", "Learn DeFi impact", "Understand macro correlations"]
                },
                LearningLevel.ADVANCED: {
                    "title": "Sophisticated Price Dynamics",
                    "explanation": "ETH pricing involves complex interactions between spot markets, derivatives, MEV, and institutional flows.",
                    "key_concepts": ["Derivatives Impact", "MEV Effects", "Institutional Flows"],
                    "tutorial": "Professional price analysis considers futures premiums, options flow, MEV extraction, and institutional accumulation patterns.",
                    "practical_example": "High futures premiums often indicate leveraged speculation, which can lead to liquidation cascades.",
                    "next_steps": ["Analyze derivatives markets", "Study MEV impact", "Track institutional flows"]
                }
            },
            ConceptType.GAS_OPTIMIZATION: {
                LearningLevel.BEGINNER: {
                    "title": "Understanding Gas Fees",
                    "explanation": "Gas fees are the cost to execute transactions on Ethereum. Lower gas = cheaper transactions.",
                    "key_concepts": ["Gas Price", "Gas Limit", "Transaction Cost"],
                    "tutorial": "Gas fees change based on network congestion. Learn when to transact for optimal costs.",
                    "practical_example": "Wait for gas fees below 30 gwei for non-urgent transactions to save money.",
                    "next_steps": ["Monitor gas prices", "Learn gas estimation", "Time your transactions"]
                },
                LearningLevel.INTERMEDIATE: {
                    "title": "Gas Optimization Strategies",
                    "explanation": "Advanced gas optimization involves understanding EIP-1559, layer 2 solutions, and transaction batching.",
                    "key_concepts": ["EIP-1559", "Layer 2", "Transaction Batching"],
                    "tutorial": "Use layer 2 solutions, batch transactions, and understand priority fees for optimal gas usage.",
                    "practical_example": "Use Arbitrum or Polygon for frequent DeFi interactions to reduce gas costs by 90%+.",
                    "next_steps": ["Explore Layer 2s", "Learn transaction batching", "Understand MEV protection"]
                },
                LearningLevel.ADVANCED: {
                    "title": "Professional Gas Management",
                    "explanation": "Expert gas optimization involves MEV protection, flashloan arbitrage, and automated gas strategies.",
                    "key_concepts": ["MEV Protection", "Gas Auctions", "Automated Strategies"],
                    "tutorial": "Implement automated gas bidding, MEV protection, and sophisticated transaction ordering strategies.",
                    "practical_example": "Use private mempools and MEV protection services for large transactions to avoid front-running.",
                    "next_steps": ["Implement MEV protection", "Use private mempools", "Automate gas strategies"]
                }
            },
            ConceptType.DEFI_PROTOCOLS: {
                LearningLevel.BEGINNER: {
                    "title": "DeFi Basics",
                    "explanation": "DeFi (Decentralized Finance) allows you to earn yield, trade, and lend without traditional banks.",
                    "key_concepts": ["Lending", "Trading", "Yield Farming"],
                    "tutorial": "Start with simple lending on Aave or trading on Uniswap to understand DeFi basics.",
                    "practical_example": "Lend USDC on Aave to earn 3-5% APY with minimal risk.",
                    "next_steps": ["Try simple lending", "Learn about AMMs", "Understand yield farming"]
                },
                LearningLevel.INTERMEDIATE: {
                    "title": "DeFi Protocol Analysis",
                    "explanation": "Advanced DeFi involves understanding protocol risks, yield optimization, and liquidity provision strategies.",
                    "key_concepts": ["Protocol Risk", "Impermanent Loss", "Yield Strategies"],
                    "tutorial": "Analyze protocol tokenomics, understand impermanent loss, and develop yield strategies.",
                    "practical_example": "Provide liquidity to ETH/USDC on Uniswap V3 with active range management.",
                    "next_steps": ["Study protocol risks", "Learn LP strategies", "Understand tokenomics"]
                },
                LearningLevel.ADVANCED: {
                    "title": "Advanced DeFi Strategies",
                    "explanation": "Expert DeFi involves complex strategies like leveraged farming, cross-protocol arbitrage, and MEV extraction.",
                    "key_concepts": ["Leveraged Farming", "Cross-Protocol Arbitrage", "MEV Strategies"],
                    "tutorial": "Implement leveraged yield farming, cross-protocol arbitrage, and MEV extraction strategies.",
                    "practical_example": "Use recursive lending strategies to amplify yield while managing liquidation risks.",
                    "next_steps": ["Implement leverage strategies", "Build arbitrage bots", "Extract MEV opportunities"]
                }
            }
        }
    
    async def analyze_market_intelligence(self, market_data: Dict[str, Any], 
                                        user_level: LearningLevel = LearningLevel.INTERMEDIATE) -> List[EducationalInsight]:
        """Analyze market intelligence and generate educational insights"""
        try:
            insights = []
            
            # Price movement insights
            if 'current_price' in market_data and 'price_change_24h' in market_data:
                price_insight = await self._analyze_price_movement(market_data, user_level)
                if price_insight:
                    insights.append(price_insight)
            
            # Volume insights  
            if 'volume_24h' in market_data:
                volume_insight = await self._analyze_volume_patterns(market_data, user_level)
                if volume_insight:
                    insights.append(volume_insight)
            
            # Market cap insights
            if 'market_cap' in market_data:
                market_cap_insight = await self._analyze_market_cap(market_data, user_level)
                if market_cap_insight:
                    insights.append(market_cap_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing market intelligence: {e}")
            return []
    
    async def _analyze_price_movement(self, market_data: Dict[str, Any], 
                                    user_level: LearningLevel) -> Optional[EducationalInsight]:
        """Analyze price movement and generate educational insight"""
        try:
            price_change = market_data.get('price_change_24h', 0)
            current_price = market_data.get('current_price', 0)
            
            if abs(price_change) < 1:
                return None  # Skip small movements
            
            # Determine context based on price change
            if price_change > 5:
                context = f"ETH surged {price_change:.1f}% to ${current_price:,.2f} in 24 hours"
                learning_opportunity = "This is a great time to learn about what drives ETH price increases"
            elif price_change < -5:
                context = f"ETH declined {abs(price_change):.1f}% to ${current_price:,.2f} in 24 hours"
                learning_opportunity = "This is an opportunity to understand market corrections and potential buying opportunities"
            else:
                context = f"ETH moved {price_change:+.1f}% to ${current_price:,.2f} in 24 hours"
                learning_opportunity = "This moderate movement is perfect for learning about market dynamics"
            
            # Get educational content based on user level
            concept_data = self.concept_explanations[ConceptType.PRICE_MOVEMENT][user_level]
            
            return EducationalInsight(
                title=f"Today's Price Movement: {price_change:+.1f}%",
                context=context,
                learning_opportunity=learning_opportunity,
                concept_to_learn=concept_data["title"],
                tutorial_link=concept_data["tutorial"],
                difficulty=user_level,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Error analyzing price movement: {e}")
            return None
    
    async def _analyze_volume_patterns(self, market_data: Dict[str, Any], 
                                     user_level: LearningLevel) -> Optional[EducationalInsight]:
        """Analyze volume patterns and generate educational insight"""
        try:
            volume_24h = market_data.get('volume_24h', 0)
            market_cap = market_data.get('market_cap', 1)
            
            volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
            
            if volume_ratio > 0.15:
                context = f"ETH trading volume is very high at ${volume_24h:,.0f}"
                learning_opportunity = "High volume often indicates strong conviction in price movements"
                concept_to_learn = "Volume Analysis and Market Conviction"
            elif volume_ratio < 0.05:
                context = f"ETH trading volume is low at ${volume_24h:,.0f}" 
                learning_opportunity = "Low volume can indicate market uncertainty or consolidation"
                concept_to_learn = "Understanding Low Volume Markets"
            else:
                return None  # Skip normal volume
            
            return EducationalInsight(
                title="Volume Analysis Opportunity",
                context=context,
                learning_opportunity=learning_opportunity,
                concept_to_learn=concept_to_learn,
                tutorial_link="Learn how trading volume indicates market strength and conviction",
                difficulty=user_level,
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {e}")
            return None
    
    async def _analyze_market_cap(self, market_data: Dict[str, Any], 
                                user_level: LearningLevel) -> Optional[EducationalInsight]:
        """Analyze market cap and generate educational insight"""
        try:
            market_cap = market_data.get('market_cap', 0)
            dominance = market_data.get('dominance', 0)
            
            if dominance > 0.20:  # ETH dominance above 20%
                context = f"ETH market dominance is strong at {dominance*100:.1f}%"
                learning_opportunity = "High ETH dominance indicates strong position in crypto ecosystem"
                concept_to_learn = "Market Dominance and Ecosystem Value"
            elif dominance < 0.15:  # ETH dominance below 15%
                context = f"ETH market dominance is at {dominance*100:.1f}%"
                learning_opportunity = "Changes in dominance reflect ETH's competitive position"
                concept_to_learn = "Competitive Analysis in Crypto Markets"
            else:
                return None  # Skip normal dominance
            
            return EducationalInsight(
                title="Market Dominance Analysis",
                context=context,
                learning_opportunity=learning_opportunity,
                concept_to_learn=concept_to_learn,
                tutorial_link="Understand how market dominance reflects ecosystem strength",
                difficulty=user_level,
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market cap: {e}")
            return None
    
    async def analyze_gas_intelligence(self, gas_data: Dict[str, Any], 
                                     user_level: LearningLevel = LearningLevel.INTERMEDIATE) -> List[EducationalInsight]:
        """Analyze gas intelligence and generate educational insights"""
        try:
            insights = []
            
            current_gas = gas_data.get('current_gas', 25)
            trend = gas_data.get('trend', 'stable')
            status = gas_data.get('status', 'Normal')
            
            # Gas optimization opportunity
            gas_insight = await self._analyze_gas_opportunity(current_gas, trend, status, user_level)
            if gas_insight:
                insights.append(gas_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing gas intelligence: {e}")
            return []
    
    async def _analyze_gas_opportunity(self, current_gas: float, trend: str, status: str, 
                                     user_level: LearningLevel) -> Optional[EducationalInsight]:
        """Analyze gas opportunity and generate educational insight"""
        try:
            if current_gas < 20:
                context = f"Gas fees are very low at {current_gas} gwei - excellent for transactions"
                learning_opportunity = "This is the perfect time to learn about gas optimization and DeFi interactions"
                concept_to_learn = "Gas Optimization Strategies"
            elif current_gas > 50:
                context = f"Gas fees are high at {current_gas} gwei - consider waiting or using Layer 2"
                learning_opportunity = "High gas periods are great for learning about Layer 2 solutions and transaction timing"
                concept_to_learn = "Layer 2 Solutions and Gas Management"
            elif trend == 'decreasing':
                context = f"Gas fees are decreasing (currently {current_gas} gwei) - good timing for transactions"
                learning_opportunity = "Learn to monitor gas trends and time your transactions optimally"
                concept_to_learn = "Gas Trend Analysis and Transaction Timing"
            else:
                return None  # Skip normal gas conditions
            
            # Get educational content based on user level
            concept_data = self.concept_explanations[ConceptType.GAS_OPTIMIZATION][user_level]
            
            return EducationalInsight(
                title=f"Gas Optimization Opportunity",
                context=context,
                learning_opportunity=learning_opportunity,
                concept_to_learn=concept_data["title"],
                tutorial_link=concept_data["tutorial"],
                difficulty=user_level,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Error analyzing gas opportunity: {e}")
            return None
    
    async def analyze_social_intelligence(self, social_data: Dict[str, Any], 
                                        user_level: LearningLevel = LearningLevel.INTERMEDIATE) -> List[EducationalInsight]:
        """Analyze social intelligence and generate educational insights"""
        try:
            insights = []
            
            # Social sentiment insights
            if 'ethereum_metrics' in social_data:
                eth_metrics = social_data['ethereum_metrics']
                social_insight = await self._analyze_social_sentiment(eth_metrics, user_level)
                if social_insight:
                    insights.append(social_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing social intelligence: {e}")
            return []
    
    async def _analyze_social_sentiment(self, eth_metrics: Dict[str, Any], 
                                      user_level: LearningLevel) -> Optional[EducationalInsight]:
        """Analyze social sentiment and generate educational insight"""
        try:
            social_score = eth_metrics.get('social_score', 50)
            social_volume = eth_metrics.get('social_volume', 0)
            
            if social_score > 80:
                context = f"ETH social sentiment is very bullish (score: {social_score})"
                learning_opportunity = "High social sentiment can be a leading indicator - learn to interpret social signals"
                concept_to_learn = "Social Sentiment Analysis and Market Psychology"
            elif social_score < 30:
                context = f"ETH social sentiment is bearish (score: {social_score})"
                learning_opportunity = "Low sentiment often presents contrarian opportunities - learn about market psychology"
                concept_to_learn = "Contrarian Analysis and Market Sentiment"
            else:
                return None  # Skip neutral sentiment
            
            return EducationalInsight(
                title="Social Sentiment Analysis",
                context=context,
                learning_opportunity=learning_opportunity,
                concept_to_learn=concept_to_learn,
                tutorial_link="Learn how social sentiment can predict market movements",
                difficulty=user_level,
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return None
    
    async def analyze_news_intelligence(self, news_data: Dict[str, Any], 
                                      user_level: LearningLevel = LearningLevel.INTERMEDIATE) -> List[EducationalInsight]:
        """Analyze news intelligence and generate educational insights"""
        try:
            insights = []
            
            # Overall sentiment from news
            overall_sentiment = news_data.get('overall_sentiment', 'neutral')
            total_articles = news_data.get('total_articles_24h', 0)
            key_themes = news_data.get('key_themes', [])
            
            if overall_sentiment in ['bullish', 'bearish'] and total_articles > 10:
                news_insight = await self._analyze_news_sentiment(overall_sentiment, key_themes, user_level)
                if news_insight:
                    insights.append(news_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing news intelligence: {e}")
            return []
    
    async def _analyze_news_sentiment(self, sentiment: str, themes: List[str], 
                                    user_level: LearningLevel) -> Optional[EducationalInsight]:
        """Analyze news sentiment and generate educational insight"""
        try:
            themes_str = ", ".join(themes[:3]) if themes else "general market news"
            
            if sentiment == 'bullish':
                context = f"News sentiment is bullish with focus on: {themes_str}"
                learning_opportunity = "Positive news often drives market momentum - learn to analyze news impact"
                concept_to_learn = "News Analysis and Market Impact"
            else:  # bearish
                context = f"News sentiment is bearish with focus on: {themes_str}"
                learning_opportunity = "Negative news can create opportunities - learn contrarian analysis"
                concept_to_learn = "Contrarian Analysis and News Interpretation"
            
            return EducationalInsight(
                title="News Impact Analysis",
                context=context,
                learning_opportunity=learning_opportunity,
                concept_to_learn=concept_to_learn,
                tutorial_link="Learn how to analyze news impact on crypto markets",
                difficulty=user_level,
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return None
    
    async def generate_comprehensive_learning_context(self, intelligence_data: Dict[str, Any], 
                                                    user_level: LearningLevel = LearningLevel.INTERMEDIATE) -> Dict[str, Any]:
        """Generate comprehensive learning context from all intelligence data"""
        try:
            all_insights = []
            
            # Analyze each intelligence source
            if 'market_intelligence' in intelligence_data:
                market_insights = await self.analyze_market_intelligence(
                    intelligence_data['market_intelligence'], user_level
                )
                all_insights.extend(market_insights)
            
            if 'gas_intelligence' in intelligence_data:
                gas_insights = await self.analyze_gas_intelligence(
                    intelligence_data['gas_intelligence'], user_level
                )
                all_insights.extend(gas_insights)
            
            if 'social_intelligence' in intelligence_data:
                social_insights = await self.analyze_social_intelligence(
                    intelligence_data['social_intelligence'], user_level
                )
                all_insights.extend(social_insights)
            
            if 'news_intelligence' in intelligence_data:
                news_insights = await self.analyze_news_intelligence(
                    intelligence_data['news_intelligence'], user_level
                )
                all_insights.extend(news_insights)
            
            # Generate today's learning summary
            learning_summary = await self._generate_daily_learning_summary(all_insights, user_level)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'user_level': user_level.value,
                'total_insights': len(all_insights),
                'educational_insights': [insight.__dict__ for insight in all_insights],
                'daily_learning_summary': learning_summary,
                'recommended_actions': self._generate_recommended_actions(all_insights, user_level),
                'progress_opportunities': self._generate_progress_opportunities(all_insights, user_level)
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive learning context: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'educational_insights': []
            }
    
    async def _generate_daily_learning_summary(self, insights: List[EducationalInsight], 
                                             user_level: LearningLevel) -> str:
        """Generate daily learning summary from insights"""
        try:
            if not insights:
                return "No significant learning opportunities detected today. Check back later for market-driven educational content."
            
            summary_parts = []
            
            # Count insights by type
            price_insights = [i for i in insights if 'Price' in i.title]
            gas_insights = [i for i in insights if 'Gas' in i.title]
            social_insights = [i for i in insights if 'Social' in i.title]
            news_insights = [i for i in insights if 'News' in i.title]
            
            summary_parts.append(f"📚 Today's Learning Opportunities ({user_level.value.title()} Level):")
            
            if price_insights:
                summary_parts.append(f"💰 Price Analysis: {len(price_insights)} learning opportunities")
            
            if gas_insights:
                summary_parts.append(f"⛽ Gas Optimization: {len(gas_insights)} learning opportunities")
            
            if social_insights:
                summary_parts.append(f"🌙 Social Sentiment: {len(social_insights)} learning opportunities")
            
            if news_insights:
                summary_parts.append(f"📰 News Analysis: {len(news_insights)} learning opportunities")
            
            # Add most important insight
            top_insight = max(insights, key=lambda x: x.confidence)
            summary_parts.append(f"\n🎯 Priority Learning: {top_insight.concept_to_learn}")
            summary_parts.append(f"Why: {top_insight.learning_opportunity}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating daily learning summary: {e}")
            return "Learning summary temporarily unavailable."
    
    def _generate_recommended_actions(self, insights: List[EducationalInsight], 
                                    user_level: LearningLevel) -> List[str]:
        """Generate recommended actions based on insights"""
        try:
            actions = []
            
            if not insights:
                return ["Check back later for personalized learning recommendations"]
            
            # Group insights by concept type
            concept_groups = {}
            for insight in insights:
                concept = insight.concept_to_learn
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append(insight)
            
            # Generate actions for each concept
            for concept, concept_insights in concept_groups.items():
                confidence = max(insight.confidence for insight in concept_insights)
                if confidence > 0.7:
                    actions.append(f"📖 Study: {concept} (High Priority)")
                else:
                    actions.append(f"📚 Review: {concept} (Medium Priority)")
            
            # Add level-specific actions
            if user_level == LearningLevel.BEGINNER:
                actions.append("🎯 Focus on understanding basic concepts before advanced strategies")
            elif user_level == LearningLevel.INTERMEDIATE:
                actions.append("🚀 Start exploring practical applications of learned concepts")
            elif user_level == LearningLevel.ADVANCED:
                actions.append("💡 Consider implementing automated strategies based on learned patterns")
            
            return actions[:5]  # Limit to 5 actions
            
        except Exception as e:
            logger.error(f"Error generating recommended actions: {e}")
            return ["Recommended actions temporarily unavailable"]
    
    def _generate_progress_opportunities(self, insights: List[EducationalInsight], 
                                       user_level: LearningLevel) -> List[str]:
        """Generate progress opportunities for level advancement"""
        try:
            opportunities = []
            
            if not insights:
                return ["Complete basic tutorials to unlock progress opportunities"]
            
            # Current level opportunities
            if user_level == LearningLevel.BEGINNER:
                opportunities = [
                    "Complete 3 price analysis tutorials to advance to Intermediate",
                    "Practice gas optimization with small transactions",
                    "Learn to read basic market indicators"
                ]
            elif user_level == LearningLevel.INTERMEDIATE:
                opportunities = [
                    "Master DeFi protocol analysis to advance to Advanced",
                    "Implement a yield farming strategy",
                    "Analyze whale movement patterns"
                ]
            elif user_level == LearningLevel.ADVANCED:
                opportunities = [
                    "Develop automated trading strategies to reach Expert",
                    "Contribute to DeFi protocol analysis",
                    "Mentor beginners in the community"
                ]
            else:  # Expert
                opportunities = [
                    "Create educational content for the community",
                    "Develop new analysis tools",
                    "Lead advanced research initiatives"
                ]
            
            # Add insight-specific opportunities
            high_confidence_insights = [i for i in insights if i.confidence > 0.8]
            if high_confidence_insights:
                opportunities.append(f"Apply learning from today's {len(high_confidence_insights)} high-confidence insights")
            
            return opportunities[:4]  # Limit to 4 opportunities
            
        except Exception as e:
            logger.error(f"Error generating progress opportunities: {e}")
            return ["Progress opportunities temporarily unavailable"]

# Initialize the connector
def create_educational_connector() -> EducationalContextConnector:
    """Create and return an EducationalContextConnector instance"""
    return EducationalContextConnector()

# Test function
async def test_educational_connector():
    """Test the educational context connector"""
    connector = EducationalContextConnector()
    
    # Sample market data
    test_market_data = {
        'current_price': 4500.0,
        'price_change_24h': -3.2,
        'volume_24h': 25000000000,
        'market_cap': 500000000000,
        'dominance': 0.18
    }
    
    test_gas_data = {
        'current_gas': 15.5,
        'trend': 'decreasing',
        'status': 'Low'
    }
    
    test_intelligence = {
        'market_intelligence': test_market_data,
        'gas_intelligence': test_gas_data
    }
    
    # Generate learning context
    result = await connector.generate_comprehensive_learning_context(
        test_intelligence, LearningLevel.INTERMEDIATE
    )
    
    print("Educational Context Analysis:")
    print(f"Total Insights: {result['total_insights']}")
    print(f"Learning Summary: {result['daily_learning_summary']}")
    print(f"Recommended Actions: {result['recommended_actions']}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_educational_connector())
