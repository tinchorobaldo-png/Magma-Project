#!/usr/bin/env python3
"""
MARKET EVENT LEARNING ENGINE - Core Innovation for MAGMA
Automatically converts market events into structured learning opportunities
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import openai
import os

@dataclass
class MarketEvent:
    """Represents a market event that can be converted to learning"""
    event_type: str
    title: str
    description: str
    timestamp: datetime
    source: str
    impact_level: str  # low, medium, high, critical
    raw_data: Dict[str, Any]

@dataclass
class LearningOpportunity:
    """Represents a learning opportunity generated from a market event"""
    event: MarketEvent
    concept: str
    difficulty: str  # beginner, intermediate, advanced, expert
    why_learn: str
    key_insights: List[str]
    practical_examples: List[str]
    related_concepts: List[str]
    estimated_time: int  # minutes
    learning_path: List[str]

class MarketEventClassifier:
    """Classifies market events into educational categories"""
    
    def __init__(self):
        self.event_patterns = {
            'whale_movement': [
                r'whale.*moved',
                r'large.*transfer',
                r'\$[0-9]+[MBK].*transfer',
                r'address.*moved.*\$\d+'
            ],
            'protocol_update': [
                r'protocol.*update',
                r'upgrade.*announced',
                r'new.*version.*released',
                r'governance.*proposal'
            ],
            'hack_exploit': [
                r'hack',
                r'exploit',
                r'vulnerability',
                r'security.*breach',
                r'funds.*stolen'
            ],
            'governance_action': [
                r'governance.*vote',
                r'proposal.*passed',
                r'dao.*decision',
                r'community.*vote'
            ],
            'defi_event': [
                r'liquidation',
                r'flash.*loan',
                r'yield.*farming',
                r'amm.*update',
                r'lending.*protocol'
            ],
            'market_sentiment': [
                r'fear.*greed',
                r'social.*sentiment',
                r'community.*mood',
                r'market.*emotion'
            ]
        }
    
    def classify_event(self, title: str, description: str) -> str:
        """Classify an event based on its title and description"""
        combined_text = f"{title} {description}".lower()
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    return event_type
        
        return 'general_market_event'
    
    def calculate_impact_level(self, event_type: str, data: Dict[str, Any]) -> str:
        """Calculate the impact level of an event"""
        impact_factors = {
            'whale_movement': self._calculate_whale_impact,
            'protocol_update': self._calculate_protocol_impact,
            'hack_exploit': self._calculate_security_impact,
            'governance_action': self._calculate_governance_impact,
            'defi_event': self._calculate_defi_impact,
            'market_sentiment': self._calculate_sentiment_impact
        }
        
        if event_type in impact_factors:
            return impact_factors[event_type](data)
        
        return 'medium'
    
    def _calculate_whale_impact(self, data: Dict[str, Any]) -> str:
        """Calculate impact based on whale movement size"""
        # Extract amount from data
        amount_str = str(data.get('amount', '0'))
        amount = self._extract_amount(amount_str)
        
        if amount > 10000000:  # $10M+
            return 'critical'
        elif amount > 1000000:  # $1M+
            return 'high'
        elif amount > 100000:  # $100K+
            return 'medium'
        else:
            return 'low'
    
    def _calculate_protocol_impact(self, data: Dict[str, Any]) -> str:
        """Calculate impact based on protocol importance"""
        protocol = data.get('protocol', '').lower()
        
        major_protocols = ['uniswap', 'aave', 'compound', 'curve', 'makerdao']
        if any(p in protocol for p in major_protocols):
            return 'high'
        elif protocol:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_security_impact(self, data: Dict[str, Any]) -> str:
        """Calculate impact based on security event severity"""
        amount = self._extract_amount(str(data.get('amount', '0')))
        
        if amount > 10000000:  # $10M+
            return 'critical'
        elif amount > 1000000:  # $1M+
            return 'high'
        else:
            return 'medium'
    
    def _calculate_governance_impact(self, data: Dict[str, Any]) -> str:
        """Calculate impact based on governance event"""
        return 'medium'  # Most governance events are medium impact
    
    def _calculate_defi_impact(self, data: Dict[str, Any]) -> str:
        """Calculate impact based on DeFi event"""
        return 'medium'  # Most DeFi events are medium impact
    
    def _calculate_sentiment_impact(self, data: Dict[str, Any]) -> str:
        """Calculate impact based on sentiment change"""
        return 'low'  # Sentiment changes are usually low impact
    
    def _extract_amount(self, amount_str: str) -> float:
        """Extract numeric amount from string"""
        try:
            # Remove common prefixes and extract numbers
            cleaned = re.sub(r'[^\d.]', '', amount_str)
            return float(cleaned) if cleaned else 0
        except:
            return 0

class LearningOpportunityGenerator:
    """Generates learning opportunities from market events"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.concept_mapping = {
            'whale_movement': {
                'concept': 'Whale Behavior Analysis',
                'difficulty': 'intermediate',
                'why_learn': 'Understanding whale movements helps predict market trends and learn about market psychology',
                'key_insights': [
                    'Whale behavior patterns',
                    'Market impact of large transfers',
                    'Address clustering techniques',
                    'Risk assessment of whale movements'
                ],
                'related_concepts': ['Market Psychology', 'Address Analysis', 'Risk Management']
            },
            'protocol_update': {
                'concept': 'Protocol Governance & Updates',
                'difficulty': 'advanced',
                'why_learn': 'Protocol updates affect the entire DeFi ecosystem and create new opportunities',
                'key_insights': [
                    'Governance mechanisms',
                    'Upgrade processes',
                    'Risk assessment of changes',
                    'Opportunity identification'
                ],
                'related_concepts': ['Governance', 'Risk Management', 'Opportunity Analysis']
            },
            'hack_exploit': {
                'concept': 'DeFi Security & Risk Management',
                'difficulty': 'expert',
                'why_learn': 'Understanding security vulnerabilities helps protect assets and identify risks',
                'key_insights': [
                    'Common attack vectors',
                    'Security best practices',
                    'Risk assessment frameworks',
                    'Recovery strategies'
                ],
                'related_concepts': ['Security', 'Risk Management', 'Incident Response']
            },
            'governance_action': {
                'concept': 'DAO Governance & Decision Making',
                'difficulty': 'intermediate',
                'why_learn': 'Governance decisions shape the future of protocols and create learning opportunities',
                'key_insights': [
                    'Voting mechanisms',
                    'Proposal analysis',
                    'Stakeholder alignment',
                    'Decision impact assessment'
                ],
                'related_concepts': ['Governance', 'Decision Making', 'Stakeholder Analysis']
            },
            'defi_event': {
                'concept': 'DeFi Protocol Mechanics',
                'difficulty': 'intermediate',
                'why_learn': 'Understanding DeFi events helps optimize strategies and identify opportunities',
                'key_insights': [
                    'Protocol mechanics',
                    'Risk-reward analysis',
                    'Strategy optimization',
                    'Market timing'
                ],
                'related_concepts': ['DeFi', 'Strategy', 'Risk Management']
            },
            'market_sentiment': {
                'concept': 'Market Psychology & Sentiment',
                'difficulty': 'beginner',
                'why_learn': 'Sentiment analysis helps understand market cycles and make informed decisions',
                'key_insights': [
                    'Fear and greed cycles',
                    'Social sentiment indicators',
                    'Market psychology patterns',
                    'Contrarian strategies'
                ],
                'related_concepts': ['Psychology', 'Market Cycles', 'Strategy']
            }
        }
    
    def generate_learning_opportunity(self, event: MarketEvent) -> LearningOpportunity:
        """Generate a learning opportunity from a market event"""
        
        # Get base concept mapping
        base_concept = self.concept_mapping.get(event.event_type, {
            'concept': 'General Market Analysis',
            'difficulty': 'beginner',
            'why_learn': 'Understanding market events helps develop trading and investment skills',
            'key_insights': ['Market analysis', 'Event impact', 'Risk assessment'],
            'related_concepts': ['Market Analysis', 'Risk Management']
        })
        
        # Generate dynamic related concepts based on event content
        dynamic_related_concepts = self._generate_dynamic_related_concepts(event)
        
        # Generate practical examples using GPT if available
        practical_examples = self._generate_practical_examples(event, base_concept['concept'])
        
        # Create dynamic learning path based on event type and content
        learning_path = self._create_dynamic_learning_path(event)
        
        # Estimate learning time
        estimated_time = self._estimate_learning_time(event.event_type, base_concept['difficulty'])
        
        return LearningOpportunity(
            event=event,
            concept=base_concept['concept'],
            difficulty=base_concept['difficulty'],
            why_learn=base_concept['why_learn'],
            key_insights=base_concept['key_insights'],
            practical_examples=practical_examples,
            related_concepts=dynamic_related_concepts,  # Use dynamic concepts
            estimated_time=estimated_time,
            learning_path=learning_path  # Use dynamic learning path
        )
    
    def _generate_dynamic_related_concepts(self, event: MarketEvent) -> List[str]:
        """Generate related concepts dynamically based on event content"""
        
        # Extract keywords from event title and description
        text_content = f"{event.title} {event.description}".lower()
        
        # Define concept categories with keywords
        concept_categories = {
            'Technical Analysis': ['price', 'chart', 'pattern', 'support', 'resistance', 'trend', 'bull', 'bear'],
            'DeFi Protocols': ['uniswap', 'aave', 'compound', 'curve', 'defi', 'yield', 'liquidity', 'amm'],
            'Blockchain Technology': ['ethereum', 'eth', 'layer 2', 'rollup', 'sharding', 'pos', 'merge'],
            'Security & Risk': ['hack', 'exploit', 'vulnerability', 'security', 'risk', 'audit', 'breach'],
            'Market Psychology': ['whale', 'sentiment', 'fear', 'greed', 'community', 'social', 'emotion'],
            'Governance': ['dao', 'governance', 'proposal', 'vote', 'decision', 'stakeholder'],
            'Regulation': ['regulation', 'compliance', 'legal', 'government', 'policy', 'sec'],
            'Institutional': ['institutional', 'etf', 'adoption', 'enterprise', 'corporate', 'traditional'],
            'Innovation': ['innovation', 'research', 'development', 'upgrade', 'new', 'feature', 'technology'],
            'Economics': ['tokenomics', 'supply', 'demand', 'inflation', 'deflation', 'monetary', 'fiscal']
        }
        
        # Find matching concepts based on content
        matched_concepts = []
        for concept, keywords in concept_categories.items():
            if any(keyword in text_content for keyword in keywords):
                matched_concepts.append(concept)
        
        # Add event-specific concepts based on event type
        event_specific_concepts = {
            'whale_movement': ['Address Analysis', 'Transaction Tracking', 'Market Impact'],
            'protocol_update': ['Smart Contracts', 'Protocol Mechanics', 'Upgrade Process'],
            'hack_exploit': ['Security Auditing', 'Incident Response', 'Risk Assessment'],
            'governance_action': ['Voting Mechanisms', 'Stakeholder Analysis', 'Decision Impact'],
            'defi_event': ['Yield Optimization', 'Liquidity Management', 'Strategy Development'],
            'market_sentiment': ['Social Analysis', 'Market Psychology', 'Trend Analysis']
        }
        
        if event.event_type in event_specific_concepts:
            matched_concepts.extend(event_specific_concepts[event.event_type])
        
        # Ensure we have at least 3 concepts, add general ones if needed
        general_concepts = ['Market Analysis', 'Risk Management', 'Blockchain Fundamentals']
        while len(matched_concepts) < 3:
            for concept in general_concepts:
                if concept not in matched_concepts:
                    matched_concepts.append(concept)
                    break
        
        # Return top 4-5 most relevant concepts
        return matched_concepts[:5]
    
    def _create_dynamic_learning_path(self, event: MarketEvent) -> List[str]:
        """Create a dynamic learning path based on event content and type"""
        
        # Base learning paths for different event types
        base_paths = {
            'whale_movement': [
                "Learn about blockchain address analysis",
                "Study transaction tracking tools",
                "Understand market impact of large movements",
                "Practice analyzing similar patterns"
            ],
            'protocol_update': [
                "Review the protocol's current state",
                "Study the proposed changes",
                "Analyze potential impact on users",
                "Practice with testnet implementations"
            ],
            'hack_exploit': [
                "Study the security vulnerability",
                "Learn about similar attack vectors",
                "Understand risk mitigation strategies",
                "Practice security best practices"
            ],
            'governance_action': [
                "Review the governance proposal",
                "Study voting mechanisms",
                "Analyze stakeholder interests",
                "Practice decision-making frameworks"
            ],
            'defi_event': [
                "Understand the protocol mechanics",
                "Study risk-reward dynamics",
                "Learn optimization strategies",
                "Practice with simulation tools"
            ],
            'market_sentiment': [
                "Learn sentiment analysis tools",
                "Study market psychology patterns",
                "Understand contrarian strategies",
                "Practice emotional discipline"
            ]
        }
        
        # Get base path for event type
        base_path = base_paths.get(event.event_type, [
            "Understand the basic concept",
            "Study real-world examples",
            "Practice with tools and simulations",
            "Apply knowledge to current events"
        ])
        
        # Customize path based on event content
        text_content = f"{event.title} {event.description}".lower()
        
        # Add specific learning steps based on content
        specific_steps = []
        
        if 'ethereum' in text_content or 'eth' in text_content:
            specific_steps.append("Study Ethereum fundamentals and ecosystem")
        
        if 'defi' in text_content or 'yield' in text_content:
            specific_steps.append("Learn DeFi protocol mechanics and risks")
        
        if 'whale' in text_content or 'large' in text_content:
            specific_steps.append("Practice whale movement analysis techniques")
        
        if 'security' in text_content or 'hack' in text_content:
            specific_steps.append("Study security best practices and tools")
        
        if 'governance' in text_content or 'dao' in text_content:
            specific_steps.append("Learn about DAO governance structures")
        
        # Combine base path with specific steps
        final_path = base_path[:2] + specific_steps + base_path[2:]
        
        # Ensure path is not too long and has good flow
        if len(final_path) > 6:
            final_path = final_path[:6]
        
        return final_path
    
    def _generate_practical_examples(self, event: MarketEvent, concept: str) -> List[str]:
        """Generate practical examples using GPT or fallback to predefined ones"""
        
        if not self.openai_api_key:
            # Fallback examples
            fallback_examples = {
                'Whale Behavior Analysis': [
                    f"Analyze the {event.title} to understand whale behavior patterns",
                    "Use Etherscan to track similar movements",
                    "Compare with historical whale activity"
                ],
                'Protocol Governance & Updates': [
                    f"Review the {event.title} governance proposal",
                    "Analyze voting patterns and stakeholder alignment",
                    "Assess potential impact on protocol users"
                ],
                'DeFi Security & Risk Management': [
                    f"Study the {event.title} security incident",
                    "Identify the root cause and attack vector",
                    "Learn prevention strategies for similar attacks"
                ]
            }
            return fallback_examples.get(concept, [
                f"Analyze the {event.title} event",
                "Research similar historical events",
                "Apply learned concepts to current market conditions"
            ])
        
        # Use GPT for dynamic examples
        try:
            prompt = f"""
            Generate 3 practical learning examples for understanding this market event:
            Event: {event.title}
            Concept: {concept}
            
            Examples should be actionable and educational. Format as a simple list.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            examples_text = response.choices[0].message.content
            # Parse examples from response
            examples = [ex.strip() for ex in examples_text.split('\n') if ex.strip()]
            return examples[:3]  # Return max 3 examples
            
        except Exception as e:
            # Fallback to predefined examples
            return [
                f"Analyze the {event.title} event",
                "Research similar historical events",
                "Apply learned concepts to current market conditions"
            ]
    

    
    def _estimate_learning_time(self, event_type: str, difficulty: str) -> int:
        """Estimate learning time in minutes"""
        
        time_estimates = {
            'beginner': 30,
            'intermediate': 60,
            'advanced': 120,
            'expert': 240
        }
        
        # Adjust based on event type complexity
        complexity_multipliers = {
            'whale_movement': 1.0,
            'protocol_update': 1.5,
            'hack_exploit': 2.0,
            'governance_action': 1.2,
            'defi_event': 1.3,
            'market_sentiment': 0.8
        }
        
        base_time = time_estimates.get(difficulty, 60)
        multiplier = complexity_multipliers.get(event_type, 1.0)
        
        return int(base_time * multiplier)

class MarketEventLearningEngine:
    """Main engine that orchestrates the entire process"""
    
    def __init__(self):
        self.classifier = MarketEventClassifier()
        self.generator = LearningOpportunityGenerator()
    
    def process_market_event(self, title: str, description: str, source: str, 
                           raw_data: Dict[str, Any]) -> LearningOpportunity:
        """Process a market event and convert it to a learning opportunity"""
        
        # Create market event
        event = MarketEvent(
            event_type=self.classifier.classify_event(title, description),
            title=title,
            description=description,
            timestamp=datetime.now(),
            source=source,
            impact_level=self.classifier.calculate_impact_level(
                self.classifier.classify_event(title, description), 
                raw_data
            ),
            raw_data=raw_data
        )
        
        # Generate learning opportunity
        learning_opportunity = self.generator.generate_learning_opportunity(event)
        
        return learning_opportunity
    
    def process_multiple_events(self, events: List[Dict[str, Any]]) -> List[LearningOpportunity]:
        """Process multiple market events"""
        opportunities = []
        
        for event_data in events:
            try:
                opportunity = self.process_market_event(
                    title=event_data.get('title', ''),
                    description=event_data.get('description', ''),
                    source=event_data.get('source', ''),
                    raw_data=event_data
                )
                opportunities.append(opportunity)
            except Exception as e:
                print(f"Error processing event: {e}")
                continue
        
        return opportunities
    
    def get_learning_opportunities_by_difficulty(self, opportunities: List[LearningOpportunity], 
                                               difficulty: str) -> List[LearningOpportunity]:
        """Filter learning opportunities by difficulty level"""
        return [opp for opp in opportunities if opp.difficulty == difficulty]
    
    def get_learning_opportunities_by_concept(self, opportunities: List[LearningOpportunity], 
                                            concept: str) -> List[LearningOpportunity]:
        """Filter learning opportunities by concept"""
        return [opp for opp in opportunities if concept.lower() in opp.concept.lower()]
    
    def generate_daily_learning_summary(self, opportunities: List[LearningOpportunity]) -> Dict[str, Any]:
        """Generate a daily learning summary"""
        
        if not opportunities:
            return {
                'total_opportunities': 0,
                'difficulty_distribution': {},
                'concept_distribution': {},
                'total_learning_time': 0,
                'recommended_focus': 'No learning opportunities available today'
            }
        
        # Calculate statistics
        difficulty_distribution = {}
        concept_distribution = {}
        total_time = 0
        
        for opp in opportunities:
            # Difficulty distribution
            difficulty_distribution[opp.difficulty] = difficulty_distribution.get(opp.difficulty, 0) + 1
            
            # Concept distribution
            concept_distribution[opp.concept] = concept_distribution.get(opp.concept, 0) + 1
            
            # Total time
            total_time += opp.estimated_time
        
        # Find most common concept for focus
        most_common_concept = max(concept_distribution.items(), key=lambda x: x[1])[0] if concept_distribution else 'General Market Analysis'
        
        return {
            'total_opportunities': len(opportunities),
            'difficulty_distribution': difficulty_distribution,
            'concept_distribution': concept_distribution,
            'total_learning_time': total_time,
            'recommended_focus': most_common_concept,
            'opportunities': opportunities
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the engine
    engine = MarketEventLearningEngine()
    
    # Sample market event
    test_event = {
        'title': 'Whale moved $5M in ETH to Binance',
        'description': 'Large whale address transferred 2,500 ETH worth $5M to Binance exchange',
        'source': 'Etherscan',
        'amount': '$5,000,000'
    }
    
    # Process the event
    opportunity = engine.process_market_event(
        title=test_event['title'],
        description=test_event['description'],
        source=test_event['source'],
        raw_data=test_event
    )
    
    print(f"Generated Learning Opportunity:")
    print(f"Concept: {opportunity.concept}")
    print(f"Difficulty: {opportunity.difficulty}")
    print(f"Why Learn: {opportunity.why_learn}")
    print(f"Estimated Time: {opportunity.estimated_time} minutes")
    print(f"Learning Path: {opportunity.learning_path}")
