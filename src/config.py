"""
Configuration file for MAGMA Platform
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
LUNARCRUSH_API_KEY = os.getenv('LUNARCRUSH_API_KEY')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# API URLs
NEWS_API_BASE_URL = "https://newsapi.org/v2"
REDDIT_API_BASE_URL = "https://www.reddit.com"
LUNARCRUSH_API_BASE_URL = "https://api.lunarcrush.com/v2"
ETHERSCAN_API_BASE_URL = "https://api.etherscan.io/api"

# App Configuration
APP_NAME = "MAGMA Educational Platform"
APP_VERSION = "1.0.0"
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
