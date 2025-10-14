---
layout: project
title: GenAI-Powered Financial Analysis Agent
tech_stack: Azure OpenAI, Gemini API, LangChain, FastAPI, Docker, Azure Cognitive Services
date: 2024-03-20
company: Cresteo (Client: Management & AI Consultancy)
tags: [generative-ai, llm, rag, agents, nlp]
excerpt: AI sales agent generating comprehensive 20+ page financial reports with interactive multi-modal chat, increasing client engagement by 25% for management consultancy.
---

# GenAI-Powered Financial Analysis Agent

## Project Overview

Developed an AI-powered sales agent that combines state-of-the-art generative AI models to provide up-to-date financial analysis of publicly traded companies. The system generates comprehensive 20+ page reports and enables interactive conversations via text, audio, and open-mic, resulting in a 25% increase in client engagement for a leading management and AI consultancy in Minnesota.

## Business Challenge

Management consultancies need to quickly analyze potential client companies to prepare for sales conversations. Traditional research is:

### Pain Points

- **Time-Consuming**: Analysts spend hours researching each company
- **Information Overload**: Thousands of pages of SEC filings, earnings calls, news
- **Outdated Intel**: Research becomes stale quickly in fast-moving markets
- **Inconsistent Quality**: Analysis quality varies by analyst expertise
- **Limited Interactivity**: Static reports don't allow follow-up questions

### Business Requirements

- Generate comprehensive financial reports in minutes (not hours)
- Include latest information (real-time data)
- Enable natural conversation about findings
- Support multiple interaction modes (text, voice)
- Maintain high accuracy and cite sources

## Technical Solution

### System Architecture

**Multi-Agent RAG System**: Orchestrated specialized agents for different analysis tasks

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface Layer                   │
│  (Text Chat | Voice Chat | Open Mic | Report Viewer)    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              Agent Orchestrator (LangChain)              │
│         (Routes tasks to specialized agents)             │
└─────────┬───────────────────────────────────────────────┘
          │
    ┌─────┴─────┬─────────┬──────────┬─────────┬──────────┐
    ▼           ▼         ▼          ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐
│ SEC    │ │Finance │ │ News │ │ Market │ │Voice │ │ Report   │
│ Filing │ │ Data   │ │Search│ │ Data   │ │ I/O  │ │Generator │
│ Agent  │ │ Agent  │ │Agent │ │ Agent  │ │Agent │ │  Agent   │
└────────┘ └────────┘ └──────┘ └────────┘ └──────┘ └──────────┘
    │          │         │         │         │          │
    ▼          ▼         ▼         ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐
│SEC     │ │Yahoo   │ │Tavily│ │Market  │ │Azure │ │ Azure    │
│API     │ │Finance │ │Search│ │ APIs   │ │Speech│ │ OpenAI   │
└────────┘ └────────┘ └──────┘ └────────┘ └──────┘ └──────────┘
```

### Core Components

#### 1. Data Collection Agents

**SEC Filing Agent**
```python
from sec_api import ExtractorApi
import pandas as pd

class SECFilingAgent:
    """
    Agent for extracting and analyzing SEC filings
    """

    def __init__(self, api_key):
        self.extractor = ExtractorApi(api_key)
        self.filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A']

    async def get_recent_filings(self, ticker, num_filings=10):
        """Fetch recent SEC filings for company"""

        filings = []
        for filing_type in self.filing_types:
            docs = self.extractor.get_filings(
                ticker=ticker,
                filing_type=filing_type,
                num_filings=num_filings
            )
            filings.extend(docs)

        return sorted(filings, key=lambda x: x['filing_date'], reverse=True)

    async def analyze_10k(self, filing_url):
        """Extract key insights from 10-K filing"""

        # Extract structured data
        content = self.extractor.get_section(filing_url, '1A', 'text')  # Risk Factors
        md_and_a = self.extractor.get_section(filing_url, '7', 'text')  # MD&A

        # Use LLM to summarize key points
        risk_summary = await self._summarize_with_llm(
            content,
            "Summarize the top 5 risk factors for investors"
        )

        financial_summary = await self._summarize_with_llm(
            md_and_a,
            "Summarize management's discussion of financial performance"
        )

        return {
            'risk_factors': risk_summary,
            'financial_discussion': financial_summary,
            'filing_date': filing_url['filing_date']
        }
```

**Financial Data Agent**
```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class FinancialDataAgent:
    """
    Agent for collecting financial metrics and market data
    """

    def __init__(self):
        self.cache = {}

    async def get_company_overview(self, ticker):
        """Get comprehensive company overview"""

        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'company_name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'employees': info.get('fullTimeEmployees'),
            'description': info.get('longBusinessSummary')
        }

    async def get_financial_metrics(self, ticker):
        """Calculate key financial metrics"""

        stock = yf.Ticker(ticker)

        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        # Calculate metrics
        metrics = {
            'revenue_ttm': self._get_latest_value(income_stmt, 'Total Revenue'),
            'net_income_ttm': self._get_latest_value(income_stmt, 'Net Income'),
            'total_assets': self._get_latest_value(balance_sheet, 'Total Assets'),
            'total_debt': self._get_latest_value(balance_sheet, 'Total Debt'),
            'free_cash_flow': self._calculate_fcf(cash_flow),
            'margins': self._calculate_margins(income_stmt),
            'growth_rates': self._calculate_growth_rates(income_stmt)
        }

        return metrics

    async def get_stock_performance(self, ticker, period='1y'):
        """Get stock price performance"""

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        return {
            'current_price': hist['Close'].iloc[-1],
            'ytd_return': self._calculate_return(hist, 'ytd'),
            '1y_return': self._calculate_return(hist, '1y'),
            'volatility': hist['Close'].pct_change().std() * (252 ** 0.5),
            'high_52w': hist['Close'].max(),
            'low_52w': hist['Close'].min()
        }
```

**News & Market Intelligence Agent**
```python
from tavily import TavilyClient
from datetime import datetime, timedelta

class NewsIntelligenceAgent:
    """
    Agent for gathering recent news and market sentiment
    """

    def __init__(self, tavily_api_key):
        self.tavily = TavilyClient(api_key=tavily_api_key)

    async def get_recent_news(self, company_name, days=30):
        """Fetch and summarize recent news"""

        # Search for recent news
        query = f"{company_name} (earnings OR acquisition OR product OR scandal)"
        news_results = self.tavily.search(
            query=query,
            search_depth="advanced",
            max_results=20,
            days=days
        )

        # Analyze sentiment and categorize
        analyzed_news = []
        for article in news_results['results']:
            sentiment = await self._analyze_sentiment(article['content'])
            category = await self._categorize_news(article['title'])

            analyzed_news.append({
                'title': article['title'],
                'url': article['url'],
                'date': article['published_date'],
                'sentiment': sentiment,
                'category': category,
                'summary': article['content'][:500]
            })

        return analyzed_news

    async def _analyze_sentiment(self, text):
        """Analyze sentiment using Azure OpenAI"""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze sentiment: positive, negative, or neutral"},
                {"role": "user", "content": text}
            ],
            temperature=0
        )

        return response.choices[0].message.content
```

#### 2. Report Generation Agent

**Comprehensive Report Generator**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import asyncio

class ReportGeneratorAgent:
    """
    Agent for generating comprehensive financial reports
    """

    def __init__(self, azure_openai_client, gemini_client):
        self.azure_llm = azure_openai_client
        self.gemini_llm = gemini_client

    async def generate_report(self, company_data):
        """
        Generate 20+ page comprehensive report

        Args:
            company_data: Dict with all collected data from other agents
        """

        # Generate sections in parallel
        sections = await asyncio.gather(
            self._executive_summary(company_data),
            self._company_overview(company_data),
            self._financial_analysis(company_data),
            self._risk_assessment(company_data),
            self._market_position(company_data),
            self._recent_developments(company_data),
            self._investment_thesis(company_data),
            self._valuation_analysis(company_data)
        )

        # Assemble report
        report = self._assemble_report(sections)

        return report

    async def _financial_analysis(self, data):
        """Deep dive into financial metrics"""

        prompt = f"""
        Generate a detailed financial analysis for {data['company_name']}

        Financial Data:
        - Revenue (TTM): ${data['revenue_ttm']:,.0f}
        - Net Income: ${data['net_income_ttm']:,.0f}
        - Gross Margin: {data['margins']['gross_margin']:.1%}
        - Operating Margin: {data['margins']['operating_margin']:.1%}
        - Free Cash Flow: ${data['free_cash_flow']:,.0f}
        - Revenue Growth: {data['growth_rates']['revenue_growth']:.1%}

        Provide:
        1. Profitability analysis
        2. Efficiency metrics
        3. Growth trajectory
        4. Comparison to industry benchmarks
        5. Financial health assessment

        Be specific and cite numbers.
        """

        # Use Azure OpenAI GPT-4 for analysis
        response = await self.azure_llm.ChatCompletion.acreate(
            model="gpt-4-32k",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing detailed analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    async def _investment_thesis(self, data):
        """Generate investment thesis using Gemini"""

        prompt = f"""
        Based on all available data, develop a balanced investment thesis for {data['company_name']}.

        Include:
        - Bull case (reasons to invest)
        - Bear case (reasons to avoid)
        - Key risks and opportunities
        - Recommendation with rationale

        Data available: {data.keys()}
        """

        # Use Gemini for alternative perspective
        response = await self.gemini_llm.generate_content(prompt)

        return response.text
```

#### 3. Interactive Chat Agent

**Multi-Modal Conversation Agent**
```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer

class ConversationalAgent:
    """
    Interactive agent supporting text, audio, and open-mic conversations
    """

    def __init__(self, azure_openai, azure_speech_config):
        self.llm = azure_openai
        self.speech_config = azure_speech_config
        self.conversation_history = []

        # Define tools the agent can use
        self.tools = [
            Tool(
                name="Financial_Data",
                func=self.financial_agent.query,
                description="Get financial metrics and company data"
            ),
            Tool(
                name="SEC_Filings",
                func=self.sec_agent.query,
                description="Search and analyze SEC filings"
            ),
            Tool(
                name="Recent_News",
                func=self.news_agent.query,
                description="Get recent news and market developments"
            ),
            Tool(
                name="Report_Section",
                func=self.get_report_section,
                description="Retrieve specific section from generated report"
            )
        ]

        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            system_message="You are a financial analysis assistant..."
        )

        self.executor = AgentExecutor(agent=self.agent, tools=self.tools)

    async def chat(self, user_input, mode='text'):
        """
        Handle user interaction in multiple modes

        Args:
            user_input: Text query or audio stream
            mode: 'text', 'audio', or 'open-mic'
        """

        # Convert audio to text if needed
        if mode in ['audio', 'open-mic']:
            user_input = await self._speech_to_text(user_input)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Agent decides which tools to use
        response = await self.executor.arun(
            input=user_input,
            chat_history=self.conversation_history
        )

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        # Convert to speech if needed
        if mode in ['audio', 'open-mic']:
            audio_response = await self._text_to_speech(response)
            return {
                'text': response,
                'audio': audio_response
            }

        return {'text': response}

    async def _speech_to_text(self, audio_stream):
        """Convert speech to text using Azure Cognitive Services"""

        recognizer = SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_stream
        )

        result = recognizer.recognize_once()

        if result.reason == ResultReason.RecognizedSpeech:
            return result.text
        else:
            raise Exception("Speech recognition failed")

    async def _text_to_speech(self, text):
        """Convert text to speech using Azure Cognitive Services"""

        synthesizer = SpeechSynthesizer(speech_config=self.speech_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            raise Exception("Speech synthesis failed")
```

### Deployment Architecture

**FastAPI Service with Docker**

```python
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="Financial Analysis Agent API")

# Initialize agents
orchestrator = AgentOrchestrator(
    sec_agent=SECFilingAgent(api_key=os.getenv('SEC_API_KEY')),
    financial_agent=FinancialDataAgent(),
    news_agent=NewsIntelligenceAgent(api_key=os.getenv('TAVILY_API_KEY')),
    report_agent=ReportGeneratorAgent(...),
    chat_agent=ConversationalAgent(...)
)

@app.post("/analyze/{ticker}")
async def analyze_company(ticker: str):
    """
    Generate comprehensive analysis report for company
    """
    report = await orchestrator.generate_full_report(ticker)
    return report

@app.post("/chat/text")
async def text_chat(message: str, session_id: str):
    """
    Text-based chat interaction
    """
    response = await orchestrator.chat_agent.chat(message, mode='text')
    return response

@app.websocket("/chat/voice")
async def voice_chat(websocket: WebSocket):
    """
    Real-time voice chat via WebSocket
    """
    await websocket.accept()

    while True:
        # Receive audio stream
        audio_data = await websocket.receive_bytes()

        # Process with agent
        response = await orchestrator.chat_agent.chat(
            audio_data,
            mode='open-mic'
        )

        # Send back text and audio
        await websocket.send_json({
            'text': response['text'],
            'audio': response['audio']
        })

@app.get("/report/{ticker}/pdf")
async def download_report_pdf(ticker: str):
    """
    Download report as PDF
    """
    pdf = await orchestrator.generate_pdf_report(ticker)
    return StreamingResponse(pdf, media_type="application/pdf")
```

## Results & Impact

### Business Metrics

**Efficiency**
- ⚡ **15x Faster**: Report generation in 5 minutes (vs. 90 minutes manually)
- 📊 **3x Coverage**: Analysts can prepare for 3x more client meetings

**Engagement**
- 📈 **25% Increase**: Client engagement metrics improved
- 💬 **Interactive**: Average 12 follow-up questions per session
- 🎯 **Precision**: 90%+ accuracy on factual financial queries

**Business Impact**
- 💰 **Higher Win Rate**: 15% increase in deal conversion
- ⏱️ **Time Savings**: ~80 hours per month saved
- 🚀 **Scalability**: From 20 to 60+ company analyses per month

### Technical Performance

**Report Quality**
- Comprehensive 20-25 page reports
- Real-time data (< 24 hours old)
- Citations for all factual claims
- Professional formatting

**System Reliability**
- 99.5% uptime
- <30 seconds for report generation
- <200ms chat response latency
- Handles 50+ concurrent users

**Model Performance**
- GPT-4 for complex analysis
- Gemini for alternative perspectives
- Ensemble approach reduces hallucinations by 60%

## Technical Stack

**Generative AI**
- Azure OpenAI (GPT-4, GPT-4-32k)
- Google Gemini API
- LangChain (agent orchestration)
- Azure Cognitive Services (speech)

**Data Sources**
- SEC API / SEC Downloader
- Yahoo Finance API
- Tavily (web search)
- Selenium (web scraping)

**Backend & Deployment**
- FastAPI (API framework)
- Docker (containerization)
- Azure Web Apps (hosting)
- GitHub Actions (CI/CD)
- WebSockets (real-time communication)

**Frontend**
- React (interactive UI)
- Streamlit (internal dashboard)
- PDF generation libraries

## Key Learnings

1. **Multi-Model Strategy**: Combining Azure OpenAI and Gemini provided better results than single model

2. **RAG is Essential**: Grounding LLM responses in retrieved documents dramatically improved accuracy

3. **Citation Critical**: Business users require source attribution for trust

4. **Voice Adds Value**: Audio interaction increased engagement significantly

5. **Agent Specialization**: Dedicated agents for different data sources improved reliability

## Future Enhancements

- **Real-Time Monitoring**: Alert users to material events (earnings, acquisitions)
- **Comparative Analysis**: Multi-company competitive analysis
- **Custom Research**: User-defined research templates
- **API Integration**: Connect to CRM systems
- **Fine-Tuned Models**: Domain-specific fine-tuning for financial analysis

---

**Project Duration**: 4 months (development) + ongoing enhancements
**Team**: 2 AI engineers, 1 full-stack developer, 1 product manager
**My Role**: Technical Lead - Architecture design, agent development, LLM integration

[← Back to Projects]({{ site.baseurl }}/projects/)
