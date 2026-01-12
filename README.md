# AutoStream AI Agent - Social-to-Lead Workflow

> **ServiceHive ML Intern Assignment - Inflx Product Demo**

An intelligent conversational AI agent that converts social media conversations into qualified business leads using RAG (Retrieval Augmented Generation), intent detection, and agentic workflows.

---

## 🎯 Project Overview

**Company:** ServiceHive  
**Product:** Inflx  
**Use Case:** AutoStream - Automated video editing SaaS for content creators

This agent demonstrates:
- ✅ **Intent Classification** (Casual, Pricing, High Intent)
- ✅ **RAG-Powered Knowledge Retrieval** (FAISS + BM25 hybrid search)
- ✅ **Stateful Conversation Management** (5-6+ turn memory)
- ✅ **Tool Execution** (Lead capture with validation)
- ✅ **Production-Ready Architecture** (Error handling, logging, security)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
Gemini API Key
```

### Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd autostream-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

4. **Run the CLI demo**
```bash
python main_backend.py
```

5. **Run the Streamlit UI** (Optional)
```bash
streamlit run streamlit_frontend.py
```

---

## 📁 Project Structure

```
autostream-agent/
│
├── main_backend.py          # Core agent logic & conversation controller
├── RAG.py                   # RAG implementation with Gemini API
├── we_are.py               # AutoStream knowledge base
├── encryption_utils.py      # Logging & security utilities
├── get_secreats.py         # Environment variable loader
├── streamlit_frontend.py    # Web UI (optional)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🎬 Features Walkthrough

### 1️⃣ Intent Detection

The agent automatically classifies user messages into three categories:

| Intent | Keywords | Example |
|--------|----------|---------|
| **Casual** | hi, hello, hey | "Hi there!" |
| **Pricing** | price, cost, plan | "What's your pricing?" |
| **High Intent** | want, try, sign up | "I want to try Pro plan" |

**Implementation:** `detect_intent()` in `main_backend.py`

---

### 2️⃣ RAG-Powered Knowledge Retrieval

**Knowledge Base Content:**
- ✅ Basic Plan: $29/month, 10 videos, 720p
- ✅ Pro Plan: $79/month, unlimited videos, 4K, AI captions
- ✅ Refund Policy: No refunds after 7 days
- ✅ Support: 24/7 available only on Pro plan

**Technical Stack:**
- **Embeddings:** Gemini `text-embedding-004`
- **Vector Store:** FAISS (similarity search)
- **Keyword Search:** BM25Retriever
- **Hybrid Retrieval:** Ensemble (65% FAISS + 35% BM25)

**Implementation:** `RAGBot` class in `RAG.py`

---

### 3️⃣ Lead Capture Workflow

When high intent is detected, the agent initiates a 3-step qualification process:

```
User: "I want to try the Pro plan"
  ↓
Agent: [Detects HIGH INTENT] → "What's your name?"
  ↓
User: "John Doe"
  ↓
Agent: "What's your email?" [Validates format]
  ↓
User: "john@example.com"
  ↓
Agent: "Which platform do you create content on?"
  ↓
User: "YouTube"
  ↓
Agent: ✅ Calls mock_lead_capture(name, email, platform)
```

**Validation Features:**
- ✅ Name: Minimum 2 characters
- ✅ Email: Regex pattern validation (`user@domain.com`)
- ✅ Platform: Minimum 2 characters

**Tool Execution:** `mock_lead_capture()` in `main_backend.py`

---

### 4️⃣ State Management

**ChatState Dataclass:**
```python
@dataclass
class ChatState:
    status: str              # active | await_name | await_email | await_platform
    intent: str              # casual | pricing | high_intent
    name: str
    email: str
    platform: str
    lead_captured: bool
    interactions: int
```

**State Transitions:**
```
ACTIVE → await_name → await_email → await_platform → ACTIVE (lead captured)
```

**Memory:** Maintains conversation history for 5-6+ turns using `RAGCache`

---

## 🏗️ Architecture Explanation

### Why LangChain?

I chose **LangChain** over LangGraph/AutoGen for these reasons:

1. **RAG Excellence:** LangChain provides native, battle-tested support for RAG pipelines with FAISS vectorstores, embeddings, and retrievers - exactly what this assignment requires.

2. **Simplicity:** For a conversational agent with linear state transitions (active → name → email → platform), LangChain's straightforward approach is more maintainable than LangGraph's complex state graphs.

3. **Production Maturity:** Extensive documentation, active community support, and proven scalability in production environments make it the safer choice for real-world deployment.

4. **Flexibility:** Easy to extend with custom components (GeminiRESTChat, GeminiRESTEmbeddings) while maintaining clean separation of concerns.

### State Management Strategy

Instead of using LangChain's built-in memory (which can be complex), I implemented a custom **ChatState dataclass** that:
- ✅ Explicitly tracks conversation status and collected data
- ✅ Prevents race conditions with proper state transitions
- ✅ Enables easy debugging and testing
- ✅ Scales well for production (can be persisted to Redis/DB)

The state machine ensures the agent **never** prematurely calls `mock_lead_capture()` until all three fields (name, email, platform) are validated and collected.

---

## 📱 WhatsApp Integration Strategy

### How to Deploy with WhatsApp Webhooks

**Step 1: Set Up WhatsApp Business API**
1. Register for Meta Business Account
2. Create WhatsApp Business App
3. Get webhook verification token + permanent access token

**Step 2: Deploy Backend**
```python
from flask import Flask, request
import asyncio

app = Flask(__name__)

# Store one controller per user
user_sessions = {}

@app.route('/webhook', methods=['GET', 'POST'])
async def webhook():
    if request.method == 'GET':
        # Webhook verification
        verify_token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if verify_token == VERIFY_TOKEN:
            return challenge, 200
    
    if request.method == 'POST':
        # Process incoming message
        data = request.json
        message = data['entry'][0]['changes'][0]['value']['messages'][0]
        
        user_id = message['from']
        user_message = message['text']['body']
        
        # Get or create user session
        if user_id not in user_sessions:
            rag_bot = RAGBot(client_id=user_id)
            user_sessions[user_id] = ChatFlowController(rag_bot)
        
        # Get agent response
        controller = user_sessions[user_id]
        response = await controller.handle(user_message)
        
        # Send response via WhatsApp API
        send_whatsapp_message(user_id, response)
        
        return 'OK', 200

def send_whatsapp_message(phone_number, message):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "text": {"body": message}
    }
    requests.post(url, headers=headers, json=payload)
```

**Step 3: Session Persistence**
- Use **Redis** or **PostgreSQL** to store `ChatState` across server restarts
- Store user sessions with TTL (e.g., 24 hours)

**Step 4: Scale Considerations**
- Deploy on **AWS Lambda** or **Google Cloud Run** for auto-scaling
- Use **load balancer** for high traffic
- Implement **rate limiting** to prevent abuse

---

## 🧪 Testing Guide

### Test Case 1: Pricing Inquiry
```
User: What are your plans?
Expected: Agent returns Basic ($29) and Pro ($79) details from RAG
```

### Test Case 2: High Intent → Lead Capture
```
User: I want to sign up for Pro
Expected: Agent asks "What's your name?"

User: John Doe
Expected: Agent asks "What's your email?"

User: john@example.com
Expected: Agent asks "Which platform?"

User: YouTube
Expected: ✅ Console shows "LEAD CAPTURED SUCCESSFULLY"
```

### Test Case 3: Email Validation
```
User: I'm interested
Agent: What's your name?

User: Jane
Agent: What's your email?

User: invalid-email
Expected: Agent rejects and asks for valid email
```

### Test Case 4: Conversation Memory
```
User: What's the Basic plan?
Agent: [Returns Basic plan details]

User: How about Pro?
Expected: Agent remembers context and explains Pro plan
```

---

## 📊 Sample Conversation Flow

```
👤 User: Hi
🤖 Agent: Hello! I'm here to help you with AutoStream. How can I assist you today?

👤 User: What's your pricing?
🤖 Agent: We have two plans:
         • Basic Plan: $29/month (10 videos, 720p)
         • Pro Plan: $79/month (Unlimited videos, 4K, AI captions)

👤 User: That sounds great! I want to try the Pro plan for my YouTube channel
🤖 Agent: Awesome! Let me get you set up with AutoStream.
         First, what's your name?

👤 User: Sarah Johnson
🤖 Agent: Great to meet you, Sarah! 📧 What's your email address?

👤 User: sarah@example.com
🤖 Agent: Perfect! 🎬 Which platform do you create content on?

👤 User: YouTube
🤖 Agent: 🚀 You're all set, Sarah!
         Our team will reach out to you shortly.

==========================================================
✅ LEAD CAPTURED SUCCESSFULLY
==========================================================
📝 Name:     Sarah Johnson
📧 Email:    sarah@example.com
📱 Platform: YouTube
==========================================================
```

---

## 🎥 Demo Video

**[Link to Demo Video: 2-3 minutes]**

The video demonstrates:
1. ✅ Agent answering pricing question using RAG
2. ✅ High intent detection ("I want to try Pro plan")
3. ✅ Sequential lead collection (name → email → platform)
4. ✅ Successful tool execution with console output
5. ✅ Email validation working correctly

---

## 📦 Dependencies

```txt
# Core LangChain
langchain==0.1.0
langchain-community==0.0.13
langchain-core==0.1.10
langchain-text-splitters==0.0.1

# Vector Store & Retrieval
faiss-cpu==1.7.4
tiktoken==0.5.2

# LLM Integration
requests==2.31.0

# Data Processing
pydantic==2.5.3

# Utilities
python-dotenv==1.0.0

# UI (Optional)
streamlit==1.29.0
```

---

## ⚙️ Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Model Configuration (in `RAG.py`)
```python
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_CHUNK_SIZE = 450
DEFAULT_TOP_K = 5
```

---

## 🔐 Security Features

1. **Input Sanitization:** All user inputs sanitized using `sanitize_input()`
2. **Email Validation:** Regex pattern matching prevents invalid emails
3. **Logging:** Secure logging with PII masking via `encryption_utils.py`
4. **API Key Management:** Environment variables, never hardcoded
5. **Error Handling:** Graceful degradation with user-friendly messages

---

## 🚀 Production Deployment Checklist

- [ ] Environment variables configured
- [ ] API key secured (AWS Secrets Manager / GCP Secret Manager)
- [ ] Logging configured (CloudWatch / Stackdriver)
- [ ] Rate limiting implemented
- [ ] Error monitoring (Sentry / Rollbar)
- [ ] Load testing completed
- [ ] Database for state persistence (Redis / PostgreSQL)
- [ ] CI/CD pipeline setup (GitHub Actions)
- [ ] Backup strategy in place

---

## 📈 Evaluation Criteria Checklist

| Criteria | Status | Notes |
|----------|--------|-------|
| Intent Detection | ✅ | 3 categories: casual, pricing, high_intent |
| RAG Knowledge Base | ✅ | FAISS + BM25, AutoStream data loaded |
| Tool Execution | ✅ | `mock_lead_capture()` called after all fields |
| State Management | ✅ | ChatState tracks 5-6+ turns |
| Code Quality | ✅ | Clean, modular, well-documented |
| Deployability | ✅ | WhatsApp integration strategy included |

---

## 🐛 Known Limitations

1. **Single Session:** CLI demo doesn't persist state across restarts (use Streamlit UI or deploy with Redis for persistence)
2. **Rate Limits:** Gemini API has rate limits (handled with retry logic)
3. **No Authentication:** Production deployment requires user auth

---

## 🔮 Future Enhancements

- [ ] Multi-language support for international users
- [ ] Voice input/output for accessibility
- [ ] Integration with CRM (Salesforce, HubSpot)
- [ ] A/B testing for conversion optimization
- [ ] Analytics dashboard for lead metrics
- [ ] Sentiment analysis for user satisfaction

---

## 👨‍💻 Author

**Nirvish Patel**
- Email: nirvishpatel36@gmail.com

---

## 📄 License

This project was created for the ServiceHive ML Intern assignment (January 2026).

---

## 🙏 Acknowledgments

- **ServiceHive** for the opportunity
- **LangChain** for the RAG framework
- **Google Gemini** for LLM capabilities
- **Streamlit** for rapid UI prototyping

---

## 📞 Support

For questions or issues:
1. Check the code comments in `main_backend.py` and `RAG.py`
2. Review the test cases in this README
3. Open an issue on GitHub
4. Contact via email: nirvishpatel36@gmail.com

---

**⭐ If you found this project helpful, please star the repository!**
