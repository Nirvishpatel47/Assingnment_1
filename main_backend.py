import asyncio
from typing import Optional
from dataclasses import dataclass
from RAG import RAGBot
import re

# Intent Detection Keywords
CASUAL_GREETING = ["hi", "hello", "hey", "good morning", "good evening", "sup", "what's up"]
PRICING_KEYWORDS = ["price", "pricing", "cost", "plan", "how much", "payment", "subscription"]
HIGH_INTENT_KEYWORDS = [
    "want", "try", "sign up", "interested", "pro plan", "buy", 
    "get started", "purchase", "subscribe", "ready", "need this"
]


def detect_intent(message: str) -> str:
    """
    Classify user intent into: casual, pricing, or high_intent
    Returns: 'casual', 'pricing', or 'high_intent'
    """
    msg = message.lower().strip()
    
    # High intent takes priority
    if any(keyword in msg for keyword in HIGH_INTENT_KEYWORDS):
        return "high_intent"
    
    # Check for pricing inquiry
    if any(keyword in msg for keyword in PRICING_KEYWORDS):
        return "pricing"
    
    # Check for casual greeting
    if any(greeting in msg for greeting in CASUAL_GREETING):
        return "casual"
    
    # Default to casual for unknown intents
    return "casual"


def validate_email(email: str) -> bool:
    """Simple email validation"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None


def mock_lead_capture(name: str, email: str, platform: str):
    """
    Mock API function for lead capture
    This simulates backend lead storage
    """
    print("\n" + "="*60)
    print("✅ LEAD CAPTURED SUCCESSFULLY")
    print("="*60)
    print(f"📝 Name:     {name}")
    print(f"📧 Email:    {email}")
    print(f"📱 Platform: {platform}")
    print("="*60 + "\n")


@dataclass
class ChatState:
    """Maintains conversation state across turns"""
    status: str = "active"              # active, await_name, await_email, await_platform
    intent: Optional[str] = None        # casual, pricing, high_intent
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None
    lead_captured: bool = False
    interactions: int = 0
    last_intent: Optional[str] = None   # Track previous intent


class ChatFlowController:
    """
    Main controller for managing conversation flow and state
    Handles intent detection, lead qualification, and tool execution
    """
    
    def __init__(self, rag_bot: RAGBot):
        self.rag = rag_bot
        self.state = ChatState()

    async def handle(self, message: str) -> str:
        """
        Main handler for processing user messages
        Manages state transitions and responses
        """
        self.state.interactions += 1
        current_intent = detect_intent(message)
        
        # Update last intent tracking
        if current_intent in ["pricing", "high_intent"]:
            self.state.last_intent = current_intent
        
        if self.state.status == "await_name":
            name = message.strip()
            if len(name) < 2:
                return "Hmm, that doesn't look like a name. Could you share your full name?"
            
            self.state.name = name
            self.state.status = "await_email"
            return f"Great to meet you, {name}! 📧 What's your email address?"

        if self.state.status == "await_email":
            email = message.strip()
            
            # Validate email format
            if not validate_email(email):
                return "That doesn't look like a valid email. Please provide a valid email address (e.g., user@example.com)"
            
            self.state.email = email
            self.state.status = "await_platform"
            return "Perfect! 🎬 Which platform do you create content on? (YouTube, Instagram, TikTok, etc.)"

        if self.state.status == "await_platform":
            platform = message.strip()
            
            if len(platform) < 2:
                return "Could you specify your content platform? (e.g., YouTube, Instagram)"
            
            self.state.platform = platform
            
            # ✅ TOOL EXECUTION: Call mock API with all collected data
            mock_lead_capture( name=self.state.name, email=self.state.email, platform=self.state.platform )
            
            self.state.lead_captured = True
            self.state.status = "active"
            
            return ( f"🚀 You're all set, {self.state.name}!\n\n" "Our team will reach out to you shortly to help you get started with AutoStream Pro. " "We're excited to help you take your content to the next level! 🎥✨" )
        
        # High Intent Detection - Trigger Lead Capture
        if current_intent == "high_intent" and not self.state.lead_captured:
            # Get RAG response first
            answer = await self.rag.invoke(message, "English")
            
            self.state.intent = "high_intent"
            self.state.status = "await_name"
            
            return ( f"{answer}\n\n" "🎉 Awesome! Let me get you set up with AutoStream.\n\n" "First, what's your name?" )
        
        # Pricing or General Inquiry - Use RAG
        if current_intent in ["pricing", "casual"]:
            answer = await self.rag.invoke(message, "English")
            return answer
        
        return await self.rag.invoke(message, "English")
    
    def reset_state(self):
        """Reset conversation state (useful for new sessions)"""
        self.state = ChatState()
        print("🔄 Conversation state reset")


async def run_demo():
    """
    Demo function to test the agent locally
    Simulates a conversation with the AutoStream agent
    """
    rag = RAGBot(client_id="demo_user")
    controller = ChatFlowController(rag)

    print("\n" + "="*60)
    print("🎬 AutoStream AI Agent - Demo Mode")
    print("="*60)
    print("💡 This agent can:")
    print("   • Answer questions about AutoStream pricing & features")
    print("   • Detect high-intent users")
    print("   • Capture leads automatically")
    print("\n📝 Type 'exit' to quit")
    print("📝 Type 'reset' to start a new conversation")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("👤 User: ")
            
            if user_input.lower() == "exit":
                print("\n👋 Thanks for chatting! Goodbye!\n")
                break
            
            if user_input.lower() == "reset":
                controller.reset_state()
                continue
            
            if not user_input.strip():
                continue
            
            response = await controller.handle(user_input)
            print(f"🤖 Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    asyncio.run(run_demo())