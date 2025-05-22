from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from typing import Tuple
import re
import html

model = OllamaLLM(model="gemma2:2b")

template = """
You are a professional banking assistant. Your role is to strictly answer questions about:
- Bank accounts
- Loans and credit
- Financial products
- Banking policies
- Customer service issues

If asked about unrelated topics, politely decline to answer and redirect to banking topics.

Make sure to format the answer in a user-friendly manner.

Here is the relevant context: {context}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

BANKING_KEYWORDS = {
    # Account Types
    'account', 'savings', 'current', 'deposit', 'checking', 'term deposit', 
    'fixed deposit', 'digital account', 'remittance account', 'roshan digital',
    'special deposit', 'maximiser', 'pls', 'profit and loss sharing',
    'asaan account', 'freelancer account', 'waqaar account', 'sahar account',
    'value plus', 'premium business', 'business account', 'pakwatan',
    'home remittance', 'digital onboarding', 'little champs', 'minor account',
    
    # Services
    'debit card', 'credit card', 'chequebook', 'internet banking', 
    'mobile banking', 'sms alert', 'e-statement', 'fund transfer',
    'cash withdrawal', 'locker', 'insurance', 'takaful', 'profit rate',
    'interest rate', 'markup rate', 'loan', 'finance', 'mortgage',
    'auto finance', 'personal finance', 'housing finance', 'home loan',
    'car loan', 'education loan', 'remittance', 'home remittance',
    'western union', 'ria', 'dandelion', 'mastercard', 'paypak',
    'unionpay', 'visa', 'naya pakistan certificate', 'real estate',
    'stock market', 'investment', 'tdr', 'term deposit receipt',
    
    # Transactions
    'withdrawal', 'deposit', 'transfer', 'transaction', 'limit',
    'balance', 'minimum balance', 'overdraft', 'emi', 'installment',
    'payment', 'bill payment', 'standing instruction', 'dd', 'cheque',
    'banker cheque', 'atm', 'cash deposit', 'online banking',
    
    # Documents & Requirements
    'cnic', 'nicop', 'poc', 'arc', 'nara', 'passport', 'visa',
    'form-b', 'birth certificate', 'student id', 'proof of income',
    'documents', 'requirements', 'eligibility', 'criteria',
    
    # Features & Benefits
    'profit', 'return', 'yield', 'feature', 'benefit', 'facility',
    'free', 'charge', 'fee', 'service', 'insurance', 'coverage',
    'discount', 'offer', 'promotion', 'bonus', 'loyalty',
    
    # Customer Segments
    'individual', 'joint', 'minor', 'guardian', 'senior citizen',
    'women', 'freelancer', 'nrp', 'overseas pakistani', 'op',
    'business', 'company', 'corporate', 'partnership', 'proprietorship',
    'ngo', 'trust', 'society', 'government', 'armed forces',
    'fauji foundation', 'spd', 'ghq', 'nhq', 'ahq',
    
    # Processes
    'open', 'close', 'apply', 'application', 'processing',
    'approval', 'verification', 'disbursement', 'repayment',
    'maturity', 'rollover', 'renewal', 'termination', 'foreclosure',
    'balloon payment', 'early settlement', 'default', 'recovery',
    
    # Locations
    'branch', 'atm', 'online', 'digital', 'mobile', 'website',
    'portal', 'ibft', 'rtgs', 'interbank', 'domestic', 'international',
    'cross border', 'overseas',
    
    # Currencies
    'pkr', 'usd', 'gbp', 'eur', 'aed', 'jpy', 'currency', 'foreign',
    'local', 'exchange', 'conversion', 'rate',
    
    # Time-related
    'tenure', 'duration', 'monthly', 'quarterly', 'semi-annual',
    'annual', 'yearly', 'maturity', 'premature', 'encashment',
    
    # Regulations
    'sbp', 'state bank', 'regulation', 'compliance', 'zakat',
    'withholding tax', 'wht', 'tax', 'legal', 'requirement',
    'guideline', 'policy',
    
    # Other Important Terms
    'debit', 'credit', 'lien', 'hypothecation', 'collateral',
    'security', 'guarantor', 'co-borrower', 'debt', 'burden',
    'ratio', 'dbr', 'income', 'salary', 'pension', 'source',
    'verification', 'statement', 'transaction', 'history',
    'limit', 'threshold', 'maximum', 'minimum', 'average',
    'balance', 'overdraft', 'standing instruction', 'customer support'
}

PROHIBITED_TOPICS = {
    'pasta', 'food', 'weather', 'sports', 'politics', 'religion',
    'personal advice', 'medical', "bypass", "override", "ignore",
    "forget", "pretend", "disregard", "system prompt", "jailbreak",
    "hack", "exploit", "password"
}

def sanitize_input(text: str) -> str:
    """Clean user input before processing"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Escape special characters
    text = html.escape(text)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text.strip()

def check_query_appropriateness(question: str) -> Tuple[bool, str]:
    """Check if question is banking-related and appropriate"""
    question_lower = question.lower()
    
    # Check for prohibited topics
    for topic in PROHIBITED_TOPICS:
        if topic in question_lower:
            return False, f"I specialize in banking topics. Could I help you with financial services instead?"
    
    # Check for banking keywords
    for keyword in BANKING_KEYWORDS:
        if keyword in question_lower:
            return True, ""
    
    # If no banking keywords found
    return False, "I can only answer banking-related questions. Could you clarify your financial query?"


def validate_response(response: str) -> str:
    """Ensure response meets quality standards"""
    if not response:
        return "I couldn't generate a response. Please try rephrasing your question."
    
    if len(response.split()) < 5:  # Very short response
        return "Could you please provide more details about your question?"
    
    return response

# ===== MAIN RESPONSE FUNCTION =====
def get_response(question: str) -> str:
    # Sanitize input
    question = sanitize_input(question)

    # First check if question is appropriate
    is_appropriate, rejection_msg = check_query_appropriateness(question)
    if not is_appropriate:
        return rejection_msg
    
    # Proceed with normal response generation
    context_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    raw_response = chain.invoke({"context": context, "question": question})
    
    # Validate the response before returning
    return validate_response(raw_response)