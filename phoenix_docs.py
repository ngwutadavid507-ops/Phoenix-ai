import os
import io
import re
import math
import threading
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq
import PyPDF2
from dotenv import load_dotenv

class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Phoenix AI is alive")
    def log_message(self, format, *args):
        pass

def run_server():
    server = HTTPServer(("0.0.0.0", 8080), PingHandler)
    server.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ADMIN_ID = os.environ.get("ADMIN_ID")

client = Groq(api_key=GROQ_API_KEY)

def tokenize(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    index = defaultdict(list)
    doc_freq = defaultdict(int)
    for cid, chunk in enumerate(chunks):
        tokens = tokenize(chunk)
        if not tokens:
            continue
        freq = defaultdict(int)
        for token in tokens:
            freq[token] += 1
        for token, count in freq.items():
            tf = count / len(tokens)
            index[token].append((cid, tf))
            doc_freq[token] += 1
    return index, doc_freq

def retrieve_chunks(query, chunks, index, doc_freq, top_k=4):
    query_tokens = tokenize(query)
    n = len(chunks)
    scores = defaultdict(float)
    for token in query_tokens:
        if token not in index:
            continue
        idf = math.log((n + 1) / (doc_freq[token] + 1)) + 1.0
        for cid, tf in index[token]:
            scores[cid] += tf * idf
    if not scores:
        return chunks[:top_k]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[cid] for cid, _ in ranked[:top_k]]

def index_document(text):
    chunks = chunk_text(text)
    index, doc_freq = build_index(chunks)
    return {
        "chunks": chunks,
        "index": index,
        "doc_freq": doc_freq,
        "total_chunks": len(chunks)
    }

def build_rag_context(query, rag_store):
    top_chunks = retrieve_chunks(
        query,
        rag_store["chunks"],
        rag_store["index"],
        rag_store["doc_freq"],
        top_k=4
    )
    return "\n\n---\n\n".join(top_chunks)

async def notify_admin(context, message):
    try:
        await context.bot.send_message(chat_id=ADMIN_ID, text=message)
    except:
        pass

def ask_groq(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        timeout=25
    )
    return response.choices[0].message.content

def ask_groq_rag(system_prompt, question, context_text):
    user_prompt = (
        f"Use ONLY the document excerpts below to answer the question.\n"
        f"If the answer is not found in the excerpts, say: "
        f"'I could not find that in your document.'\n\n"
        f"Document excerpts:\n{context_text}\n\n"
        f"Question: {question}"
    )
    return ask_groq(system_prompt, user_prompt)

SUMMARISE_KEYWORDS = [
    "summarise", "summarize", "summary", "summarise all",
    "give me a summary", "what is this document about",
    "overview", "give overview"
]

QUIZ_KEYWORDS = [
    "quiz", "test me", "generate quiz", "quiz me",
    "give me questions", "make questions"
]

COMPARE_KEYWORDS = [
    "compare", "difference between", "similarities",
    "compare documents", "compare both"
]

def detect_intent(text):
    t = text.lower()
    if any(k in t for k in SUMMARISE_KEYWORDS):
        return "summarise"
    if any(k in t for k in QUIZ_KEYWORDS):
        return "quiz"
    if any(k in t for k in COMPARE_KEYWORDS):
        return "compare"
    return "question"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await notify_admin(
        context,
        f"🔔 New User Started Bot\n\n"
        f"Name: {user.full_name}\n"
        f"Username: @{user.username}\n"
        f"ID: {user.id}\n"
        f"Time: {update.message.date}"
    )
    keyboard = [
        [InlineKeyboardButton("📄 How to use", callback_data="how_to_use")],
        [InlineKeyboardButton("ℹ️ About Phoenix AI", callback_data="about")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🔥 Welcome to Phoenix Docs\n\n"
        "Part of the Phoenix AI platform\n\n"
        "What I can do:\n"
        "→ 📝 Summarise any document\n"
        "→ ❓ Answer any question\n"
        "→ 🎯 Generate quiz questions\n"
        "→ 🔍 Compare two documents\n"
        "→ 🌍 Respond in your language\n"
        "→ 🧠 RAG-powered smart search\n"
        "→ 📊 Auto document intelligence\n\n"
        "Send me a PDF or ask me anything!",
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "how_to_use":
        await query.message.reply_text(
            "📖 How to use Phoenix Docs\n\n"
            "1. Ask me any question directly\n"
            "2. Or send a PDF for deeper analysis\n"
            "3. After PDF I auto-analyse it for you\n\n"
            "Commands:\n"
            "/summarise - Summarise document\n"
            "/quiz - Generate quiz questions\n"
            "/compare - Compare two documents\n"
            "/language - Set response language\n"
            "/clear - Clear current document\n\n"
            "🧠 v4: Smart RAG search active"
        )
    elif query.data == "about":
        await query.message.reply_text(
            "🔥 Phoenix AI Platform\n\n"
            "Built by Chidibless from Nigeria 🇳🇬\n\n"
            "Phoenix Docs is the first tool in the "
            "Phoenix AI platform — privacy-first AI "
            "for everyone.\n\n"
            "Your documents never leave your hands.\n"
            "No big tech. No surveillance.\n\n"
            "Follow our journey: @PhoenixAi_Dev on X"
        )
    elif query.data == "summarise":
        await handle_summarise(query.message, context)
    elif query.data == "quiz":
        await handle_quiz(query.message, context)
    elif query.data == "compare":
        await handle_compare_request(query.message, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Phoenix Docs Commands\n\n"
        "/start - Welcome message\n"
        "/help - This help message\n"
        "/summarise - Summarise loaded document\n"
        "/quiz - Generate quiz from document\n"
        "/compare - Compare two documents\n"
        "/language - Set response language\n"
        "/clear - Clear current document\n\n"
        "🧠 v4: Smart RAG search active.\n"
        "Ask anything even without a document!"
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🗑️ All documents cleared.\n"
        "Send a new PDF to continue."
    )

async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("🇬🇧 English", callback_data="lang_English"),
            InlineKeyboardButton("🇫🇷 French", callback_data="lang_French"),
        ],
        [
            InlineKeyboardButton("🇦🇪 Arabic", callback_data="lang_Arabic"),
            InlineKeyboardButton("🇳🇬 Yoruba", callback_data="lang_Yoruba"),
        ],
        [
            InlineKeyboardButton("🇳🇬 Igbo", callback_data="lang_Igbo"),
            InlineKeyboardButton("🇳🇬 Hausa", callback_data="lang_Hausa"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🌍 Choose your response language:",
        reply_markup=reply_markup
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    doc = update.message.document

    if not doc.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("⚠️ Please send a PDF file only.")
        return

    await notify_admin(
        context,
        f"📄 Document Uploaded\n\n"
        f"Name: {user.full_name}\n"
        f"Username: @{user.username}\n"
        f"File: {doc.file_name}\n"
        f"Time: {update.message.date}"
    )

    is_second = 'document' in context.user_data

    if is_second:
        await update.message.reply_text("📄 Reading second document for comparison...")
