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

# ─── KEEP-ALIVE SERVER ────────────────────────────────────

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

# ─── ENV & CLIENT ─────────────────────────────────────────

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ADMIN_ID = os.environ.get("ADMIN_ID")

client = Groq(api_key=GROQ_API_KEY)

# ─── RAG SYSTEM ───────────────────────────────────────────

def tokenize(text):
    """Lowercase and extract meaningful word tokens."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def chunk_text(text, chunk_size=250, overlap=40):
    """Split text into overlapping word chunks."""
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
    """Build TF-IDF inverted index from chunks."""
    index = defaultdict(list)   # term -> [(chunk_id, tf)]
    doc_freq = defaultdict(int) # term -> number of chunks containing it

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
    """Score and return the top_k most relevant chunks for a query."""
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
        # Fallback: return first top_k chunks if no match
        return chunks[:top_k]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[cid] for cid, _ in ranked[:top_k]]

def index_document(text):
    """Index a document and return the RAG store dict."""
    chunks = chunk_text(text)
    index, doc_freq = build_index(chunks)
    return {
        "chunks": chunks,
        "index": index,
        "doc_freq": doc_freq,
        "total_chunks": len(chunks)
    }

def build_rag_context(query, rag_store):
    """Retrieve relevant chunks and format as context string."""
    top_chunks = retrieve_chunks(
        query,
        rag_store["chunks"],
        rag_store["index"],
        rag_store["doc_freq"],
        top_k=4
    )
    return "\n\n---\n\n".join(top_chunks)

# ─── AI HELPERS ───────────────────────────────────────────

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
    """Ask Groq using retrieved RAG context only."""
    user_prompt = (
        f"Use ONLY the document excerpts below to answer the question.\n"
        f"If the answer is not found in the excerpts, say: "
        f"'I could not find that in your document.'\n\n"
        f"Document excerpts:\n{context_text}\n\n"
        f"Question: {question}"
    )
    return ask_groq(system_prompt, user_prompt)

# ─── COMMAND HANDLERS ─────────────────────────────────────

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
        "→ ❓ Answer questions about it\n"
        "→ 🎯 Generate quiz questions\n"
        "→ 🔍 Compare two documents\n"
        "→ 🌍 Respond in your language\n"
        "→ 🧠 RAG-powered smart search\n\n"
        "Send me a PDF to get started.",
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "how_to_use":
        await query.message.reply_text(
            "📖 How to use Phoenix Docs\n\n"
            "1. Send any PDF document\n"
            "2. Wait for confirmation\n"
            "3. Choose what you want:\n\n"
            "Commands after sending PDF:\n"
            "/summarise - Get full summary\n"
            "/quiz - Generate quiz questions\n"
            "/compare - Compare two documents\n"
            "/language - Set response language\n"
            "/clear - Clear current document\n\n"
            "Or just ask any question about it!\n\n"
            "🧠 v4: Questions now use smart RAG search\n"
            "for more accurate answers."
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
        "🧠 v4: Smart RAG search active\n"
        "Ask any question for precise answers."
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

# ─── DOCUMENT HANDLER ─────────────────────────────────────

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
    else:
        await update.message.reply_text("📄 Phoenix Docs is reading your document...")

    file = await doc.get_file()
    file_bytes = await file.download_as_bytearray()

    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    total_pages = len(pdf_reader.pages)

    text = ""
    for i, page in enumerate(pdf_reader.pages):
        if i >= 20:
            break
        extracted = page.extract_text()
        if extracted:
            text += extracted

    text = text[:12000]

    if is_second:
        context.user_data['document2'] = text
        # Index second document for RAG too
        context.user_data['rag2'] = index_document(text)
        keyboard = [
            [InlineKeyboardButton("🔍 Compare now", callback_data="compare")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"✅ Second document loaded\n\n"
            f"📊 Pages: {total_pages}\n\n"
            f"Ready to compare both documents.",
            reply_markup=reply_markup
        )
    else:
        context.user_data['document'] = text
        context.user_data['doc_name'] = doc.file_name

        # ── Build RAG index ──
        await update.message.reply_text("🧠 Building smart search index...")
        rag_store = index_document(text)
        context.user_data['rag'] = rag_store
        # ────────────────────

        pages_analysed = min(total_pages, 20)
        keyboard = [
            [
                InlineKeyboardButton("📝 Summarise", callback_data="summarise"),
                InlineKeyboardButton("🎯 Quiz me", callback_data="quiz"),
            ],
            [
                InlineKeyboardButton("🔍 Compare with another PDF", callback_data="compare")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"✅ Document loaded & indexed\n\n"
            f"📊 Total pages: {total_pages}\n"
            f"🔍 Pages analysed: {pages_analysed}\n"
            f"🧠 RAG chunks indexed: {rag_store['total_chunks']}\n\n"
            f"What would you like to do?",
            reply_markup=reply_markup
        )

# ─── FEATURE HANDLERS ─────────────────────────────────────

async def handle_summarise(message, context):
    if 'document' not in context.user_data:
        await message.reply_text("⚠️ No document loaded. Please send a PDF first.")
        return

    await message.reply_text("📝 Generating summary...")
    lang = context.user_data.get('language', 'English')
    document = context.user_data['document']

    try:
        answer = ask_groq(
            f"You are Phoenix Docs. Provide a comprehensive summary. Respond in {lang}.",
            f"Document:\n{document}\n\nProvide a detailed summary."
        )
        await message.reply_text(f"📝 Summary:\n\n{answer}")
    except Exception:
        await message.reply_text("⚠️ Summary timed out. Please try again.")

async def handle_quiz(message, context):
    if 'document' not in context.user_data:
        await message.reply_text("⚠️ No document loaded. Please send a PDF first.")
        return

    await message.reply_text("🎯 Generating quiz questions...")
    lang = context.user_data.get('language', 'English')
    document = context.user_data['document']

    try:
        answer = ask_groq(
            f"You are Phoenix Docs. Generate quiz questions. Respond in {lang}.",
            f"Document:\n{document}\n\n"
            f"Generate 5 multiple choice questions with 4 options each (A, B, C, D). "
            f"Include the correct answer at the end of each question."
        )
        await message.reply_text(f"🎯 Quiz Questions:\n\n{answer}")
    except Exception:
        await message.reply_text("⚠️ Quiz generation timed out. Please try again.")

async def handle_compare_request(message, context):
    if 'document' not in context.user_data:
        await message.reply_text("⚠️ No document loaded. Please send a PDF first.")
        return

    if 'document2' not in context.user_data:
        await message.reply_text("📄 Send me the second PDF to compare with.")
        return

    await message.reply_text("🔍 Comparing documents...")
    lang = context.user_data.get('language', 'English')
    doc1 = context.user_data['document']
    doc2 = context.user_data['document2']

    try:
        answer = ask_groq(
            f"You are Phoenix Docs. Compare two documents clearly. Respond in {lang}.",
            f"Document 1:\n{doc1}\n\nDocument 2:\n{doc2}\n\n"
            f"Compare these documents. List:\n"
            f"1. Key similarities\n"
            f"2. Key differences\n"
            f"3. Which is more comprehensive and why"
        )
        await message.reply_text(f"🔍 Comparison Result:\n\n{answer}")
    except Exception:
        await message.reply_text("⚠️ Comparison timed out. Please try again.")

# ─── QUESTION HANDLER (RAG-POWERED) ───────────────────────

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'document' not in context.user_data:
        await update.message.reply_text(
            "⚠️ No document loaded.\n"
            "Please send a PDF first."
        )
        return

    user = update.effective_user
    question = update.message.text
    lang = context.user_data.get('language', 'English')

    await notify_admin(
        context,
        f"❓ Question Asked\n\n"
        f"Name: {user.full_name}\n"
        f"Username: @{user.username}\n"
        f"Question: {question}\n"
        f"Time: {update.message.date}"
    )

    await update.message.reply_text("🧠 Searching your document...")

    try:
        rag_store = context.user_data.get('rag')

        if rag_store:
            # ── RAG path: retrieve relevant chunks, answer from them ──
            context_text = build_rag_context(question, rag_store)
            answer = ask_groq_rag(
                f"You are Phoenix Docs. Answer questions based only on the provided "
                f"document excerpts. Be clear and helpful. "
                f"Keep answers under 500 words. Respond in {lang}.",
                question,
                context_text
            )
        else:
            # ── Fallback: full document (old v3 behaviour) ──
            document = context.user_data['document']
            answer = ask_groq(
                f"You are Phoenix Docs. Answer questions based only on the document. "
                f"Be clear and helpful. Keep answers under 500 words. Respond in {lang}.",
                f"Document:\n{document}\n\nQuestion:\n{question}"
            )

        await update.message.reply_text(answer)

    except Exception:
        await update.message.reply_text("⚠️ Analysis timed out. Please try again.")

# ─── LANGUAGE CALLBACK ────────────────────────────────────

async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang = query.data.replace("lang_", "")
    context.user_data['language'] = lang
    await query.message.reply_text(
        f"✅ Language set to {lang}\n\n"
        f"All responses will now be in {lang}."
    )

# ─── COMMAND WRAPPERS ─────────────────────────────────────

async def summarise_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_summarise(update.message, context)

async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_quiz(update.message, context)

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_compare_request(update.message, context)

# ─── MAIN ─────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("summarise", summarise_command))
    app.add_handler(CommandHandler("quiz", quiz_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CallbackQueryHandler(handle_language_callback, pattern="^lang_"))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    print("🔥 Phoenix Docs v4 is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
