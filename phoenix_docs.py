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
from duckduckgo_search import DDGS

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ADMIN_ID = os.environ.get("ADMIN_ID")

client = Groq(api_key=GROQ_API_KEY)

def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return None
            search_text = ""
            for r in results:
                search_text += f"Title: {r.get('title', '')}\n"
                search_text += f"Summary: {r.get('body', '')}\n\n"
            return search_text
    except Exception:
        return None

def needs_web_search(question):
    current_keywords = [
        "current", "now", "today", "latest", "recent", "right now",
        "this year", "2024", "2025", "2026", "who is the president",
        "who is president", "prime minister", "ceo of", "price of",
        "stock price", "weather", "news", "just happened", "recently",
        "score", "winner", "election", "result"
    ]
    q = question.lower()
    return any(k in q for k in current_keywords)

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
        "→ 📊 Auto document intelligence\n"
        "→ 🌐 Live web search\n\n"
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
            "🌐 Live web search for current events\n"
            "🧠 Smart RAG search for documents"
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
        "🌐 Asks current questions? I search the web!\n"
        "🧠 Smart RAG search for your documents"
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
    else:
        await update.message.reply_text("📄 Phoenix Docs is reading your document...")

    file = await doc.get_file()
    file_bytes = await file.download_as_bytearray()

    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    total_pages = len(pdf_reader.pages)

    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if is_second:
        context.user_data['document2'] = text
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

        await update.message.reply_text("🧠 Building smart search index...")
        rag_store = index_document(text)
        context.user_data['rag'] = rag_store

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
            f"🧠 RAG chunks: {rag_store['total_chunks']}\n\n"
            f"What would you like to do?",
            reply_markup=reply_markup
        )

        await update.message.reply_text("📊 Generating document intelligence report...")

        try:
            lang = context.user_data.get('language', 'English')
            sample = text[:8000]
            report = ask_groq(
                f"You are Phoenix Docs. Analyse this document and provide "
                f"a structured intelligence report. Respond in {lang}.",
                f"Document:\n{sample}\n\n"
                f"Provide a document intelligence report with:\n"
                f"1. Document Type\n"
                f"2. Main Topic (one sentence)\n"
                f"3. Key Sections (top 3-5)\n"
                f"4. Important Facts (top 3)\n"
                f"5. Suggested Questions (3 questions)\n\n"
                f"Keep it concise and structured."
            )
            await update.message.reply_text(
                f"📊 Document Intelligence Report\n\n{report}"
            )
        except Exception:
            pass

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
            f"Document:\n{document}\n\nProvide a detailed summary covering all key points."
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

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    if 'document' not in context.user_data:
        await update.message.reply_text("🔍 Phoenix Docs is thinking...")
        try:
            if needs_web_search(question):
                await update.message.reply_text("🌐 Searching the web...")
                search_results = web_search(question)
                if search_results:
                    answer = ask_groq(
                        f"You are Phoenix Docs. Answer the question using "
                        f"the web search results provided. Be accurate and "
                        f"cite that information is from current web search. "
                        f"Respond in {lang}.",
                        f"Web search results:\n{search_results}\n\n"
                        f"Question: {question}"
                    )
                else:
                    answer = ask_groq(
                        f"You are Phoenix Docs. Answer helpfully. "
                        f"If unsure about current info say to verify online. "
                        f"Respond in {lang}.",
                        question
                    )
            else:
                answer = ask_groq(
                    f"You are Phoenix Docs, part of the Phoenix AI platform. "
                    f"A powerful AI assistant built by Chidibless from Nigeria. "
                    f"Answer any question helpfully and accurately. "
                    f"If the question is about documents mention that the user "
                    f"can send a PDF for deeper analysis. "
                    f"Be clear and concise. Respond in {lang}.",
                    question
                )
            await update.message.reply_text(answer)
        except Exception:
            await update.message.reply_text(
                "⚠️ Something went wrong. Please try again."
            )
        return

    intent = detect_intent(question)
    if intent == "summarise":
        await handle_summarise(update.message, context)
        return
    if intent == "quiz":
        await handle_quiz(update.message, context)
        return
    if intent == "compare":
        await handle_compare_request(update.message, context)
        return

    await update.message.reply_text("🧠 Searching your document...")

    try:
        rag_store = context.user_data.get('rag')

        if rag_store:
            context_text = build_rag_context(question, rag_store)
            answer = ask_groq_rag(
                f"You are Phoenix Docs. Answer questions based only on the provided "
                f"document excerpts. Be clear and helpful. "
                f"Keep answers under 500 words. Respond in {lang}.",
                question,
                context_text
            )
        else:
            document = context.user_data['document']
            answer = ask_groq(
                f"You are Phoenix Docs. Answer questions based only on the document. "
                f"Be clear and helpful. Keep answers under 500 words. Respond in {lang}.",
                f"Document:\n{document}\n\nQuestion:\n{question}"
            )

        await update.message.reply_text(answer)

    except Exception:
        await update.message.reply_text("⚠️ Analysis timed out. Please try again.")

async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang = query.data.replace("lang_", "")
    context.user_data['language'] = lang
    await query.message.reply_text(
        f"✅ Language set to {lang}\n\n"
        f"All responses will now be in {lang}."
    )

async def summarise_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_summarise(update.message, context)

async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_quiz(update.message, context)

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_compare_request(update.message, context)

if __name__ == "__main__":
    def run_server():
        cla
