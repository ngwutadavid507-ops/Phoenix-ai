import os
import io
import re
import math
import threading
import requests
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ADMIN_ID = os.environ.get("ADMIN_ID")
PORT = int(os.environ.get("PORT", 8080))
BACKEND_URL = "https://phoenix-backend-hvvc.onrender.com"

def ask_backend(question, language="English", platform="telegram"):
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json={"question": question, "language": language, "platform": platform}, timeout=30)
        return response.json().get("answer", "Error communicating with Core Engine.")
    except Exception:
        return "Backend processing error. Please retry."

def analyse_backend(text, action="summarise", question="", language="English"):
    try:
        response = requests.post(f"{BACKEND_URL}/analyse", json={"text": text, "action": action, "question": question, "language": language}, timeout=30)
        return response.json().get("answer", "Error running document intelligence loops.")
    except Exception:
        return "Backend analysis error."

def compare_backend(doc1, doc2, language="English"):
    try:
        response = requests.post(f"{BACKEND_URL}/compare", json={"document1": doc1, "document2": doc2, "language": language}, timeout=30)
        return response.json().get("answer", "Error comparing assets.")
    except Exception:
        return "Backend comparison error."

# Text Utility fallbacks for UI routing stability
def tokenize(text): return re.findall(r'\b[a-z]{2,}\b', text.lower())
def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split(); chunks = []; i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words): break
        i += chunk_size - overlap
    return chunks

def build_index(chunks):
    index = defaultdict(list); dfreq = defaultdict(int)
    for cid, chunk in enumerate(chunks):
        tks = tokenize(chunk)
        if not tks: continue
        freq = defaultdict(int)
        for t in tks: freq[t] += 1
        for t, count in freq.items():
            index[t].append((cid, count / len(tks)))
            dfreq[t] += 1
    return index, dfreq

def retrieve_chunks(query, chunks, index, doc_freq, top_k=4):
    tks = tokenize(query); scores = defaultdict(float)
    for t in tks:
        if t not in index: continue
        idf = math.log((len(chunks) + 1) / (doc_freq[t] + 1)) + 1.0
        for cid, tf in index[t]: scores[cid] += tf * idf
    if not scores: return chunks[:top_k]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[cid] for cid, _ in ranked[:top_k]]

async def notify_admin(context, message):
    try: await context.bot.send_message(chat_id=ADMIN_ID, text=message)
    except Exception: pass

def detect_intent(text):
    t = text.lower()
    if any(k in t for k in ["summarise", "summarize", "summary", "overview"]): return "summarise"
    if any(k in t for k in ["quiz", "test me", "generate quiz", "questions"]): return "quiz"
    if any(k in t for k in ["compare", "difference between", "similarities"]): return "compare"
    return "question"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await notify_admin(context, f"User Started Bot: {user.full_name} (@{user.username})")
    keyboard = [[InlineKeyboardButton("How to use", callback_data="how_to_use")], [InlineKeyboardButton("About Phoenix AI", callback_data="about")]]
    await update.message.reply_text("🔥 Welcome to Phoenix Docs\n\nSend a PDF, ask a question, send an image, or drop a voice note!", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    voice = update.message.voice
    lang = context.user_data.get('language', 'English')
    
    await update.message.reply_text("🎙️ Transcribing voice note...")
    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()
    
    try:
        # Route voice data stream into the Backend Core
        files = {"file": ("voice.ogg", io.BytesIO(file_bytes), "audio/ogg")}
        resp = requests.post(f"{BACKEND_URL}/voice", files=files, timeout=30)
        transcript = resp.json().get("transcript", "")
        
        if not transcript:
            await update.message.reply_text("Could not extract speech from that audio.")
            return
            
        await update.message.reply_text(f"🗣️ *Transcribed Text:* {transcript}\n\n*Phoenix is processing answer...*", parse_mode="Markdown")
        
        # Pass transcript into standard response pipeline
        if 'document' not in context.user_data:
            answer = ask_backend(transcript, lang, "telegram")
        else:
            rag_store = context.user_data.get('rag')
            context_text = "\n\n---\n\n".join(retrieve_chunks(transcript, rag_store["chunks"], rag_store["index"], rag_store["doc_freq"]))
            answer = analyse_backend(context_text, "question", question=transcript, language=lang)
            
        await update.message.reply_text(answer)
    except Exception:
        await update.message.reply_text("Failed to process audio thread downstream.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    photo_file = update.message.photo[-1] # Pull the sharpest/largest option
    lang = context.user_data.get('language', 'English')
    caption = update.message.caption if update.message.caption else "Analyze this image accurately."
    
    await update.message.reply_text("👁️ Analyzing image details...")
    file = await context.bot.get_file(photo_file.file_id)
    
    # Telegram CDN links are public and perfectly accessible by your Render backend servers
    telegram_cdn_url = file.file_path 
    
    try:
        payload = {"image_url": telegram_cdn_url, "prompt": caption, "language": lang}
        resp = requests.post(f"{BACKEND_URL}/vision", json=payload, timeout=30)
        answer = resp.json().get("answer", "Vision engine timed out.")
        await update.message.reply_text(answer)
    except Exception:
        await update.message.reply_text("Vision analysis request failed.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("Please drop PDF extensions only.")
        return

    is_second = 'document' in context.user_data
    await update.message.reply_text("Reading document data blocks...")
    
    file = await doc.get_file()
    file_bytes = await file.download_as_bytearray()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted: text += extracted

    if is_second:
        context.user_data['document2'] = text
        await update.message.reply_text("Second PDF indexed. Ready to run /compare commands.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Compare now", callback_data="compare")]]))
    else:
        context.user_data['document'] = text
        chunks = chunk_text(text)
        idx, dfreq = build_index(chunks)
        context.user_data['rag'] = {"chunks": chunks, "index": idx, "doc_freq": dfreq}
        
        keyboard = [[InlineKeyboardButton("Summarise", callback_data="summarise"), InlineKeyboardButton("Quiz me", callback_data="quiz")]]
        await update.message.reply_text("Document saved into temporary RAG layer. What are we processing?", reply_markup=InlineKeyboardMarkup(keyboard))
        
        try:
            report = analyse_backend(text, "intelligence", language=context.user_data.get('language', 'English'))
            await update.message.reply_text(f"📊 Document Intelligence:\n\n{report}")
        except Exception: pass

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    lang = context.user_data.get('language', 'English')

    if 'document' not in context.user_data:
        await update.message.reply_text("Phoenix is thinking...")
        await update.message.reply_text(ask_backend(question, lang, "telegram"))
        return

    intent = detect_intent(question)
    if intent == "summarise":
        await handle_summarise(update.message, context)
        return
    if intent == "quiz":
        await handle_quiz(update.message, context)
        return

    await update.message.reply_text("Scanning document index...")
    rag_store = context.user_data.get('rag')
    context_text = "\n\n---\n\n".join(retrieve_chunks(question, rag_store["chunks"], rag_store["index"], rag_store["doc_freq"]))
    await update.message.reply_text(analyse_backend(context_text, "question", question=question, language=lang))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    if query.data == "summarise": await handle_summarise(query.message, context)
    elif query.data == "quiz": await handle_quiz(query.message, context)
    elif query.data == "compare": await handle_compare_request(query.message, context)
    elif query.data == "how_to_use": await query.message.reply_text("Send Text, PDFs, Photos, or Audio Notes directly to interact.")

async def handle_summarise(msg, context):
    await msg.reply_text(analyse_backend(context.user_data['document'], "summarise", language=context.user_data.get('language', 'English')))

async def handle_quiz(msg, context):
    await msg.reply_text(analyse_backend(context.user_data['document'], "quiz", language=context.user_data.get('language', 'English')))

async def handle_compare_request(msg, context):
    await msg.reply_text(compare_backend(context.user_data['document'], context.user_data['document2'], language=context.user_data.get('language', 'English')))

async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("English", callback_data="lang_English"), InlineKeyboardButton("Yoruba", callback_data="lang_Yoruba")],
                [InlineKeyboardButton("Igbo", callback_data="lang_Igbo"), InlineKeyboardButton("Hausa", callback_data="lang_Hausa")]]
    await update.message.reply_text("Set output layout language:", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    lang = query.data.replace("lang_", "")
    context.user_data['language'] = lang
    await query.message.reply_text(f"Language configured to: {lang}")

if __name__ == "__main__":
    def run_server():
        class PingHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers()
                self.wfile.write(b"Phoenix Frontend Is Live")
            def log_message(self, format, *args): pass
        HTTPServer(("0.0.0.0", PORT), PingHandler).serve_forever()

    threading.Thread(target=run_server, daemon=True).start()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CallbackQueryHandler(handle_language_callback, pattern="^lang_"))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    print("Phoenix Docs Production Engine running smoothly...")
    app.run_polling()
