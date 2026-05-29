import os
import io
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
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

client = Groq(api_key=GROQ_API_KEY)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔥 Welcome to Phoenix Docs\n\n"
        "Part of the Phoenix AI platform\n\n"
        "What I can do:\n"
        "→ Read any PDF you send me\n"
        "→ Answer any question about it\n"
        "→ Summarise documents instantly\n\n"
        "📌 Note: Analyses first 20 pages\n\n"
        "Send me a PDF to get started."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Phoenix Docs Help\n\n"
        "1. Send any PDF document\n"
        "2. Wait for confirmation\n"
        "3. Ask any question about it\n\n"
        "Commands:\n"
        "/start - Welcome message\n"
        "/help - This help message\n"
        "/clear - Clear current document"
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🗑️ Document cleared.\n"
        "Send a new PDF to continue."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📄 Phoenix Docs is reading your document...")

    file = await update.message.document.get_file()
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
    context.user_data['document'] = text

    pages_analysed = min(total_pages, 20)

    await update.message.reply_text(
        f"✅ Document loaded successfully\n\n"
        f"📊 Total pages: {total_pages}\n"
        f"🔍 Pages analysed: {pages_analysed}\n\n"
        f"Now ask me anything about it."
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'document' not in context.user_data:
        await update.message.reply_text(
            "⚠️ No document loaded.\n"
            "Please send a PDF first."
        )
        return

    question = update.message.text
    document = context.user_data['document']

    await update.message.reply_text("🔍 Phoenix Docs is analysing...")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are Phoenix Docs, part of the Phoenix AI platform. You are a powerful document analyst. Answer questions based only on the document provided. Be clear, precise and helpful. Keep answers under 500 words."
                },
                {
                    "role": "user",
                    "content": f"Document:\n{document}\n\nQuestion:\n{question}"
                }
            ],
            max_tokens=800,
            timeout=25
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        await update.message.reply_text(
            "⚠️ Analysis timed out. Please try again.\n"
            "If the document is large try asking a simpler question."
        )

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    print("🔥 Phoenix Docs is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
