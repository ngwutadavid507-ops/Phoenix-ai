import os
import io
import threading
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

client = Groq(api_key=GROQ_API_KEY)

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        "→ 🌍 Respond in your language\n\n"
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
            "/clear - Clear current document\n\n"
            "Or just ask any question about your document!"
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
        "/clear - Clear current document"
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
    doc = update.message.document

    if not doc.file_name.lower().endswith('.pdf'):
        await update.message.reply_text(
            "⚠️ Please send a PDF file only."
        )
        return

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
            f"✅ Document loaded successfully\n\n"
            f"📊 Total pages: {total_pages}\n"
            f"🔍 Pages analysed: {pages_analysed}\n\n"
            f"What would you like to do?",
            reply_markup=reply_markup
        )

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
            f"You are Phoenix Docs. Generate quiz questions from the document. Respond in {lang}.",
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
        await message.reply_text(
            "📄 Send me the second PDF to compare with."
        )
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

async def summarise_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_summarise(update.message, context)

async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_quiz(update.message, context)

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_compare_request(update.message, context)

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'document' not in context.user_data:
        await update.message.reply_text(
            "⚠️ No document loaded.\n"
            "Please send a PDF first."
        )
        return

    question = update.message.text
    document = context.user_data['document']
    lang = context.user_data.get('language', 'English')

    await update.message.reply_text("🔍 Phoenix Docs is analysing...")

    try:
        answer = ask_groq(
            f"You are Phoenix Docs. Answer questions based only on the document. "
            f"Be clear and helpful. Keep answers under 500 words. Respond in {lang}.",
            f"Document:\n{document}\n\nQuestion:\n{question}"
        )
        await update.message.reply_text(answer)
    except Exception:
        await update.message.reply_text(
            "⚠️ Analysis timed out. Please try again."
        )

async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    lang = query.data.replace("lang_", "")
    context.user_data['language'] = lang

    await query.message.reply_text(
        f"✅ Language set to {lang}\n\n"
        f"All responses will now be in {lang}."
    )

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

    print("🔥 Phoenix Docs is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
