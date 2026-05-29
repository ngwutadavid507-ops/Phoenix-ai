import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq
import PyPDF2
import io
from dotenv import load_dotenv
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
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    context.user_data['document'] = text
    page_count = len(pdf_reader.pages)
    
    await update.message.reply_text(
        f"✅ Document loaded successfully\n\n"
        f"📊 Pages: {page_count}\n\n"
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
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are Phoenix Docs, part of the Phoenix AI platform. You are a powerful document analyst. Answer questions based only on the document provided. Be clear, precise and helpful."
            },
            {
                "role": "user",
                "content": f"Document:\n{document}\n\nQuestion:\n{question}"
            }
        ]
    )
    
    answer = response.choices[0].message.content
    await update.message.reply_text(answer)

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
