import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, FSInputFile, InputMediaPhoto
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode
from aiogram.utils.media_group import MediaGroupBuilder
from run import run
# from dotenv import load_dotenv

# load_dotenv()

bot = Bot(token=os.environ.get("BOT_TOKEN"))
dp = Dispatcher()

@dp.message(F.text, CommandStart())
async def start(message: Message):
	await bot.send_message(chat_id=message.chat.id,
						   text=f'Этот бот генерирует 3D сцену из текстового описания. Введите текст, для обучения:',
						   parse_mode=ParseMode.HTML)

@dp.message(F.text)
async def start(message: Message):
	asyncio.create_task(run(message.text))
	await bot.send_message(chat_id=message.chat.id,
						   text=f"Промпт {message.text} принят для обучения.",
						   parse_mode=ParseMode.HTML)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())