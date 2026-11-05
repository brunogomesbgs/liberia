
import asyncio
from tortoise import Tortoise
from app.tortoise_config import TORTOISE_ORM

async def apply_fix():
    config = TORTOISE_ORM.copy()
    config["connections"]["default"] = config["connections"]["default"].replace("@db:", "@localhost:")
    await Tortoise.init(config=config)
    conn = Tortoise.get_connection("default")
    
    print("Applying fix: Adding user_id column to chat_messages...")
    try:
        await conn.execute_query("ALTER TABLE chat_messages ADD COLUMN user_id UUID REFERENCES users(id);")
        print("Successfully added user_id column.")
    except Exception as e:
        print(f"Error adding column: {e}")

if __name__ == "__main__":
    asyncio.run(apply_fix())
