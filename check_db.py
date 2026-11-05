
import asyncio
from tortoise import Tortoise
from app.tortoise_config import TORTOISE_ORM

async def check_schema():
    config = TORTOISE_ORM.copy()
    config["connections"]["default"] = config["connections"]["default"].replace("@db:", "@localhost:")
    await Tortoise.init(config=config)
    conn = Tortoise.get_connection("default")
    
    # Check if it's sqlite or postgres
    db_type = conn.__class__.__name__
    print(f"Database type: {db_type}")
    
    if "Sqlite" in db_type:
        res = await conn.execute_query("SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_messages';")
        print("Schema for chat_messages:")
        print(res[0][0])
    else:
        # Postgres
        res = await conn.execute_query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users';")
        print("Columns in users:")
        for r in res:
            print(r)
        res = await conn.execute_query("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'chat_messages';")
        print("Columns in chat_messages:")
        for r in res:
            print(r)

if __name__ == "__main__":
    asyncio.run(check_schema())
