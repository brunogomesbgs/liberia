from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from tortoise.contrib.fastapi import register_tortoise
from .models import User, User_Pydantic, UserIn_Pydantic, Book_Pydantic
from .auth import verify_password, get_password_hash, create_access_token, decode_access_token
from .agent import agent_app
from .rag import init_vector_extension
from .cqrs import BookCommandHandler, BookQueryHandler, CreateBookCommand
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv(override=True)

app = FastAPI(title="Liberia Agentic Book Reader")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for now to avoid UI connectivity issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tortoise ORM configuration (Self-healing strategy with generate_schemas=True)
DATABASE_URL = os.getenv("DATABASE_URL", "postgres://user:password@localhost:5432/liberia")

register_tortoise(
    app,
    db_url=DATABASE_URL,
    modules={"models": ["app.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    token_data = decode_access_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await User.get_or_none(username=token_data.username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.get_or_none(username=form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=User_Pydantic)
async def create_user(user: UserIn_Pydantic):
    user_obj = await User.create(
        username=user.username,
        password_hash=get_password_hash(user.password_hash)
    )
    return await User_Pydantic.from_tortoise_orm(user_obj)

@app.get("/books", response_model=list[Book_Pydantic])
async def list_books(user: User = Depends(get_current_user)):
    books = await BookQueryHandler.list_books(user_id=str(user.id))
    return [await Book_Pydantic.from_tortoise_orm(b) for b in books]

@app.post("/books", response_model=Book_Pydantic)
async def create_book(book_in: CreateBookCommand, user: User = Depends(get_current_user)):
    book_in.user_id = str(user.id)
    book = await BookCommandHandler.create_book(book_in)
    return await Book_Pydantic.from_tortoise_orm(book)

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket, bookId: str = None):
    await websocket.accept()
    # Initial authentication via token in message or query param
    # For simplicity, we'll wait for a token message first
    try:
        token_msg = await websocket.receive_text()
        token_data = decode_access_token(token_msg)
        if not token_data:
            await websocket.send_text("Authentication failed")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await websocket.send_text("Authenticated")
        
        while True:
            data = await websocket.receive_text()
            # Invoke LangGraph agent with bookId context
            # Handle potential dictionary response from agent_app
            response = await agent_app.ainvoke({
                "messages": [HumanMessage(content=data)],
                "book_id": bookId
            })
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content"):
                content = last_msg.content
            else:
                content = str(last_msg)
            await websocket.send_text(content)
    except WebSocketDisconnect:
        pass

@app.on_event("startup")
async def startup():
    await init_vector_extension()

@app.get("/")
async def root():
    return {"message": "Welcome to Liberia API"}
