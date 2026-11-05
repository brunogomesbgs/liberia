from pydantic import BaseModel as PydanticBaseModel
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from tortoise.contrib.fastapi import register_tortoise
from app.models import User, User_Pydantic, Book, ChatSession, ChatMessage, ChatSession_Pydantic
from app.tortoise_config import TORTOISE_ORM
from app.auth import verify_password, get_password_hash, create_access_token, decode_access_token
from app.agent import agent_app
from app.sentiment import analyze_sentiment
from app.rag import init_vector_extension
from aerich import Command
from app.cqrs import BookCommandHandler, BookQueryHandler, CreateBookCommand
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections and self.active_connections[session_id] == websocket:
            del self.active_connections[session_id]

    async def send_personal_message(self, message: str, session_id: str, sender: str = "admin"):
        # Ensure session_id is a string
        session_id = str(session_id)
        if session_id in self.active_connections:
            payload = {
                "sender": sender,
                "content": message
            }
            try:
                await self.active_connections[session_id].send_text(json.dumps(payload))
            except Exception as e:
                print(f"Error sending message to {session_id}: {e}")
                # We need the websocket here too, but for now we just remove whatever is there if it failed
                # Actually, let's be safe and not disconnect here unless we know which one it is
                # or just let the websocket_endpoint handle the disconnect via its try-except
                pass
        else:
            print(f"DEBUG: session_id {session_id} not found in active_connections. Current keys: {list(self.active_connections.keys())}")

manager = ConnectionManager()

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

# Tortoise ORM configuration (Self-healing strategy with Aerich)
register_tortoise(
    app,
    config=TORTOISE_ORM,
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

async def require_admin(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user does not have enough privileges"
        )
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.get_or_none(username=form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        print(f"Login failed for user: {form_data.username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=User_Pydantic)
async def create_user(request: Request, x_origin: Optional[str] = Header(None)):
    user_data = await request.json()
    role = "standard"
    if x_origin == "admin-ui":
        role = "admin"
    elif user_data.get("role"):
        role = user_data.get("role")

    username = user_data.get("username")
    password = user_data.get("password") or user_data.get("password_hash")
    
    if not username or not password:
         raise HTTPException(status_code=422, detail="Username and Password are required")

    user_obj = await User.create(
        username=username,
        password_hash=get_password_hash(password),
        role=role
    )
    return await User_Pydantic.from_tortoise_orm(user_obj)

@app.get("/books")
async def list_books(user: User = Depends(get_current_user)):
    books = await BookQueryHandler.list_books(user_id=str(user.id))
    return books

@app.post("/books")
async def create_book(book_in: CreateBookCommand, user: User = Depends(get_current_user)):
    book_in.user_id = str(user.id)
    book = await BookCommandHandler.create_book(book_in)
    return book

@app.get("/admin/users")
async def admin_list_users(user: User = Depends(require_admin)):
    return await User.all()

@app.get("/admin/books")
async def admin_list_books(user: User = Depends(require_admin)):
    return await Book.all()

@app.get("/admin/chats/requested")
async def list_requested_chats(user: User = Depends(require_admin)):
    sessions = await ChatSession.filter(is_human_required=True, is_active=True).prefetch_related('book').order_by("-created_at")
    result = []
    for s in sessions:
        eval_count = await ChatMessage.filter(session=s, feedback__isnull=False).count()
        session_data = await ChatSession_Pydantic.from_tortoise_orm(s)
        session_dict = session_data.dict()
        session_dict["evaluation_count"] = eval_count
        session_dict["book_title"] = s.book.title if s.book else "General Chat"
        result.append(session_dict)
    return result

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket, bookId: str = None):
    # We'll use the token passed in the first message for auth
    session_id_val = None
    try:
        await websocket.accept()
        token_msg = await websocket.receive_text()
        token_data = decode_access_token(token_msg)
        if not token_data:
            await websocket.send_text("Authentication failed")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        user = await User.get_or_none(username=token_data.username)
        if not user:
            await websocket.send_text("User not found")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.send_text("Authenticated")
        
        # Create or get active session
        session = await ChatSession.filter(user=user, book=await Book.get_or_none(id=bookId) if bookId else None, is_active=True).order_by("-created_at").first()
        if not session:
            session = await ChatSession.create(user=user, book=await Book.get_or_none(id=bookId) if bookId else None)
        
        session_id_val = str(session.id)
        # Clear any existing connections for this session first to avoid ghosts
        if session_id_val in manager.active_connections:
            try:
                await manager.active_connections[session_id_val].close()
            except:
                pass
        manager.active_connections[session_id_val] = websocket
        
        while True:
            data = await websocket.receive_text()
            
            # Check if it's feedback
            try:
                payload = json.loads(data)
                if payload.get("type") == "feedback":
                    msg_index = payload.get("messageIndex")
                    feedback_val = payload.get("feedback")
                    
                    # Find the message by index in this session
                    # The UI sends the index of the message in the current local messages array
                    messages = await ChatMessage.filter(session=session).order_by("created_at")
                    if 0 <= msg_index < len(messages):
                        target_msg = messages[msg_index]
                        target_msg.feedback = feedback_val
                        await target_msg.save()
                        print(f"DEBUG: Feedback updated for message {target_msg.id}: {feedback_val}")
                    continue
            except (json.JSONDecodeError, TypeError):
                # Not JSON or not a dict, proceed as normal message
                pass

            # Save user message
            await ChatMessage.create(session=session, sender_type="user", content=data, user=user)

            # If human is already required, don't invoke agent
            await session.refresh_from_db()
            if session.is_human_required:
                print(f"DEBUG: Session {session.id} is already flagged for human. Skipping agent.")
                await websocket.send_text("Human assistant has been notified. They will respond shortly.")
                continue

            # Invoke LangGraph agent
            response = await agent_app.ainvoke({
                "messages": [HumanMessage(content=data)],
                "book_id": bookId,
                "human_required": False
            })
            
            last_msg = response["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            
            # Save agent message
            await ChatMessage.create(session=session, sender_type="agent", content=content)
            
            # Update human requested status
            if response.get("human_required"):
                session.is_human_required = True
                
                # Perform sentiment analysis on the whole session
                all_messages = await ChatMessage.filter(session=session).order_by("created_at")
                message_contents = [m.content for m in all_messages]
                session.sentiment = analyze_sentiment(message_contents)
                
                await session.save()
                print(f"DEBUG: Session {session.id} flagged for human intervention. Sentiment: {session.sentiment}")

            await websocket.send_text(content)
    except WebSocketDisconnect:
        if session_id_val:
            manager.disconnect(session_id_val, websocket)
    except Exception as e:
        print(f"Error in user websocket: {e}")
        if session_id_val:
            manager.disconnect(session_id_val, websocket)

@app.get("/admin/chat/{session_id}/history")
async def get_chat_history(session_id: str, user: User = Depends(require_admin)):
    session = await ChatSession.get_or_none(id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await ChatMessage.filter(session=session).order_by("created_at")
    return messages

@app.websocket("/admin/chat/{session_id}")
async def admin_websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        token_msg = await websocket.receive_text()
        token_data = decode_access_token(token_msg)
        if not token_data:
            await websocket.send_text("Authentication failed")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        user = await User.get_or_none(username=token_data.username)
        if not user or user.role != "admin":
            await websocket.send_text("Forbidden")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await websocket.send_text("Authenticated")
        
        session = await ChatSession.get_or_none(id=session_id)
        if not session:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        while True:
            data = await websocket.receive_text()
            # Save admin message
            await ChatMessage.create(session=session, sender_type="admin", content=data, user=user)
            # Notify the user's websocket here.
            await manager.send_personal_message(data, session_id, sender="admin")
            await websocket.send_text(f"Admin sent: {data}")
    except WebSocketDisconnect:
        pass

class AdminResponse(PydanticBaseModel):
    message: str

@app.post("/admin/chat/{session_id}/respond")
async def admin_respond(session_id: str, response: AdminResponse, user: User = Depends(require_admin)):
    session = await ChatSession.get_or_none(id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await ChatMessage.create(session=session, sender_type="admin", content=response.message, user=user)
    # Reset human required flag after responding
    session.is_human_required = False
    await session.save()
    # Notify the user's websocket here.
    await manager.send_personal_message(response.message, session_id, sender="admin")
    return {"status": "success"}

@app.on_event("startup")
async def startup():
    # Self-healing database: Run migrations
    try:
        command = Command(tortoise_config=TORTOISE_ORM, app="models")
        await command.init()
        await command.upgrade()
        print("Database migrations applied successfully")
    except Exception as e:
        print(f"Error applying migrations: {e}")
        # If it's the first time, we might need to init-db
        try:
            await command.init_db(safe=True)
            print("Database initialized successfully")
        except Exception as e2:
            print(f"Error initializing database: {e2}")

    await init_vector_extension()

@app.get("/")
async def root():
    return {"message": "Welcome to Liberia API"}
