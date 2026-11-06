from pydantic import BaseModel as PydanticBaseModel
from typing import List, Optional
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
    Header,
    Request,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from tortoise.contrib.fastapi import register_tortoise
from tortoise.functions import Count
from tortoise.expressions import Q
from app.models import (
    User,
    User_Pydantic,
    Book,
    BookChunk,
    ChatSession,
    ChatMessage,
    ChatSession_Pydantic,
)
from app.tortoise_config import TORTOISE_ORM
from app.auth import verify_password, get_password_hash, create_access_token, decode_access_token
from app.agent import agent_app
from app.sentiment import analyze_sentiment
from app.rag import init_vector_extension, _validate_uuid
from app.evaluation import get_online_evaluation_summary, run_retrieval_benchmark
from aerich import Command
from app.cqrs import (
    BookCommandHandler,
    BookQueryHandler,
    CreateBookCommand,
    UpdateBookCommand,
)
from langchain_core.messages import HumanMessage, AIMessage
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

load_dotenv(override=False)

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
if os.getenv("TESTING") != "1":
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

    existing_user = await User.get_or_none(username=username)
    if existing_user:
        return await User_Pydantic.from_tortoise_orm(existing_user)

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

@app.get("/books/{book_id}")
async def get_book(book_id: str, user: User = Depends(get_current_user)):
    try:
        validated_id = _validate_uuid(book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book ID")
    book = await BookQueryHandler.get_book(
        validated_id,
        user_id=None if user.role == "admin" else str(user.id),
    )
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.get("/books/{book_id}/chunks")
async def get_book_chunks(book_id: str, user: User = Depends(get_current_user)):
    try:
        validated_id = _validate_uuid(book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book ID")
    book = await BookQueryHandler.get_book(
        validated_id,
        user_id=None if user.role == "admin" else str(user.id),
    )
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    chunks = await BookQueryHandler.list_book_chunks(validated_id)
    return [
        {
            "id": str(c.id),
            "book_id": str(c.book_id),
            "chunk_index": c.chunk_index,
            "chapter": c.chapter,
            "page_number": c.page_number,
            "token_count": c.token_count,
            "content": c.content,
        }
        for c in chunks
    ]

@app.post("/books")
async def create_book(book_in: CreateBookCommand, user: User = Depends(get_current_user)):
    book_in.user_id = str(user.id)
    book = await BookCommandHandler.create_book(book_in)
    return book

@app.post("/books/upload")
async def upload_book(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    filename = file.filename or "uploaded_book.txt"
    allowed_exts = (".pdf", ".epub", ".txt", ".md", ".markdown")
    if not filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_exts)}",
        )
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    book = await BookCommandHandler.ingest_uploaded_book(
        file_bytes=file_bytes,
        filename=filename,
        title=title,
        author=author,
        user_id=str(user.id),
    )
    return book

class UpdateBookPayload(PydanticBaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    content: Optional[str] = None

@app.put("/books/{book_id}")
async def update_book(
    book_id: str,
    payload: UpdateBookPayload,
    user: User = Depends(get_current_user),
):
    try:
        validated_id = _validate_uuid(book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book ID")
    existing = await BookQueryHandler.get_book(
        validated_id,
        user_id=None if user.role == "admin" else str(user.id),
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Book not found")

    cmd_data = {"id": validated_id, "user_id": str(user.id)}
    if payload.title is not None:
        cmd_data["title"] = payload.title
    if payload.author is not None:
        cmd_data["author"] = payload.author
    if payload.content is not None:
        cmd_data["content"] = payload.content

    updated = await BookCommandHandler.update_book(UpdateBookCommand(**cmd_data))
    return updated

@app.get("/admin/users")
async def admin_list_users(user: User = Depends(require_admin)):
    return await User.all()

@app.get("/admin/books")
async def admin_list_books(user: User = Depends(require_admin)):
    return await Book.all().order_by("-created_at")

@app.get("/admin/chats/requested")
async def list_requested_chats(user: User = Depends(require_admin)):
    sessions = (
        await ChatSession.filter(is_human_required=True, is_active=True)
        .annotate(
            evaluation_count=Count(
                "messages",
                _filter=Q(messages__feedback__not_isnull=True),
            )
        )
        .prefetch_related("book")
        .order_by("-created_at")
    )
    result = []
    for s in sessions:
        session_data = await ChatSession_Pydantic.from_tortoise_orm(s)
        session_dict = (
            session_data.model_dump()
            if hasattr(session_data, "model_dump")
            else session_data.dict()
        )
        session_dict["evaluation_count"] = getattr(s, "evaluation_count", 0)
        session_dict["book_title"] = s.book.title if s.book else "General Chat"
        result.append(session_dict)
    return result

class BenchmarkRequest(PydanticBaseModel):
    k: int = 5
    book_id: Optional[str] = None
    max_samples: int = 10

@app.get("/admin/evaluation/summary")
async def admin_evaluation_summary(user: User = Depends(require_admin)):
    return await get_online_evaluation_summary()

@app.post("/admin/evaluation/benchmark")
async def admin_evaluation_benchmark(
    req: Optional[BenchmarkRequest] = None,
    user: User = Depends(require_admin),
):
    k = req.k if req else 5
    book_id = req.book_id if req else None
    max_samples = req.max_samples if req else 10
    return await run_retrieval_benchmark(k=k, book_id=book_id, max_samples=max_samples)

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

        validated_book_id = None
        book_obj = None
        if bookId:
            try:
                validated_book_id = _validate_uuid(bookId)
                book_obj = await Book.get_or_none(id=validated_book_id)
            except ValueError:
                validated_book_id = None
                book_obj = None

        # Create or get active session (single DB lookup for book_obj)
        session = (
            await ChatSession.filter(user=user, book=book_obj, is_active=True)
            .order_by("-created_at")
            .first()
        )
        if not session:
            session = await ChatSession.create(user=user, book=book_obj)
        
        session_id_val = str(session.id)
        # Clear any existing connections for this session first to avoid ghosts
        if session_id_val in manager.active_connections:
            try:
                await manager.active_connections[session_id_val].close()
            except Exception:
                pass
        manager.active_connections[session_id_val] = websocket
        
        while True:
            data = await websocket.receive_text()
            
            # Check if it's feedback
            try:
                payload = json.loads(data)
                if isinstance(payload, dict) and payload.get("type") == "feedback":
                    msg_id = payload.get("messageId")
                    msg_index = payload.get("messageIndex")
                    total_client_msgs = payload.get("totalMessages")
                    feedback_val = payload.get("feedback")

                    target_msg = None
                    if msg_id:
                        target_msg = await ChatMessage.get_or_none(id=msg_id, session=session)
                    elif isinstance(msg_index, int):
                        messages = await ChatMessage.filter(session=session).order_by("created_at")
                        if (
                            isinstance(total_client_msgs, int)
                            and 0 < total_client_msgs <= len(messages)
                            and 0 <= msg_index < total_client_msgs
                        ):
                            window = messages[-total_client_msgs:]
                            target_msg = window[msg_index]
                        elif 0 <= msg_index < len(messages):
                            target_msg = messages[msg_index]

                    if target_msg is not None:
                        target_msg.feedback = feedback_val
                        await target_msg.save(update_fields=["feedback", "updated_at"])
                        print(f"DEBUG: Feedback updated for message {target_msg.id}: {feedback_val}")
                    continue
            except (json.JSONDecodeError, TypeError):
                # Not JSON or not a dict, proceed as normal message
                pass

            # Save user message
            await ChatMessage.create(session=session, sender_type="user", content=data, user=user)

            # Refresh session state from DB to see if an admin cleared is_human_required
            await session.refresh_from_db()
            if session.is_human_required:
                print(f"DEBUG: Session {session.id} is already flagged for human. Skipping agent.")
                await websocket.send_text("Human assistant has been notified. They will respond shortly.")
                continue

            # Load recent conversation history for stateful multi-turn context
            recent_db_msgs = (
                await ChatMessage.filter(session=session)
                .order_by("-created_at")
                .limit(6)
            )
            history_messages = [
                HumanMessage(content=m.content)
                if m.sender_type == "user"
                else AIMessage(content=m.content)
                for m in reversed(recent_db_msgs)
            ]
            if not history_messages:
                history_messages = [HumanMessage(content=data)]

            # Invoke LangGraph agent with tenant user_id & conversation history
            response = await agent_app.ainvoke({
                "messages": history_messages,
                "book_id": validated_book_id,
                "user_id": str(user.id),
                "human_required": False,
            })
            
            last_msg = response["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            
            # Save agent message with retrieval telemetry
            await ChatMessage.create(
                session=session,
                sender_type="agent",
                content=content,
                retrieved_chunk_ids=response.get("retrieved_chunk_ids"),
                top_similarity_score=response.get("top_similarity_score"),
            )
            
            # Update human requested status
            if response.get("human_required"):
                session.is_human_required = True
                
                # Perform sentiment analysis on recent session messages
                all_messages = await ChatMessage.filter(session=session).order_by("-created_at").limit(50)
                message_contents = [m.content for m in reversed(all_messages)]
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
            # Clear is_human_required lock when admin responds via WebSocket
            await session.refresh_from_db()
            if session.is_human_required:
                session.is_human_required = False
                await session.save(update_fields=["is_human_required", "updated_at"])
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
    if os.getenv("TESTING") == "1":
        return
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
