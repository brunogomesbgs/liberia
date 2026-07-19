from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator
import uuid6

class UUID7Field(fields.UUIDField):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_db_value(self, value, instance):
        if value is None:
            return None
        return str(value)

class BaseModel(models.Model):
    id = fields.UUIDField(pk=True, default=uuid6.uuid7)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        abstract = True

class User(BaseModel):
    username = fields.CharField(max_length=50, unique=True)
    password_hash = fields.CharField(max_length=128)
    
    class Meta:
        table = "users"

class Book(BaseModel):
    title = fields.CharField(max_length=255)
    author = fields.CharField(max_length=255)
    content = fields.TextField()
    vector = fields.TextField(null=True) # Will store vector as string for now, pgvector handled via raw SQL or custom field
    user = fields.ForeignKeyField("models.User", related_name="books", null=True)

    class Meta:
        table = "books"

class AuditLog(BaseModel):
    event_type = fields.CharField(max_length=50) # e.g., "BOOK_CREATED", "BOOK_UPDATED"
    entity_id = fields.UUIDField()
    payload = fields.JSONField()
    user_id = fields.UUIDField(null=True)

    class Meta:
        table = "audit_logs"

User_Pydantic = pydantic_model_creator(User, name="User")
UserIn_Pydantic = pydantic_model_creator(User, name="UserIn", exclude_readonly=True)
Book_Pydantic = pydantic_model_creator(Book, name="Book")
BookIn_Pydantic = pydantic_model_creator(Book, name="BookIn", exclude_readonly=True)
