"""Mapped chat tables. Queries go through these objects instead of SQL strings."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Registry for the chat tables."""


class ChatSession(Base):
    """One conversation, with the title stored beside the messages."""

    __tablename__ = "chat_session"
    __table_args__ = (Index("idx_chat_session_updated", "updated_at"),)

    session_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChatImage(Base):
    """Image bytes for one session. Messages store only the image id."""

    __tablename__ = "chat_image"
    __table_args__ = (Index("idx_chat_image_session", "session_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_session.session_id", ondelete="CASCADE"), nullable=False
    )
    mime: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChatMessage(Base):
    """One stored turn. Image parts point at ``/aiapi/images/{id}``."""

    __tablename__ = "chat_message"
    __table_args__ = (Index("idx_chat_message_session_seq", "session_id", "seq"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_session.session_id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
