"""Object-mapped access to chat sessions, messages, and images."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select

from agent import db
from agent.models import ChatImage, ChatMessage, ChatSession


def _updated_at(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _replace_sync(messages: list[dict[str, Any]], session_id: str) -> None:
    from agent.history import apply_image_storage, session_title

    with db.orm_session() as session:
        chat = session.get(ChatSession, session_id)
        now = datetime.now(timezone.utc)
        if chat is None:
            chat = ChatSession(
                session_id=session_id, title=session_title(messages), updated_at=now
            )
            session.add(chat)
        else:
            chat.title = session_title(messages)
            chat.updated_at = now
        session.flush()

        def save_image(mime: str, data: bytes) -> int:
            image = ChatImage(session_id=session_id, mime=mime, data=data)
            session.add(image)
            session.flush()
            return int(image.id)

        def image_belongs(image_id: int) -> bool:
            image = session.get(ChatImage, image_id)
            return image is not None and image.session_id == session_id

        stored, keep_ids = apply_image_storage(
            messages,
            save_image=save_image,
            image_belongs=image_belongs,
        )
        session.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        for index, message in enumerate(stored):
            session.add(
                ChatMessage(
                    session_id=session_id,
                    seq=index,
                    role=str(message["role"]),
                    content=json.dumps(message["content"], ensure_ascii=False),
                )
            )
        unwanted = delete(ChatImage).where(ChatImage.session_id == session_id)
        if keep_ids:
            unwanted = unwanted.where(ChatImage.id.not_in(list(keep_ids)))
        session.execute(unwanted)


def _list_sync(session_id: str) -> list[dict[str, Any]]:
    from agent.history import message_from_row

    with db.orm_session() as session:
        rows = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.seq.asc(), ChatMessage.id.asc())
        ).all()
        return [
            message_from_row(
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "created_at": row.created_at,
                }
            )
            for row in rows
        ]


def _list_sessions_sync() -> list[dict[str, Any]]:
    with db.orm_session() as session:
        rows = session.scalars(
            select(ChatSession).order_by(
                ChatSession.updated_at.desc(),
                ChatSession.session_id.desc(),
            )
        ).all()
        return [
            {
                "id": row.session_id,
                "title": row.title,
                "updatedAt": _updated_at(row.updated_at),
            }
            for row in rows
        ]


def _delete_session_sync(session_id: str) -> None:
    with db.orm_session() as session:
        chat = session.get(ChatSession, session_id)
        if chat is not None:
            session.delete(chat)


def _get_image_sync(image_id: int) -> tuple[str, bytes] | None:
    from agent.history import _bytes_of

    with db.orm_session() as session:
        image = session.get(ChatImage, image_id)
        if image is None:
            return None
        return image.mime, _bytes_of(image.data)


def _image_data_url_sync(image_id: int, session_id: str) -> str | None:
    from agent.history import _bytes_of, data_url

    with db.orm_session() as session:
        image = session.get(ChatImage, image_id)
        if image is None or image.session_id != session_id:
            return None
        return data_url(image.mime, _bytes_of(image.data))
