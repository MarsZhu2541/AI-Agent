from datetime import datetime

from agent.history import message_from_row


def test_message_from_row_decodes_text_and_images() -> None:
    message = message_from_row(
        {
            "id": 3,
            "role": "user",
            "content": (
                '[{"type": "text", "text": "这是什么"}, '
                '{"type": "image", "image": "data:image/png;base64,aa"}]'
            ),
            "created_at": datetime(2026, 10, 30, 6, 0, 0),
        }
    )

    assert message == {
        "id": "3",
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么"},
            {"type": "image", "image": "data:image/png;base64,aa"},
        ],
        "createdAt": "2026-10-30T06:00:00",
    }
