import base64
from datetime import datetime

import pytest

from agent.history import (
    apply_image_storage,
    decode_data_image,
    message_from_row,
    session_title,
)

PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_message_from_row_decodes_text_and_image_urls() -> None:
    message = message_from_row(
        {
            "id": 3,
            "role": "user",
            "content": (
                '[{"type": "text", "text": "这是什么"}, '
                '{"type": "image", "image": "/aiapi/images/9"}]'
            ),
            "created_at": datetime(2026, 10, 30, 6, 0, 0),
        }
    )

    assert message == {
        "id": "3",
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么"},
            {"type": "image", "image": "/aiapi/images/9"},
        ],
        "createdAt": "2026-10-30T06:00:00",
    }


def test_session_title_uses_the_first_user_message() -> None:
    assert (
        session_title([{"role": "user", "content": "  你好   世界  "}]) == "你好 世界"
    )
    assert (
        session_title(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "/aiapi/images/1"}],
                }
            ]
        )
        == "图片"
    )


def test_apply_image_storage_replaces_data_urls() -> None:
    saved: list[tuple[str, bytes]] = []

    def save_image(mime: str, data: bytes) -> int:
        saved.append((mime, data))
        return 9

    messages, keep = apply_image_storage(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是什么"},
                    {"type": "image", "image": PNG},
                ],
            }
        ],
        save_image=save_image,
        image_belongs=lambda _image_id: False,
    )

    assert keep == {9}
    assert saved[0][0] == "image/png"
    assert saved[0][1] == base64.b64decode(PNG.split(",", 1)[1])
    assert messages[0]["content"] == [
        {"type": "text", "text": "这是什么"},
        {"type": "image", "image": "/aiapi/images/9"},
    ]


def test_apply_image_storage_keeps_an_existing_image() -> None:
    messages, keep = apply_image_storage(
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": "/aiapi/images/4"}],
            }
        ],
        save_image=lambda _mime, _data: 0,
        image_belongs=lambda image_id: image_id == 4,
    )

    assert keep == {4}
    assert messages[0]["content"] == [{"type": "image", "image": "/aiapi/images/4"}]


def test_apply_image_storage_rejects_an_unknown_image() -> None:
    with pytest.raises(ValueError, match="图片不存在"):
        apply_image_storage(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "/aiapi/images/4"}],
                }
            ],
            save_image=lambda _mime, _data: 0,
            image_belongs=lambda _image_id: False,
        )


def test_decode_data_image_rejects_an_unsupported_type() -> None:
    with pytest.raises(ValueError, match="不支持的图片格式"):
        decode_data_image("data:image/bmp;base64,Qk0=")
