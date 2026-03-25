from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# helping func - string "B,G,R" → tuple[int, int, int]
def _parse_bgr(value: object) -> tuple[int, int, int]:
    if isinstance(value, (tuple, list)):
        b, g, r = value
        return (int(b), int(g), int(r))
    parts = str(value).split(",")
    if len(parts) != 3:
        raise ValueError(f"Цвет должен быть в формате 'B,G,R', получено: {value!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


BgrColor = Annotated[tuple[int, int, int], BeforeValidator(_parse_bgr)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    yolo_model: str = Field(
        default="yolov8n.pt",
        description="Имя или путь к файлу весов",
    )
    person_class_id: int = Field(
        default=0,
        description="ID класса 'person', для yolo 0",
    )
    conf_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Минимальная уверенность детекции модели",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Порог iou для nms внутри модели",
    )

    empty_frames_needed: int = Field(
        default=20,
        gt=0,
        description="Кадров подряд без человека -> стол считается пустым",
    )
    occupied_frames_needed: int = Field(
        default=8,
        gt=0,
        description="Кадров подряд с человеком -> стол считается занятым",
    )

    output_video: str = Field(default="output.mp4")
    report_file: str = Field(default="report.txt")
    events_csv: str = Field(default="events.csv")

    color_empty: BgrColor = Field(default=(0, 220, 0))
    color_occupied: BgrColor = Field(default=(0, 0, 220))
    color_approach: BgrColor = Field(default=(0, 200, 255))


settings = Settings()
