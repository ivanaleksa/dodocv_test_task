from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# тип для цвета
BgrColor = tuple[int, int, int]


def _parse_bgr(value: str) -> BgrColor:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"Цвет должен быть в формате 'B,G,R', получено: {value!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
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

    output_video: str = "output.mp4"
    report_file: str = "report.txt"
    events_csv: str = "events.csv"

    color_empty: BgrColor = Field(default=(0, 220, 0))
    color_occupied: BgrColor = Field(default=(0, 0, 220))
    color_approach: BgrColor = Field(default=(0, 200, 255))

    @field_validator(
        "color_empty",
        "color_occupied",
        "color_approach",
        mode="before",
    )
    @classmethod
    def parse_color(cls, v):
        if isinstance(v, str):
            return _parse_bgr(v)
        return v


settings = Settings()
