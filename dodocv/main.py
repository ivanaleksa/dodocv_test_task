import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

_COLOR_TEXT = (255, 255, 255)
_COLOR_SHADOW = (0, 0, 0)


# model loading-------------------------------------------------------
def load_yolo() -> YOLO:
    model_path = settings.yolo_model
    try:
        model = YOLO(model_path)
        logger.info("YOLO модель загружена: %s", model_path)
        return model
    except Exception as exc:
        logger.error(
            "Не удалось загрузить модель '%s'.\nПричина: %s\n", model_path, exc
        )
        sys.exit(1)


# persons detection-------------------------------------------------------
def detect_persons(model: YOLO, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect persons in a frame.
    Returns a list of tuples (xmin, ymin, xmax, ymax).
    """
    results = model(
        frame,
        conf=settings.conf_threshold,
        iou=settings.iou_threshold,
        classes=[settings.person_class_id],
        verbose=False,
    )
    boxes: list[tuple[int, int, int, int]] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    return boxes


# geometry-------------------------------------------------------
def _iou_box_roi(
    box: tuple[int, int, int, int],
    roi: tuple[int, int, int, int],
) -> float:
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi

    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (
        max(1, (bx2 - bx1) * (by2 - by1)) + max(1, (rx2 - rx1) * (ry2 - ry1)) - inter
    )
    return inter / union if union > 0 else 0.0


def person_in_roi(
    boxes: list[tuple[int, int, int, int]],
    roi: tuple[int, int, int, int],
    min_overlap: float = 0.10,
) -> bool:
    """
    True if at least one person is in ROI.
    At least one of two conditions must be met:
      - center of bbox is in ROI
      - IoU(bbox, ROI) ≥ min_overlap
    """
    rx1, ry1, rx2, ry2 = roi
    for box in boxes:
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        center_inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        if center_inside or _iou_box_roi(box, roi) >= min_overlap:
            return True
    return False


# visualization-------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    state: str,
    event_label: str,
    fps: float,
    frame_no: int,
) -> np.ndarray:
    out = frame.copy()
    rx1, ry1, rx2, ry2 = roi

    color_map: dict[str, tuple[tuple[int, int, int], str]] = {
        "empty": (settings.color_empty, "EMPTY"),
        "approach": (settings.color_approach, "APPROACH"),
        "occupied": (settings.color_occupied, "OCCUPIED"),
    }
    color, state_label = color_map.get(state, (_COLOR_TEXT, state.upper()))

    # ROI boundaries
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), color, 3)

    # Boundary title
    font, fs, th = cv2.FONT_HERSHEY_DUPLEX, 0.75, 2
    lx, ly = rx1, ry1 - 8
    cv2.putText(out, state_label, (lx + 1, ly + 1), font, fs, _COLOR_SHADOW, th + 1)
    cv2.putText(out, state_label, (lx, ly), font, fs, color, th)

    # status bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # time
    seconds = frame_no / max(fps, 1)
    tc = f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
    cv2.putText(
        out, f"Time: {tc}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COLOR_TEXT, 2
    )

    # event
    if event_label:
        ev_fs = 0.65
        (ew, _), _ = cv2.getTextSize(event_label, cv2.FONT_HERSHEY_SIMPLEX, ev_fs, 2)
        cv2.putText(
            out,
            event_label,
            (frame.shape[1] - ew - 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            ev_fs,
            settings.color_approach,
            2,
        )

    return out


# ROI selection-------------------------------------------------------
def select_roi_interactive(first_frame: np.ndarray) -> tuple[int, int, int, int]:
    window_name = "Select ROI (Press ENTER/SPACE to confirm, C to cancel)"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, first_frame)
    cv2.waitKey(1)

    logger.info("Выделите зону столика мышью, затем нажмите ENTER или SPACE.")

    r = cv2.selectROI(window_name, first_frame, showCrosshair=True, fromCenter=False)

    cv2.destroyWindow(window_name)

    if r[2] == 0 or r[3] == 0:
        logger.warning("ROI не был выделен корректно.")
        raise ValueError("ROI selection failed")

    x, y, w, h = r
    return x, y, x + w, y + h


def auto_roi_center(frame: np.ndarray) -> tuple[int, int, int, int]:
    """In case we don't have GUI just select center of a frame"""
    hw, ww = frame.shape[:2]
    cx, cy = ww // 2, hw // 2
    w, h = int(ww * 0.40), int(hw * 0.40)
    return cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2


# analytics and reports-------------------------------------------------------
def generate_report(
    events: list[dict],
    video_path: str,
    roi: tuple[int, int, int, int],
    fps: float,
    total_frames: int,
) -> None:
    if not events:
        logger.warning("Событий не зафиксировано. Отчёт не создан.")
        return

    df = pd.DataFrame(events)
    df.to_csv(settings.events_csv, index=False)
    logger.info("Таблица событий сохранена: %s", settings.events_csv)

    delays = df.loc[
        (df["event"] == "approach") & df["delay_after_empty"].notna(),
        "delay_after_empty",
    ]

    def _fmt(val: float) -> str:
        return f"{val:.2f}s" if not np.isnan(val) else "—"

    avg_delay = delays.mean() if len(delays) > 0 else float("nan")
    med_delay = delays.median() if len(delays) > 0 else float("nan")
    min_delay = delays.min() if len(delays) > 0 else float("nan")
    max_delay = delays.max() if len(delays) > 0 else float("nan")

    report = (
        f"Видео: {video_path}\n"
        f"Длительность: {total_frames / max(fps, 1):.1f}s  "
        f"({total_frames} кадров, FPS={fps:.2f})\n"
        f"ROI столика: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}\n"
        "\n"
        "--------------------------------------------------------------\n"
        "СТАТИСТИКА СОБЫТИЙ\n"
        "--------------------------------------------------------------\n"
        f"  Подходы к столу (approach) : {int((df['event'] == 'approach').sum())}\n"
        f"  Стол занят (occupied) : {int((df['event'] == 'occupied').sum())}\n"
        f"  Стол пуст (empty) : {int((df['event'] == 'empty').sum())}\n"
        "\n"
        "--------------------------------------------------------------\n"
        "ВРЕМЯ РЕАКЦИИ (задержка: уход гостя -> следующий подход)\n"
        "--------------------------------------------------------------\n"
        f"  Измерений  : {len(delays)}\n"
        f"  Среднее    : {_fmt(avg_delay)}\n"
        f"  Медиана    : {_fmt(med_delay)}\n"
        f"  Минимум    : {_fmt(min_delay)}\n"
        f"  Максимум   : {_fmt(max_delay)}\n"
        "\n"
        "--------------------------------------------------------------\n"
        "ЛОГ СОБЫТИЙ\n"
        "--------------------------------------------------------------\n"
        f"{df.to_string(index=False)}\n"
        "\n"
        "--------------------------------------------------------------\n"
        f"Выходное видео : {settings.output_video}\n"
        f"CSV событий    : {settings.events_csv}\n"
    )

    logger.info("Отчёт:\n%s", report)
    with open(settings.report_file, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Текстовый отчёт сохранён: %s", settings.report_file)


# main pipeline-------------------------------------------------------
def process_video(video_path: str, force_roi: bool = False) -> None:
    start_wall = time.time()

    # 1. video loading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видео: %s", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "Видео: %s | Разрешение: %dx%d | FPS: %.2f | Кадров: %d | Длительность: %.1fs",
        video_path,
        ww,
        hw,
        fps,
        total_f,
        total_f / fps,
    )

    # 2. ROI selection
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Не удалось прочитать первый кадр.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # to the beginning of the video

    if force_roi:
        roi = select_roi_interactive(first_frame)
    else:
        try:
            roi = select_roi_interactive(first_frame)
        except cv2.error:
            logger.warning("GUI недоступен. Использую автоматический центральный ROI.")
            roi = auto_roi_center(first_frame)

    logger.info("ROI столика: x1=%d, y1=%d, x2=%d, y2=%d", *roi)

    # 3. Model loading
    model = load_yolo()

    # 4. Video writing settings
    writer = cv2.VideoWriter(
        settings.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ww, hw),
    )

    # 5. State vars
    state = "empty"
    empty_streak = settings.empty_frames_needed  # start in the state of "empty"
    occupied_streak = 0

    event_label = ""
    event_label_timer = 0

    events: list[dict] = []
    last_empty_ts: float | None = None

    frame_no = 0
    logger.info("Начинаю обработку видео... (нажмите Q для досрочного выхода)")

    # 6. main loop
    cv2.namedWindow("Table Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Table Monitor", 1280, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        timestamp = frame_no / fps

        # detections and ROI checking
        boxes = detect_persons(model, frame)
        occupied_now = person_in_roi(boxes, roi)

        if occupied_now:
            occupied_streak += 1
            empty_streak = 0
        else:
            empty_streak += 1
            occupied_streak = 0

        # transfer state
        new_state = state
        if state == "empty" and occupied_streak >= settings.occupied_frames_needed:
            new_state = "approach"
        elif state == "approach":
            new_state = "occupied"
        elif state == "occupied" and empty_streak >= settings.empty_frames_needed:
            new_state = "empty"

        # events fixation
        if new_state != state:
            state = new_state

            if state == "approach":
                delay = (
                    round(timestamp - last_empty_ts, 3)
                    if last_empty_ts is not None
                    else None
                )
                events.append(
                    {
                        "event": "approach",
                        "timestamp": round(timestamp, 3),
                        "frame": frame_no,
                        "delay_after_empty": delay,
                    }
                )
                event_label = f">>> APPROACH @ {timestamp:.1f}s"
                event_label_timer = int(fps * 3)
                logger.info(
                    "EVENT APPROACH  | t=%.2fs | frame=%d | delay=%ss",
                    timestamp,
                    frame_no,
                    delay,
                )

            elif state == "occupied":
                events.append(
                    {
                        "event": "occupied",
                        "timestamp": round(timestamp, 3),
                        "frame": frame_no,
                        "delay_after_empty": None,
                    }
                )
                logger.info("EVENT OCCUPIED  | t=%.2fs | frame=%d", timestamp, frame_no)

            elif state == "empty":
                last_empty_ts = timestamp
                events.append(
                    {
                        "event": "empty",
                        "timestamp": round(timestamp, 3),
                        "frame": frame_no,
                        "delay_after_empty": None,
                    }
                )
                logger.info("EVENT EMPTY     | t=%.2fs | frame=%d", timestamp, frame_no)

        # event time
        if event_label_timer > 0:
            event_label_timer -= 1
        else:
            event_label = ""

        vis = frame.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 140, 0), 2)
        vis = draw_overlay(vis, roi, state, event_label, fps, frame_no)

        writer.write(vis)

        if frame_no % 100 == 0:
            pct = frame_no / max(total_f, 1) * 100
            logger.info("PROGRESS %d/%d кадров (%.1f%%)", frame_no, total_f, pct)

        cv2.imshow("Table Monitor", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Досрочный выход по нажатию Q.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    logger.info(
        "Обработка завершена за %.1fs. Обработано кадров: %d",
        time.time() - start_wall,
        frame_no,
    )
    generate_report(events, video_path, roi, fps, frame_no)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Детекция уборки столиков по видео (YOLOv8)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Путь к входному видео (например: video1.mp4)",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Принудительный интерактивный выбор ROI",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Переопределить CONF_THRESHOLD из .env (например: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования (по умолчанию: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    if not Path(args.video).exists():
        logger.error("Файл не найден: %s", args.video)
        sys.exit(1)

    if args.conf is not None:
        settings.conf_threshold = args.conf

    process_video(video_path=args.video, force_roi=args.roi)
