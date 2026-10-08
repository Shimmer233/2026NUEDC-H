from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import label_for_image  # noqa: E402


WINDOW_NAME = "Steel ball label review"
REPORT_NAME = "completion_report.json"
REVIEW_ACTIONS = {"accepted_auto", "redrawn", "empty"}
INFO_HEIGHT = 88


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and correct every generated ball box.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v4_incremental"),
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Resume at the first pending image (this is also the default start position).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Integer display scale; scale 2 keeps the complete track visible on common screens.",
    )
    return parser.parse_args()


def read_status(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ("object_expected", "review_action", "reviewed_at"):
        if field not in fields:
            fields.append(field)
    for row in rows:
        row.setdefault("object_expected", "1")
        row.setdefault("review_action", "")
        row.setdefault("reviewed_at", "")
    return rows, fields


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_text(path: Path, text: str, encoding: str) -> None:
    payload = text.encode(encoding)
    _atomic_replace(path, lambda stream: stream.write(payload))


def save_status(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), encoding="utf-8")


def read_box(label_path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    lines = [line.split() for line in label_path.read_text(encoding="ascii").splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) != 1 or len(lines[0]) != 5:
        raise ValueError(f"label must contain exactly one YOLO box: {label_path}")
    fields = lines[0]
    if fields[0] != "0":
        raise ValueError(f"label class must be 0: {label_path}")
    try:
        center_x, center_y, box_width, box_height = (float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"label contains a non-numeric box: {label_path}") from error
    x0 = round((center_x - box_width / 2) * width)
    y0 = round((center_y - box_height / 2) * height)
    x1 = round((center_x + box_width / 2) * width)
    y1 = round((center_y + box_height / 2) * height)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError(f"label box is outside the image or smaller than 4x4: {label_path}")
    return x0, y0, x1, y1


def box_text(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x0, y0, x1, y1 = box
    x0 = min(width, max(0, x0))
    x1 = min(width, max(0, x1))
    y0 = min(height, max(0, y0))
    y1 = min(height, max(0, y1))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("box must be at least 4x4 pixels and inside the image")
    center_x = (x0 + x1) / 2 / width
    center_y = (y0 + y1) / 2 / height
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def write_box(label_path: Path, box: tuple[int, int, int, int], width: int, height: int) -> None:
    atomic_write_text(label_path, box_text(box, width, height), encoding="ascii")


def review_labels_digest(dataset: Path, rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_image"]):
        relative = Path(row["relative_image"])
        label_path = label_for_image(dataset, dataset / relative)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(label_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _reported_action(dataset: Path, row: dict[str, str]) -> str:
    action = row.get("review_action", "")
    if action in REVIEW_ACTIONS:
        return action
    if row.get("object_expected", "1") != "1":
        return "empty"
    image_path = dataset / row["relative_image"]
    label_path = label_for_image(dataset, image_path)
    auto_path = dataset / "labels_auto" / row["split"] / f"{image_path.stem}.txt"
    if auto_path.is_file() and label_path.read_bytes() != auto_path.read_bytes():
        return "redrawn"
    return "accepted_auto"


def write_completion_report(dataset: Path, rows: list[dict[str, str]]) -> Path | None:
    if not rows or any(row.get("status") != "accepted" for row in rows):
        return None
    actions = [_reported_action(dataset, row) for row in rows]
    ball_count = sum(row.get("object_expected", "1") == "1" for row in rows)
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "ball": ball_count,
        "empty": len(rows) - ball_count,
        "redrawn": actions.count("redrawn"),
        "accepted_auto": actions.count("accepted_auto"),
        "labels_sha256": review_labels_digest(dataset, rows),
    }
    report_path = dataset / "review" / REPORT_NAME
    atomic_write_text(report_path, json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def first_pending_index(rows: list[dict[str, str]]) -> int | None:
    return next((index for index, row in enumerate(rows) if row.get("status") != "accepted"), None)


class Reviewer:
    def __init__(
        self,
        dataset: Path,
        rows: list[dict[str, str]],
        fields: list[str],
        status_path: Path | None = None,
        start_index: int | None = None,
        scale: int = 2,
    ) -> None:
        self.dataset = dataset
        self.rows = rows
        self.fields = fields
        self.status_path = status_path or dataset / "review" / "status.csv"
        pending_index = first_pending_index(rows)
        self.index = start_index if start_index is not None else (pending_index or 0)
        if scale not in (1, 2, 3):
            raise ValueError("scale must be 1, 2, or 3")
        self.scale = scale
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.current_box: tuple[int, int, int, int] | None = None
        self.auto_box: tuple[int, int, int, int] | None = None
        self.was_redrawn = False
        self.current_image = None
        self.message = ""

    def _paths(self) -> tuple[Path, Path, Path]:
        row = self.rows[self.index]
        image_path = self.dataset / row["relative_image"]
        label_path = label_for_image(self.dataset, image_path)
        auto_path = self.dataset / "labels_auto" / row["split"] / f"{image_path.stem}.txt"
        return image_path, label_path, auto_path

    def load(self) -> None:
        image_path, label_path, auto_path = self._paths()
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            raise RuntimeError(f"unable to read {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if not auto_path.is_file():
            raise FileNotFoundError(auto_path)
        width, height = self.current_image.shape[1], self.current_image.shape[0]
        self.current_box = read_box(label_path, width, height)
        self.auto_box = read_box(auto_path, width, height)
        self.was_redrawn = label_path.read_bytes() != auto_path.read_bytes()
        self.drag_start = None
        self.drag_end = None
        self.message = ""

    def _image_point(self, x: int, y: int) -> tuple[int, int]:
        height, width = self.current_image.shape[:2]
        return (
            min(width, max(0, x // self.scale)),
            min(height, max(0, y // self.scale)),
        )

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        point = self._image_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            self.drag_end = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = point
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = point
            x0, x1 = sorted((self.drag_start[0], self.drag_end[0]))
            y0, y1 = sorted((self.drag_start[1], self.drag_end[1]))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                self.current_box = (x0, y0, x1, y1)
                self.was_redrawn = True
                self.message = "New BALL box ready; press Enter/Space to accept"
            else:
                self.message = "Box ignored: draw at least 4x4 pixels"
            self.drag_start = None
            self.drag_end = None

    def render(self):
        canvas = self.current_image.copy()
        row = self.rows[self.index]
        color = (0, 220, 0) if row["status"] == "accepted" else (0, 180, 255)
        if self.current_box is not None:
            x0, y0, x1, y1 = self.current_box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        if self.drag_start is not None and self.drag_end is not None:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (255, 120, 0), 1)
        scaled = cv2.resize(
            canvas,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = cv2.copyMakeBorder(
            scaled,
            0,
            INFO_HEIGHT,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(28, 28, 28),
        )
        proposal = "REDRAWN" if self.was_redrawn else "AUTO"
        line_one = (
            f"{self.index + 1}/{len(self.rows)}  {row['status']}  "
            f"{'BALL' if self.current_box is not None else 'EMPTY'}  {proposal}  "
            f"remaining={sum(item['status'] != 'accepted' for item in self.rows)}  "
            f"uncertain={row.get('uncertain', '')}"
        )
        line_two = Path(row["relative_image"]).name
        text_y = scaled.shape[0] + 23
        for text, y in ((line_one, text_y), (line_two, text_y + 25)):
            cv2.putText(canvas, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
            cv2.putText(canvas, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
        if self.message:
            y = text_y + 50
            cv2.putText(canvas, self.message, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
            cv2.putText(canvas, self.message, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 230, 255), 1)
        return canvas

    def _commit(self, label_text: str, expected: str, action: str) -> Path | None:
        if action not in REVIEW_ACTIONS:
            raise ValueError(f"unsupported review action: {action}")
        _, label_path, _ = self._paths()
        atomic_write_text(label_path, label_text, encoding="ascii")

        row = self.rows[self.index]
        previous = {
            key: row.get(key, "")
            for key in ("status", "object_expected", "review_action", "reviewed_at")
        }
        row.update(
            {
                "status": "accepted",
                "object_expected": expected,
                "review_action": action,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            save_status(self.status_path, self.rows, self.fields)
        except Exception:
            row.update(previous)
            raise
        return write_completion_report(self.dataset, self.rows)

    def accept_current_box(self) -> Path | None:
        if self.current_box is None:
            raise ValueError("no ball box; press N to explicitly mark this image EMPTY")
        height, width = self.current_image.shape[:2]
        text = box_text(self.current_box, width, height)
        action = "redrawn" if self.was_redrawn else "accepted_auto"
        report = self._commit(text, expected="1", action=action)
        self.message = "Accepted BALL box"
        return report

    def mark_current_empty(self) -> Path | None:
        self.current_box = None
        self.was_redrawn = False
        report = self._commit("", expected="0", action="empty")
        self.message = "Accepted EMPTY image"
        return report

    def reset_current(self) -> None:
        self.current_box = self.auto_box
        self.was_redrawn = False
        self.drag_start = None
        self.drag_end = None
        self.message = "Automatic proposal restored; confirm it explicitly"

    def move(self, offset: int) -> None:
        self.index = min(len(self.rows) - 1, max(0, self.index + offset))
        self.load()

    def move_to_next_pending(self) -> bool:
        pending = [
            index
            for index, row in enumerate(self.rows)
            if row.get("status") != "accepted" and index != self.index
        ]
        if not pending:
            return False
        following = [index for index in pending if index > self.index]
        self.index = following[0] if following else pending[0]
        self.load()
        return True


def main() -> int:
    args = parse_args()
    status_path = args.dataset / "review" / "status.csv"
    rows, fields = read_status(status_path)
    if not rows:
        print("No labels require review.")
        return 0
    pending_index = first_pending_index(rows)
    if pending_index is None:
        report_path = write_completion_report(args.dataset, rows)
        print(f"No labels require review. Completion report: {report_path}")
        return 0

    reviewer = Reviewer(
        args.dataset,
        rows,
        fields,
        status_path=status_path,
        start_index=pending_index,
        scale=args.scale,
    )
    reviewer.load()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, reviewer.mouse)
    report_path: Path | None = None
    while True:
        cv2.imshow(WINDOW_NAME, reviewer.render())
        key = cv2.waitKeyEx(20)
        if key in (ord("q"), ord("Q"), 27):
            break
        try:
            if key in (13, 32):
                report_path = reviewer.accept_current_box()
                if report_path is not None or not reviewer.move_to_next_pending():
                    break
            elif key in (ord("d"), ord("D"), 2555904):
                reviewer.move(1)
            elif key in (ord("a"), ord("A"), 2424832):
                reviewer.move(-1)
            elif key in (ord("r"), ord("R")):
                reviewer.reset_current()
            elif key in (ord("n"), ord("N")):
                report_path = reviewer.mark_current_empty()
                if report_path is not None or not reviewer.move_to_next_pending():
                    break
        except ValueError as error:
            reviewer.message = str(error)

    cv2.destroyAllWindows()
    accepted = sum(row["status"] == "accepted" for row in reviewer.rows)
    print(f"Accepted {accepted}/{len(reviewer.rows)} labels in this review set.")
    if report_path is not None:
        print(f"Completion report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
