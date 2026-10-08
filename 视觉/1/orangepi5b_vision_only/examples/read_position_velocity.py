from __future__ import annotations

import argparse
from contextlib import ExitStack
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.realtime import BallState  # noqa: E402
from steel_ball.realtime import SteelBallVision  # noqa: E402
from steel_ball.camera_controls import list_v4l2_controls  # noqa: E402
from steel_ball.camera_controls import set_v4l2_controls  # noqa: E402
from steel_ball.camera_controls import V4L2Control  # noqa: E402
from steel_ball.uart import UartPositionVelocitySender  # noqa: E402


class TargetSelectionGui:
    def __init__(
        self,
        vision: SteelBallVision,
        *,
        target_cm: float = 0.0,
        window_name: str = "SteelBall API",
        fullscreen: bool = False,
    ) -> None:
        self.calibration = vision.calibration
        self.center_cm = vision.center_cm
        self.track_length_cm = float(
            vision.config.get("coordinate", {}).get("track_length_cm", 25.0)
        )
        self.target_cm = self._clamp_centered(target_cm)
        self.window_name = window_name
        self.camera_device = self._camera_control_device(vision.config["camera"])
        self.camera_controls: dict[str, V4L2Control] = {}
        self.exposure_control: V4L2Control | None = None
        self.exposure_value: int | None = None
        self.exposure_min = 0
        self.exposure_max = 0
        self.exposure_step = 1
        self.exposure_error: str | None = None
        self.exposure_dragging = False
        self._last_exposure_write = 0.0
        self._fps = 0.0
        self._last_show_time: float | None = None
        self.custom_armed = False
        self.exit_requested = False
        self._button_mid = (12, 12, 136, 58)
        self._button_custom = (152, 12, 316, 58)
        self._button_exit = (520, 12, 628, 58)
        self._exposure_slider = (12, 410, 316, 456)
        self._init_exposure_control(vision.config["camera"])

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

    def show(self, state: BallState) -> bool:
        if state.frame is None:
            return not self.exit_requested
        now = time.monotonic()
        if self._last_show_time is not None:
            interval = now - self._last_show_time
            if interval > 0:
                instant_fps = 1.0 / interval
                self._fps = (
                    instant_fps
                    if self._fps == 0.0
                    else 0.85 * self._fps + 0.15 * instant_fps
                )
        self._last_show_time = now
        canvas = state.frame.copy()
        self._draw_track_axis(canvas)
        self._draw_target(canvas)
        self._draw_ball(canvas, state)
        self._draw_controls(canvas, state)
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        return not self.exit_requested and key not in (ord("q"), 27)

    def relative_position(self, centered_position_cm: float) -> float:
        return centered_position_cm - self.target_cm

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONUP and self.exposure_dragging:
            self.exposure_dragging = False
            self._set_exposure_from_slider(x, force=True)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.exposure_dragging:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._set_exposure_from_slider(x)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._hit(self._button_mid, x, y):
            self.target_cm = 0.0
            self.custom_armed = False
            print("target=+0.000 cm (mid)", flush=True)
            return
        if self._hit(self._button_custom, x, y):
            self.custom_armed = True
            print("custom target: touch a point on the image", flush=True)
            return
        if self._hit(self._button_exit, x, y):
            self.exit_requested = True
            print("exit requested", flush=True)
            return
        if self._hit(self._exposure_slider, x, y):
            self.exposure_dragging = True
            self._set_exposure_from_slider(x, force=True)
            return
        if self.custom_armed:
            self.target_cm = self._centered_coordinate_from_frame_x(float(x))
            self.custom_armed = False
            print(f"target={self.target_cm:+.3f} cm (custom)", flush=True)

    @staticmethod
    def _hit(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def _clamp_centered(self, value: float) -> float:
        return float(
            min(
                max(value, -self.center_cm),
                self.track_length_cm - self.center_cm,
            )
        )

    @staticmethod
    def _camera_control_device(camera_config: dict) -> str:
        source_value = camera_config.get("device", camera_config.get("index", 0))
        source_text = str(source_value)
        if source_text.startswith("/dev/"):
            return source_text
        if source_text.isdigit():
            return f"/dev/video{source_text}"
        return source_text

    def _init_exposure_control(self, camera_config: dict) -> None:
        try:
            self.camera_controls = list_v4l2_controls(self.camera_device)
        except RuntimeError as error:
            self.exposure_error = str(error)
            print(f"exposure control unavailable: {error}", flush=True)
            return

        control = self.camera_controls.get("exposure_absolute")
        if control is None:
            self.exposure_error = "exposure_absolute control unavailable"
            print(self.exposure_error, flush=True)
            return

        self.exposure_control = control
        self.exposure_min = control.minimum
        self.exposure_max = control.maximum
        self.exposure_step = max(1, control.step)
        self.exposure_value = control.value

    def _set_exposure_from_slider(self, x: int, *, force: bool = False) -> None:
        if self.exposure_control is None or self.exposure_value is None:
            if self.exposure_error:
                print(f"exposure control unavailable: {self.exposure_error}", flush=True)
            return

        slider_left, slider_right = self._exposure_track_bounds()
        left, right = slider_left, slider_right
        ratio = (min(right, max(left, x)) - left) / max(1, right - left)
        next_value = self.exposure_control.clamp(
            self.exposure_min + ratio * (self.exposure_max - self.exposure_min)
        )
        self.exposure_value = next_value

        now = time.monotonic()
        if not force and now - self._last_exposure_write < 0.08:
            return
        self._last_exposure_write = now
        requested = {"exposure_absolute": next_value}

        try:
            if "exposure_auto" in self.camera_controls:
                set_v4l2_controls(
                    self.camera_device,
                    {"exposure_auto": 1},
                    self.camera_controls,
                )
            applied = set_v4l2_controls(
                self.camera_device,
                requested,
                self.camera_controls,
            )
            refreshed = list_v4l2_controls(self.camera_device)
        except RuntimeError as error:
            self.exposure_error = str(error)
            print(f"exposure control warning: {error}", flush=True)
            return

        self.exposure_error = None
        self.camera_controls = refreshed
        refreshed_control = refreshed.get("exposure_absolute")
        self.exposure_value = (
            refreshed_control.value
            if refreshed_control is not None
            else applied.get("exposure_absolute", next_value)
        )
        print(f"exposure_absolute={self.exposure_value}", flush=True)

    def _centered_coordinate_from_frame_x(self, x_px: float) -> float:
        points = sorted(self.calibration.points, key=lambda item: item[0])
        xs = np.asarray([item[0] for item in points], dtype=np.float64)
        ys = np.asarray([item[1] for item in points], dtype=np.float64)
        x_px = float(np.clip(x_px, xs[0], xs[-1]))
        y_on_track = float(np.interp(x_px, xs, ys))
        absolute_cm = self.calibration.position_cm(x_px, y_on_track)
        return self._clamp_centered(absolute_cm - self.center_cm)

    def _frame_point_for_centered_cm(self, centered_cm: float) -> tuple[int, int]:
        absolute_cm = float(np.clip(centered_cm + self.center_cm, 0.0, self.track_length_cm))
        points = sorted(self.calibration.points, key=lambda item: item[2])
        positions = np.asarray([item[2] for item in points], dtype=np.float64)
        xs = np.asarray([item[0] for item in points], dtype=np.float64)
        ys = np.asarray([item[1] for item in points], dtype=np.float64)
        return (
            round(float(np.interp(absolute_cm, positions, xs))),
            round(float(np.interp(absolute_cm, positions, ys))),
        )

    def _draw_track_axis(self, canvas: np.ndarray) -> None:
        start = self._frame_point_for_centered_cm(-self.center_cm)
        end = self._frame_point_for_centered_cm(self.track_length_cm - self.center_cm)
        cv2.line(canvas, start, end, (80, 80, 80), 2, cv2.LINE_AA)
        for tick in range(round(self.track_length_cm) + 1):
            centered = tick - self.center_cm
            point = self._frame_point_for_centered_cm(centered)
            cv2.line(
                canvas,
                (point[0], point[1] - 6),
                (point[0], point[1] + 6),
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            if tick % 5 == 0:
                cv2.putText(
                    canvas,
                    f"{centered:+.1f}",
                    (point[0] - 22, point[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def _draw_target(self, canvas: np.ndarray) -> None:
        x, y = self._frame_point_for_centered_cm(self.target_cm)
        cv2.line(canvas, (x, 70), (x, canvas.shape[0] - 12), (0, 180, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (x, y), (0, 180, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(
            canvas,
            f"target {self.target_cm:+.3f} cm",
            (max(8, x - 82), min(canvas.shape[0] - 18, y + 42)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_ball(self, canvas: np.ndarray, state: BallState) -> None:
        if not state.valid or state.centered_position_cm is None:
            return
        x, y = self._frame_point_for_centered_cm(state.centered_position_cm)
        cv2.circle(canvas, (x, y), 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"x {self.relative_position(state.centered_position_cm):+.3f}",
            (max(8, x - 58), max(92, y - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_controls(self, canvas: np.ndarray, state: BallState) -> None:
        self._exposure_slider = (12, canvas.shape[0] - 58, 316, canvas.shape[0] - 12)
        self._draw_button(canvas, self._button_mid, "MID 0", self.target_cm == 0.0)
        self._draw_button(canvas, self._button_custom, "CUSTOM", self.custom_armed)
        self._draw_button(canvas, self._button_exit, "EXIT", False)
        self._draw_exposure_slider(canvas)
        velocity = "--" if state.velocity_cm_s is None else f"{state.velocity_cm_s:+.3f}"
        raw_position = (
            "--"
            if state.centered_position_cm is None
            else f"{state.centered_position_cm:+.3f}"
        )
        text = f"raw {raw_position} cm   out {self.target_cm:+.3f}->0   v {velocity} cm/s"
        cv2.putText(
            canvas,
            text,
            (330, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        fps_text = f"FPS {self._fps:.1f}" if self._last_show_time is not None else "FPS --"
        text_size, _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        fps_x = canvas.shape[1] - text_size[0] - 12
        cv2.rectangle(
            canvas,
            (fps_x - 8, 66),
            (canvas.shape[1] - 6, 94),
            (35, 35, 35),
            -1,
        )
        cv2.putText(
            canvas,
            fps_text,
            (fps_x, 87),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _exposure_track_bounds(self) -> tuple[int, int]:
        left, _top, right, _bottom = self._exposure_slider
        return left + 112, right - 16

    def _draw_exposure_slider(self, canvas: np.ndarray) -> None:
        left, top, right, bottom = self._exposure_slider
        disabled = self.exposure_control is None or self.exposure_value is None
        background = (35, 35, 35) if disabled else (55, 55, 55)
        cv2.rectangle(canvas, (left, top), (right, bottom), background, -1)
        cv2.rectangle(canvas, (left, top), (right, bottom), (230, 230, 230), 1)

        label = "EXP --" if self.exposure_value is None else f"EXP {self.exposure_value}"
        cv2.putText(
            canvas,
            label,
            (left + 12, top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (150, 150, 150) if disabled else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        track_left, track_right = self._exposure_track_bounds()
        track_y = (top + bottom) // 2
        cv2.line(
            canvas,
            (track_left, track_y),
            (track_right, track_y),
            (125, 125, 125),
            4,
            cv2.LINE_AA,
        )
        if disabled:
            return
        value_range = max(1, self.exposure_max - self.exposure_min)
        ratio = (self.exposure_value - self.exposure_min) / value_range
        handle_x = round(track_left + ratio * (track_right - track_left))
        cv2.line(
            canvas,
            (track_left, track_y),
            (handle_x, track_y),
            (0, 180, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (handle_x, track_y), 9, (0, 180, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_button(
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        label: str,
        active: bool,
        disabled: bool = False,
    ) -> None:
        left, top, right, bottom = rect
        color = (35, 35, 35) if disabled else (0, 150, 80) if active else (55, 55, 55)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
        cv2.rectangle(canvas, (left, top), (right, bottom), (230, 230, 230), 1)
        cv2.putText(
            canvas,
            label,
            (left + 14, top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read steel-ball position and velocity.")
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "vision.yaml"
    )
    parser.add_argument(
        "--uart-device",
        help="UART device for downstream controller output, for example /dev/ttyS1.",
    )
    parser.add_argument("--uart-baud", type=int, default=115200)
    parser.add_argument("--gui", action="store_true", help="Show target selection GUI.")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--target-cm", type=float, default=0.0)
    args = parser.parse_args()

    last_frame_id = -1
    try:
        with ExitStack() as stack:
            vision = stack.enter_context(
                SteelBallVision(
                    config_path=args.config,
                    camera_device=args.camera_device,
                    publish_frame=args.gui,
                )
            )
            gui = None
            if args.gui:
                gui = TargetSelectionGui(
                    vision,
                    target_cm=args.target_cm,
                    fullscreen=args.fullscreen,
                )
                stack.callback(gui.close)

            uart = None
            if args.uart_device:
                uart = stack.enter_context(
                    UartPositionVelocitySender(
                        device=args.uart_device,
                        baud_rate=args.uart_baud,
                    )
                )

            while True:
                state = vision.wait_for_update(last_frame_id, timeout=1.0)
                if state is None:
                    continue
                last_frame_id = state.frame_id

                if gui is not None and not gui.show(state):
                    break

                # Downstream position is relative to the selected target.
                raw_position_cm = state.centered_position_cm
                ball_velocity_cm_s = state.velocity_cm_s

                if (
                    state.valid
                    and raw_position_cm is not None
                    and ball_velocity_cm_s is not None
                ):
                    ball_position_cm = (
                        gui.relative_position(raw_position_cm)
                        if gui is not None
                        else raw_position_cm
                    )
                    if uart is not None:
                        uart.send(ball_position_cm, ball_velocity_cm_s)
                    source = "predicted" if state.predicted else "detected"
                    print(
                        f"position={ball_position_cm:+.3f} cm, "
                        f"velocity={ball_velocity_cm_s:+.3f} cm/s, "
                        f"target={0.0 if gui is None else gui.target_cm:+.3f} cm, "
                        f"source={source}",
                        flush=True,
                    )
                else:
                    print("ball state unavailable", flush=True)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
