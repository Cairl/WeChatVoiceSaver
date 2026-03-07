import sys
import time
import os
import ctypes
import threading
import re
import subprocess
import unicodedata
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from ctypes import wintypes
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
from process_audio_capture import ProcessAudioCapture
# pip install process-audio-capture 

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


@dataclass(frozen=True)
class RecorderConfig:
    """录音与检测参数（集中管理，默认值保持现状）。"""

    trigger_db: float = -45.0
    stable_stop_db: float = -45.0
    stable_delta_db: float = 0.001
    stable_duration_s: float = 0.3
    silence_db: float = -50.0
    silence_timeout_s: float = 1.5

    min_duration_s: int = 1
    max_duration_s: int = 60

    pipe_connect_timeout_s: float = 5.0
    tick_sleep_s: float = 0.05
    no_audio_sleep_s: float = 0.2
    after_session_sleep_s: float = 0.5


@dataclass
class RecordingSession:
    """一次录音会话的可变状态。"""

    start_time: float
    temp_path: Path
    locked_friend: str
    log_row: int

    silence_start: Optional[float] = None
    last_level: float = -999.0
    stable_start: Optional[float] = None

    last_elapsed: int = 0
    ellipsis_index: int = 0
    ellipsis_count: int = 0


class ProcessUtils:
    """进程工具类"""
    
    @staticmethod
    def find_process_by_name(target_name: str) -> List[int]:
        result = []
        try:
            output = subprocess.check_output(
                ['tasklist', '/FI', f'IMAGENAME eq {target_name}', '/FO', 'CSV', '/NH'],
                text=True, encoding='gbk', errors='ignore'
            )
            for line in output.strip().split('\n'):
                if target_name.lower() in line.lower():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pid_str = parts[1].strip().strip('"')
                        try:
                            pid = int(pid_str)
                            result.append(pid)
                        except ValueError:
                            pass
        except Exception:
            pass
        return result
    
    @staticmethod
    def get_audio_process_pid(name_pattern: str) -> Optional[int]:
        try:
            processes = ProcessAudioCapture.enumerate_audio_processes()
            for p in processes:
                if name_pattern.lower() in p.name.lower():
                    return p.pid
        except Exception:
            pass
        return None


class WeChatWindowInfo:
    """微信窗口信息获取器"""
    
    WECHAT_CLASS_NAMES = [
        "WeChatMainWndForPC",
        "Chrome_WidgetWin_0",
        "WeChatLoginWndForPC",
        "Qt51514QWindowIcon",
    ]
    
    @staticmethod
    def get_window_text(hwnd: int) -> str:
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value
    
    @staticmethod
    def get_window_pid(hwnd: int) -> int:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value
    
    @classmethod
    def find_wechat_windows(cls) -> list:
        windows = []
        
        def enum_callback(hwnd, _):
            for class_name in cls.WECHAT_CLASS_NAMES:
                buffer = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, buffer, 256)
                if buffer.value == class_name:
                    title = cls.get_window_text(hwnd)
                    pid = cls.get_window_pid(hwnd)
                    if title:
                        windows.append({
                            'hwnd': hwnd,
                            'title': title,
                            'pid': pid,
                            'class': buffer.value
                        })
                    break
            return True
        
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
        return windows
    
    @classmethod
    def get_friend_name(cls, hwnd: int = None) -> Optional[str]:
        if hwnd:
            title = cls.get_window_text(hwnd)
        else:
            windows = cls.find_wechat_windows()
            if not windows:
                return None
            title = windows[0]['title']
        
        if not title or title in ["微信", "WeChat", "Weixin", ""]:
            return None
        
        if title == "文件传输":
            return "文件传输助手"
        
        name = title.strip()
        
        if name.endswith(" - 微信") or name.endswith(" - WeChat"):
            name = name.rsplit(" - ", 1)[0]
        
        patterns = [
            r'^\[(.*?)\]$',
            r'^(.+?)\s*\(\d+人\)$',
            r'^(.+?)\s*\(\d+\)$',
            r'^(.+?)\s*\[\d+\]$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                name = match.group(1).strip()
                break
        
        return name if name else None


class PipeAudioSink:
    """命名管道音频接收器 - 支持缓冲和分段录制"""
    
    def __init__(self):
        self._pipe_handle = None
        self._thread = None
        self._stop_event = threading.Event()
        self._pipe_name = fr'\\.\pipe\wechat_rec_{int(time.time())}_{id(self)}'
        
        # 录制状态
        self._recording = False
        self._file_handle = None
        self._output_path = None
        self._total_written = 0
        self._data_size_offset = None
        
        # 缓冲相关
        self._buffer_lock = threading.Lock()
        self._chunks_buffer = []
        self._buffer_size = 0
        self._max_buffer_size = 192000 * 2  # 约2秒 (48k * 2ch * 2byte)
        
        self._connected = threading.Event()
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    @property
    def pipe_name(self) -> str:
        return self._pipe_name
    
    @property
    def bytes_written(self) -> int:
        return self._total_written
    
    @property
    def is_recording(self) -> bool:
        return self._recording
        
    def start(self):
        PIPE_ACCESS_INBOUND = 0x00000001
        PIPE_TYPE_BYTE = 0x00000000
        PIPE_WAIT = 0x00000000
        INVALID_HANDLE_VALUE = -1
        
        self._pipe_handle = kernel32.CreateNamedPipeW(
            self._pipe_name,
            PIPE_ACCESS_INBOUND,
            PIPE_TYPE_BYTE | PIPE_WAIT,
            1, 65536, 65536, 0, None
        )
        
        if self._pipe_handle == INVALID_HANDLE_VALUE:
            raise OSError(f"Failed to create named pipe: {ctypes.get_last_error()}")
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def stop(self):
        self.stop_recording()
        self._stop_event.set()
        if self._pipe_handle:
            kernel32.DisconnectNamedPipe(self._pipe_handle)
            kernel32.CloseHandle(self._pipe_handle)
            self._pipe_handle = None
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def _generate_default_wav_header(self) -> bytes:
        import struct
        
        sample_rate = 48000
        num_channels = 2
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        
        header = bytearray(44)
        struct.pack_into('<4sI', header, 0, b'RIFF', 0)
        struct.pack_into('<4s', header, 8, b'WAVE')
        struct.pack_into('<4sIHHIIHH', header, 12, b'fmt ', 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample)
        struct.pack_into('<4sI', header, 36, b'data', 0)
        
        return bytes(header)
    
    def start_recording(self, output_path: Path):
        with self._buffer_lock:
            if self._recording:
                return
            
            try:
                self._output_path = output_path
                self._file_handle = open(output_path, 'wb')
                self._recording = True
                self._total_written = 0
                self._data_size_offset = 40
                
                default_header = self._generate_default_wav_header()
                self._file_handle.write(default_header)
                self._total_written += len(default_header)
                
                for chunk in self._chunks_buffer:
                    self._write_chunk_to_file(chunk)
                
                self._chunks_buffer = []
                self._buffer_size = 0
                
            except Exception as e:
                sys.stderr.write(f"\nError starting recording: {e}\n")
                self._recording = False
                if self._file_handle:
                    self._file_handle.close()
                    self._file_handle = None

    def stop_recording(self):
        with self._buffer_lock:
            if not self._recording:
                return
            
            try:
                if self._file_handle:
                    self._file_handle.flush()
                    self._update_wav_header(self._file_handle, self._total_written)
                    self._file_handle.close()
            except Exception as e:
                sys.stderr.write(f"\nError stopping recording: {e}\n")
            finally:
                self._file_handle = None
                self._recording = False
                self._total_written = 0

    def _write_chunk_to_file(self, data):
        if not self._file_handle:
            return
            
        # 尝试查找 data chunk offset
        if self._data_size_offset is None:
            self._data_size_offset = self._find_data_chunk_offset(data, self._total_written)
            
        self._file_handle.write(data)
        self._total_written += len(data)

    def _run(self):
        kernel32.ConnectNamedPipe(self._pipe_handle, None)
        self._connected.set()
        
        buffer = ctypes.create_string_buffer(65536)
        bytes_read = wintypes.DWORD()
        last_flush_time = time.time()
        
        try:
            while not self._stop_event.is_set():
                success = kernel32.ReadFile(
                    self._pipe_handle,
                    buffer,
                    len(buffer),
                    ctypes.byref(bytes_read),
                    None
                )
                
                if not success or bytes_read.value == 0:
                    break
                    
                data = buffer.raw[:bytes_read.value]
                
                with self._buffer_lock:
                    if self._recording:
                        self._write_chunk_to_file(data)
                        
                        now = time.time()
                        if now - last_flush_time > 1.0:
                            if self._file_handle:
                                self._file_handle.flush()
                                self._update_wav_header(self._file_handle, self._total_written)
                            last_flush_time = now
                    else:
                        # 缓冲模式
                        self._chunks_buffer.append(data)
                        self._buffer_size += len(data)
                        
                        # 维持缓冲区大小
                        while self._buffer_size > self._max_buffer_size:
                            if self._chunks_buffer:
                                removed = self._chunks_buffer.pop(0)
                                self._buffer_size -= len(removed)
                            else:
                                break
            
            # 停止时确保文件关闭
            self.stop_recording()
            
        except Exception as e:
            sys.stderr.write(f"\n[PipeSink] Error: {e}\n")

    def _find_data_chunk_offset(self, data: bytes, current_offset: int) -> Optional[int]:
        if current_offset > 200: 
            return None
        try:
            data_idx = data.find(b'data')
            if data_idx != -1:
                return current_offset + data_idx + 4
        except Exception:
            pass
        return None

    def _update_wav_header(self, f, file_size: int):
        if file_size < 44 or self._data_size_offset is None:
            return
        try:
            current_pos = f.tell()
            f.seek(4)
            f.write((file_size - 8).to_bytes(4, 'little'))
            data_size = file_size - (self._data_size_offset + 4)
            if data_size > 0:
                f.seek(self._data_size_offset)
                f.write(data_size.to_bytes(4, 'little'))
            f.seek(current_pos)
        except Exception:
            pass


class ConsoleUI:
    """控制台界面管理器"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_status_line = ""
        self._log_history = []
        self._max_history = 10
        # 定义内容区域的固定显示宽度
        self._content_width = 80
        # 状态栏固定占用 4 行（上框/内容/下框/空行）
        self._status_rows = 4
        # 记录已使用到的最后一条日志行号（从状态栏下面开始）
        self._current_log_row = self._status_rows

        # 终端若不是 UTF-8（Windows 常见 GBK），框线/音量条字符可能乱码或触发编码异常。
        # 这里自动降级为 ASCII 字符集，UTF-8 终端保持原样式。
        enc = (getattr(sys.stdout, "encoding", None) or "").lower().replace("_", "-")
        unicode_ok = "utf-8" in enc
        if unicode_ok:
            self._frame = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"}
            self._bar = {"full": "█", "empty": "░"}
        else:
            self._frame = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"}
            self._bar = {"full": "#", "empty": "."}
        
    def clear_screen(self):
        # \033[?25l: 隐藏光标
        # \033[2J\033[H: 清屏并移动到左上角
        with self._lock:
            sys.stdout.write("\033[?25l\033[2J\033[H")
            sys.stdout.flush()
            # 清屏后重置日志行号
            self._current_log_row = self._status_rows

    def reserve_log_line(self, *, blank_line: bool = False) -> int:
        """预留一条新的日志行并返回其行号。"""
        with self._lock:
            self._current_log_row += 1
            row = self._current_log_row
            if blank_line:
                sys.stdout.write(f"\033[{row};1H\033[K\n")
                sys.stdout.flush()
            return row

    def write_recording_line(self, row: int, text: str, *, newline: bool = False):
        """在指定行刷新录制进度（不会影响状态栏区域）。"""
        with self._lock:
            sys.stdout.write(f"\033[{row};1H\033[K{text}")
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()
        
    def update_status(self, level: float, is_recording: bool, duration: int = 0, friend_name: str = ""):
        """更新顶部状态栏"""
        
        def get_display_width(text: str) -> int:
            text_without_ansi = re.sub(r'\x1b\[[0-9;]*m', '', text)
            return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text_without_ansi)

        def truncate_by_width(text: str, max_width: int) -> str:
            if get_display_width(text) <= max_width:
                return text
            # 预留3个点 (...) 的宽度
            safe_width = max_width - 3
            current_w = 0
            res = ""
            for c in text:
                w = 2 if unicodedata.east_asian_width(c) in 'WF' else 1
                if current_w + w > safe_width:
                    return res + "..."
                res += c
                current_w += w
            return res + "..."

        # 1. 构造内容
        if is_recording:
            status_part = "\033[31m●\033[0m 录制"
            time_str = f"{duration // 60:02d}:{duration % 60:02d}"
        else:
            status_part = "\033[32m●\033[0m 待机"
            time_str = "--:--"
            
        level_normalized = max(0, min(1, (level + 60) / 60))
        bars = int(level_normalized * 20)
        level_bar = self._bar["full"] * bars + self._bar["empty"] * (20 - bars)
        
        friendly_name = friend_name if friend_name else "等待微信..."
        # 截断函数已确保是 3 个点
        fixed_friend_name = truncate_by_width(friendly_name, 16)
        
        content = f"{status_part} | {time_str} | 音量: {level_bar} {level:+5.1f}dB | 当前好友: {fixed_friend_name}"
        final_content = f" {content} "
        border_line_len = get_display_width(final_content)
        
        # 2. 渲染 (关键改动)
        with self._lock:
            # \033[H: 回到左上角
            sys.stdout.write("\033[H")
            
            # 第一行：顶边框
            sys.stdout.write(f"\033[K{self._frame['tl']}{self._frame['h'] * border_line_len}{self._frame['tr']}\n")
            # 第二行：内容区域
            sys.stdout.write(f"\033[K{self._frame['v']}{final_content}{self._frame['v']}\n")
            # 第三行：底边框
            sys.stdout.write(f"\033[K{self._frame['bl']}{self._frame['h'] * border_line_len}{self._frame['br']}\n")
            
            # 第四行：隔离空行，并清除残留
            sys.stdout.write("\033[K\n")
            sys.stdout.flush()

    def log(self, message: str):
        """添加日志，确保不会跳回顶部"""
        with self._lock:
            # 始终在当前光标位置打印，并清除该行末尾
            sys.stdout.write(f"\033[K{message}\n")
            sys.stdout.flush()

    # 兼容更直观的命名
    def print_log(self, message: str):
        self.log(message)


class WeChatVoiceRecorder:
    """微信语音录制器"""
    
    def __init__(self, output_dir: str = "."):
        self.config = RecorderConfig()
        self.output_dir = Path(output_dir)
        self._wechat_pid: Optional[int] = None
        self._wechat_hwnd: Optional[int] = None
        self._level_db: float = -60.0
        self._level_lock = threading.Lock()
        self.ui = ConsoleUI()
        self._session: Optional[RecordingSession] = None
        self._previous_friend = None
    
    def find_wechat_process(self) -> Optional[Tuple[int, int]]:
        windows = WeChatWindowInfo.find_wechat_windows()
        
        for name in ["Weixin.exe", "WeChat.exe", "wechat.exe"]:
            pids = ProcessUtils.find_process_by_name(name)
            if pids:
                for win in windows:
                    if win['pid'] in pids:
                        return (win['pid'], win['hwnd'])
                return (pids[0], None)
        
        if windows:
            win = windows[0]
            return (win['pid'], win['hwnd'])
            
        return None
    
    def _generate_filename(self, friend_name: str, duration_seconds: int) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        bad_chars = '<>:"/\\|?*'
        folder_name = "特殊名称" if any(c in bad_chars for c in friend_name) else (friend_name or "未知好友")
        
        friend_folder = self.output_dir / folder_name
        friend_folder.mkdir(exist_ok=True)
        
        return friend_folder / f"{timestamp} {duration_seconds:02d}s.wav"
    
    def _get_current_friend_name(self) -> str:
        windows = WeChatWindowInfo.find_wechat_windows()
        if windows:
            self._wechat_hwnd = windows[0]['hwnd']
            name = WeChatWindowInfo.get_friend_name(self._wechat_hwnd)
            if name:
                return name
        return "未知好友"
    
    def _level_callback(self, level_db: float):
        with self._level_lock:
            self._level_db = level_db
    
    def _get_level(self) -> float:
        with self._level_lock:
            return self._level_db

    def _tick_idle(self, sink: "PipeAudioSink", level: float):
        if level <= self.config.trigger_db:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = self.output_dir / f"_temp_{timestamp}.wav"
        sink.start_recording(temp_path)

        locked_friend = self._get_current_friend_name()

        # 检查好友是否变更，如果变更则空出一行
        if self._previous_friend and self._previous_friend != locked_friend:
            self.ui.reserve_log_line(blank_line=True)

        log_row = self.ui.reserve_log_line()
        self._session = RecordingSession(
            start_time=time.time(),
            temp_path=temp_path,
            locked_friend=locked_friend,
            log_row=log_row,
            silence_start=None,
            last_level=level,
            stable_start=None,
            last_elapsed=0,
            ellipsis_index=0,
            ellipsis_count=0,
        )

        self.ui.write_recording_line(log_row, f"[{locked_friend}] 00s.", newline=True)
        self._previous_friend = locked_friend

    def _tick_recording(self, sink: "PipeAudioSink", level: float, friend_name_for_status: str):
        session = self._session
        if session is None:
            # 理论上不应发生：正在录制但会话为空，做一次自愈，避免 UI/停止条件异常
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = self.output_dir / f"_temp_{timestamp}.wav"
            session = RecordingSession(
                start_time=time.time(),
                temp_path=temp_path,
                locked_friend=self._get_current_friend_name(),
                log_row=self.ui.reserve_log_line(),
            )
            self._session = session

        elapsed = int(time.time() - session.start_time)
        self.ui.update_status(level, True, elapsed, friend_name_for_status)

        # 每秒更新一次秒数
        if elapsed != session.last_elapsed:
            session.last_elapsed = elapsed
            session.ellipsis_index = 0
            session.ellipsis_count = 0
            ellipsis = "."
        else:
            # 减慢省略号滚动速度（每3次循环更新一次）
            session.ellipsis_count += 1
            if session.ellipsis_count % 3 == 0:
                session.ellipsis_index = (session.ellipsis_index + 1) % 3
            ellipsis = "." * (session.ellipsis_index + 1)

        self.ui.write_recording_line(session.log_row, f"[{session.locked_friend}] {elapsed:02d}s{ellipsis}")

        should_stop = False

        # --- 结束条件 1: 分贝值稳定不变（快速结束） ---
        if (
            abs(level - session.last_level) < self.config.stable_delta_db
            and level < self.config.stable_stop_db
        ):
            if session.stable_start is None:
                session.stable_start = time.time()
            elif time.time() - session.stable_start > self.config.stable_duration_s:
                should_stop = True
        else:
            session.stable_start = None

        session.last_level = level

        # --- 结束条件 2: 传统静音超时（兜底） ---
        if not should_stop:
            if level > self.config.silence_db:
                session.silence_start = None
            else:
                if session.silence_start is None:
                    session.silence_start = time.time()
                elif time.time() - session.silence_start > self.config.silence_timeout_s:
                    should_stop = True

        # --- 结束条件 3: 最大时长 ---
        if elapsed >= self.config.max_duration_s:
            should_stop = True

        if should_stop:
            self._finalize_recording(sink, session)

    def _finalize_recording(self, sink: "PipeAudioSink", session: RecordingSession):
        sink.stop_recording()
        duration = int(time.time() - session.start_time)

        if duration >= self.config.min_duration_s and session.temp_path.exists():
            friend_name = self._get_current_friend_name()
            final_path = self._generate_filename(friend_name, duration)
            try:
                # 避免文件名冲突
                if final_path.exists():
                    base_name = final_path.stem
                    ext = final_path.suffix
                    final_path = final_path.parent / f"{base_name}_{int(time.time())}{ext}"

                # Windows 上偶尔会出现 WinError 32（文件正被占用），这里做有限次重试
                for i in range(5):
                    try:
                        session.temp_path.rename(final_path)
                        break
                    except PermissionError as e:
                        if getattr(e, "winerror", None) == 32 and i < 4:
                            # 稍等一会儿再试，通常是杀毒/索引程序短暂占用
                            time.sleep(0.1)
                            continue
                        raise

                self.ui.write_recording_line(
                    session.log_row,
                    f"[{session.locked_friend}] {duration:02d}s",
                    newline=True,
                )
            except Exception as e:
                # 如果文件已经不存在（WinError 2），直接视为该段录音被系统清理，避免刷出长错误行
                winerr = getattr(e, "winerror", None)
                if isinstance(e, FileNotFoundError) or winerr == 2:
                    # 清掉这行进度即可，不再提示
                    self.ui.write_recording_line(session.log_row, "")
                else:
                    self.ui.print_log(f"保存失败（已丢弃临时文件）: {e}")
                    if session.temp_path.exists():
                        try:
                            session.temp_path.unlink()
                        except Exception:
                            pass
        else:
            if session.temp_path.exists():
                try:
                    session.temp_path.unlink()
                except Exception:
                    pass
            self.ui.write_recording_line(session.log_row, "")

        self._session = None

    @contextmanager
    def _open_audio_pipeline(self, pid: int):
        """统一管理 PipeAudioSink + ProcessAudioCapture 的生命周期。"""
        sink: Optional[PipeAudioSink] = None
        capture = None
        try:
            sink = PipeAudioSink()
            sink.start()

            capture = ProcessAudioCapture(
                pid=pid,
                output_path=sink.pipe_name,
                level_callback=self._level_callback,
            )
            capture.__enter__()
            capture.start()

            sink._connected.wait(timeout=self.config.pipe_connect_timeout_s)
            yield sink
        finally:
            self._session = None
            if capture:
                try:
                    capture.__exit__(None, None, None)
                except Exception:
                    pass
            if sink:
                try:
                    sink.stop()
                except Exception:
                    pass
    
    def run(self):
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        except Exception:
            pass

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(errors="replace")
        except Exception:
            pass

        os.system('')
        self.ui.clear_screen()
        
        wechat_info = self.find_wechat_process()
        
        if not wechat_info:
            self.ui.print_log("错误: 未找到微信进程，请确保微信正在运行")
            return 1
        
        self._wechat_pid, self._wechat_hwnd = wechat_info
        
        while True:
            windows = WeChatWindowInfo.find_wechat_windows()
            if windows:
                self._wechat_hwnd = windows[0]['hwnd']
            
            # 默认更新 IDLE 状态
            friend_name = self._get_current_friend_name()
            level = self._get_level()
            self.ui.update_status(level, False, 0, friend_name)
            
            audio_pid = ProcessUtils.get_audio_process_pid("weixin")
            
            if audio_pid is not None:
                if self._wechat_pid != audio_pid:
                    self._wechat_pid = audio_pid
                
                with self._level_lock:
                    self._level_db = -60.0

                with self._open_audio_pipeline(self._wechat_pid) as sink:
                    self._session = None

                    while True:
                        current_audio_pid = ProcessUtils.get_audio_process_pid("weixin")
                        if current_audio_pid is None:
                            break

                        level = self._get_level()
                        friend_name = self._get_current_friend_name()

                        if not sink.is_recording:
                            self.ui.update_status(level, False, 0, friend_name)
                            self._tick_idle(sink, level)
                        else:
                            self._tick_recording(sink, level, friend_name)

                        time.sleep(self.config.tick_sleep_s)

                time.sleep(self.config.after_session_sleep_s)
            
            else:
                time.sleep(self.config.no_audio_sleep_s)
        
        return 0


def main():
    try:
        if not ProcessAudioCapture.is_supported():
            sys.stderr.write("错误: 需要 Windows 10 2004+ 或 Windows 11，并安装 process-audio-capture 库 (pip install process-audio-capture)\n")
            return 1
    except NameError:
        sys.stderr.write("错误: 缺少 process-audio-capture 库，请运行 pip install process-audio-capture\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"错误: 环境检查失败 - {e}\n")
        return 1
    
    try:
        recorder = WeChatVoiceRecorder()
        return recorder.run()
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        sys.stderr.write(f"错误: 程序异常退出 - {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
