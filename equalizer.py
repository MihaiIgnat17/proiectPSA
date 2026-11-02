"""
Egalizator Audio pe 3 Benzi (Bass / Mid / Treble) - tkinter + numpy + scipy

Benzi:
    Bass   : < 250 Hz        -> filtru Low-Shelf  (biquad RBJ), f0 = 250 Hz
    Mid    : 250 Hz - 4 kHz  -> filtru Peaking    (biquad RBJ), f0 = 1 kHz, BW = 4 octave
    Treble : > 4 kHz         -> filtru High-Shelf (biquad RBJ), f0 = 4 kHz
Filtre suplimentare:
    HPF (Low-Cut)  : Butterworth high-pass, ordin 2/4/6 (ex. 40 Hz)
    LPF (High-Cut) : Butterworth low-pass,  ordin 2/4/6 (ex. 12 kHz)

Toate filtrele sunt exprimate în format SOS (Second-Order Sections) și aplicate
cu scipy.signal.sosfilt, pentru stabilitate numerică.

Rulare:  python equalizer.py
"""

import os
import queue
import threading
from dataclasses import dataclass

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import signal
from scipy.io import wavfile

# Redarea este opțională: dacă sounddevice (sau PortAudio) lipsește,
# aplicația funcționează în continuare, doar butonul Play este dezactivat.
try:
    import sounddevice as sd
except Exception:  # ImportError sau OSError (PortAudio lipsă)
    sd = None

# freqz_sos este numele nou (SciPy >= 1.15); sosfreqz este numele vechi.
_freqz_sos = getattr(signal, "freqz_sos", None) or signal.sosfreqz


# =============================================================================
#  Constante
# =============================================================================
LOW_CROSSOVER = 250.0     # Hz - limita Bass / Mid
HIGH_CROSSOVER = 4000.0   # Hz - limita Mid / Treble
GAIN_MIN, GAIN_MAX, GAIN_STEP = -12.0, 12.0, 0.5

HPF_FREQS = ("20", "30", "40", "60", "80", "100")
LPF_FREQS = ("8000", "10000", "12000", "16000")
FILTER_ORDERS = ("2", "4", "6")
CLIP_MODES = ("Soft clip", "Normalizare (-0.2 dBFS)", "Hard clip")
EXPORT_FORMATS = ("PCM 16-bit", "Float 32-bit")


@dataclass(frozen=True)
class EQSettings:
    """Instantaneu imutabil al parametrilor; servește și drept cheie de cache."""
    bass_db: float = 0.0
    mid_db: float = 0.0
    treble_db: float = 0.0
    hpf_on: bool = False
    hpf_freq: float = 40.0
    lpf_on: bool = False
    lpf_freq: float = 12000.0
    filter_order: int = 4
    clip_mode: str = CLIP_MODES[0]
    auto_headroom: bool = False


# =============================================================================
#  DSP
# =============================================================================
class Equalizer3Band:
    """Construiește lanțul de filtre și procesează semnalul audio."""

    # ---------------------------------------------------------------- biquad
    # Formulele provin din "Audio EQ Cookbook" (Robert Bristow-Johnson).
    # Un biquad are funcția de transfer:
    #
    #            b0 + b1 z^-1 + b2 z^-2
    #   H(z) = --------------------------
    #            a0 + a1 z^-1 + a2 z^-2
    #
    # Coeficienții sunt normalizați cu a0 și returnați ca un rând SOS
    # [b0, b1, b2, 1, a1, a2], formatul cerut de sosfilt.

    @staticmethod
    def _to_sos(b0, b1, b2, a0, a1, a2):
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

    @staticmethod
    def _safe_freq(f0, fs):
        # Frecvența trebuie să rămână sub Nyquist (fs/2); altfel filtrul degenerează.
        return min(f0, 0.45 * fs)

    @classmethod
    def low_shelf(cls, fs, f0, gain_db, slope=1.0):
        """Low-Shelf: amplifică/atenuează uniform tot ce e sub f0.
        La f0 câștigul este jumătate din gain_db (în dB)."""
        f0 = cls._safe_freq(f0, fs)
        A = 10.0 ** (gain_db / 40.0)              # amplitudine (rădăcina câștigului liniar)
        w0 = 2.0 * np.pi * f0 / fs                # frecvența unghiulară normalizată
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
        k = 2.0 * np.sqrt(A) * alpha
        return cls._to_sos(
            A * ((A + 1) - (A - 1) * cw + k),
            2 * A * ((A - 1) - (A + 1) * cw),
            A * ((A + 1) - (A - 1) * cw - k),
            (A + 1) + (A - 1) * cw + k,
            -2 * ((A - 1) + (A + 1) * cw),
            (A + 1) + (A - 1) * cw - k,
        )

    @classmethod
    def high_shelf(cls, fs, f0, gain_db, slope=1.0):
        """High-Shelf: amplifică/atenuează uniform tot ce e peste f0."""
        f0 = cls._safe_freq(f0, fs)
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * f0 / fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
        k = 2.0 * np.sqrt(A) * alpha
        return cls._to_sos(
            A * ((A + 1) + (A - 1) * cw + k),
            -2 * A * ((A - 1) + (A + 1) * cw),
            A * ((A + 1) + (A - 1) * cw - k),
            (A + 1) - (A - 1) * cw + k,
            2 * ((A - 1) - (A + 1) * cw),
            (A + 1) - (A - 1) * cw - k,
        )

    @classmethod
    def peaking(cls, fs, f_low, f_high, gain_db):
        """Peaking (clopot) între f_low și f_high.
        Frecvența centrală este media geometrică: f0 = sqrt(f_low * f_high)
        (pentru 250 Hz - 4 kHz => f0 = 1 kHz), iar lățimea de bandă în octave
        este BW = log2(f_high / f_low) (=> 4 octave)."""
        f0 = cls._safe_freq(np.sqrt(f_low * f_high), fs)
        bw = np.log2(f_high / f_low)
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * f0 / fs
        cw, sw = np.cos(w0), np.sin(w0)
        # Relația RBJ dintre lățimea de bandă (octave) și alpha, cu corecție biliniară
        alpha = sw * np.sinh(np.log(2.0) / 2.0 * bw * w0 / sw)
        return cls._to_sos(
            1 + alpha * A,
            -2 * cw,
            1 - alpha * A,
            1 + alpha / A,
            -2 * cw,
            1 - alpha / A,
        )

    # ------------------------------------------------------------ lanț filtre
    @classmethod
    def build_sos(cls, settings: EQSettings, fs: int, skip_neutral: bool = True) -> np.ndarray:
        """Concatenează toate etajele într-o singură matrice SOS (N x 6).
        Filtrele în cascadă = produsul funcțiilor de transfer, ceea ce în
        format SOS înseamnă simpla suprapunere a rândurilor."""
        nyq = fs / 2.0
        sections = []

        # Benzile la 0 dB sunt identitate (A = 1 => b == a) - le omitem ca să
        # economisim calcul. În redarea live le păstrăm (skip_neutral=False) ca
        # numărul de secțiuni, deci și starea filtrului, să rămână constant.
        if settings.bass_db != 0 or not skip_neutral:
            sections.append(cls.low_shelf(fs, LOW_CROSSOVER, settings.bass_db))
        if settings.mid_db != 0 or not skip_neutral:
            sections.append(cls.peaking(fs, LOW_CROSSOVER, HIGH_CROSSOVER, settings.mid_db))
        if settings.treble_db != 0 or not skip_neutral:
            sections.append(cls.high_shelf(fs, HIGH_CROSSOVER, settings.treble_db))

        # Butterworth: răspuns maxim plat în banda de trecere.
        # Ordinul N dă o pantă de 6*N dB/octavă dincolo de frecvența de tăiere.
        if settings.hpf_on and settings.hpf_freq < 0.99 * nyq:
            sections.append(signal.butter(settings.filter_order, settings.hpf_freq,
                                          btype="highpass", fs=fs, output="sos"))
        # Un LPF peste Nyquist nu are ce tăia (ex. 12 kHz la fs = 22050 Hz).
        if settings.lpf_on and settings.lpf_freq < 0.99 * nyq:
            sections.append(signal.butter(settings.filter_order, settings.lpf_freq,
                                          btype="lowpass", fs=fs, output="sos"))

        if not sections:
            return np.empty((0, 6))
        return np.vstack(sections)

    # -------------------------------------------------------- anti-clipping
    @staticmethod
    def soft_clip(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Limitator soft: liniar până la prag, apoi compresie tanh spre 1.0.
        y = sign(x) * (t + (1 - t) * tanh((|x| - t) / (1 - t)))   pentru |x| > t
        Funcția e continuă și are derivata 1 în t, deci nu introduce „colțuri”
        (armonice dure) ca hard clipping-ul, iar |y| < 1 întotdeauna."""
        t = threshold
        mag = np.abs(x)
        over = mag > t
        if not np.any(over):
            return x
        y = x.copy()
        y[over] = np.sign(x[over]) * (t + (1.0 - t) * np.tanh((mag[over] - t) / (1.0 - t)))
        return y

    @staticmethod
    def normalize(x: np.ndarray, target_db: float = -0.2) -> np.ndarray:
        """Scalează liniar semnalul astfel încât vârful să fie la target_db dBFS."""
        peak = np.max(np.abs(x)) if x.size else 0.0
        if peak == 0:
            return x
        return x * (10.0 ** (target_db / 20.0) / peak)

    @staticmethod
    def headroom_gain(settings: EQSettings) -> float:
        """Pre-atenuare egală cu cel mai mare boost (ex. +6 dB -> x0.5).
        Muzica masterizată are deja vârfuri aproape de 0 dBFS, deci orice boost
        ar intra în limitator și s-ar auzi ca distorsiune, nu ca schimbare de timbru."""
        if not settings.auto_headroom:
            return 1.0
        boost = max(0.0, settings.bass_db, settings.mid_db, settings.treble_db)
        return 10.0 ** (-boost / 20.0)

    # ------------------------------------------------------------ procesare
    @classmethod
    def process(cls, data: np.ndarray, fs: int, settings: EQSettings) -> np.ndarray:
        """data: float32 în [-1, 1], formă (n,) pentru mono sau (n, canale).
        Filtrarea se face pe axa 0 (timp), deci fiecare canal e filtrat
        independent, fără probleme de shape între mono și stereo."""
        sos = cls.build_sos(settings, fs)
        # float64 pentru precizie mai bună în filtrare; câștigul e liniar, deci
        # poate fi aplicat înainte de filtre.
        y = data.astype(np.float64) * cls.headroom_gain(settings)
        if len(sos):
            y = signal.sosfilt(sos, y, axis=0)

        if settings.clip_mode == CLIP_MODES[0]:
            y = cls.soft_clip(y)
        elif settings.clip_mode == CLIP_MODES[1]:
            y = cls.normalize(y)
        # Plasă de siguranță finală pentru toate modurile.
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    @classmethod
    def frequency_response(cls, settings: EQSettings, fs: int, freqs: np.ndarray) -> np.ndarray:
        """Răspunsul în amplitudine (dB) al lanțului de filtre la frecvențele date."""
        sos = cls.build_sos(settings, fs)
        if not len(sos):
            return np.zeros_like(freqs)
        _, h = _freqz_sos(sos, worN=freqs, fs=fs)
        return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


# =============================================================================
#  I/O WAV
# =============================================================================
def load_wav(path: str):
    """Citește un WAV și îl convertește în float32 în intervalul [-1, 1]."""
    fs, data = wavfile.read(path)
    if data.dtype == np.uint8:                    # 8-bit este fără semn, centrat la 128
        audio = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:                  # 32-bit (și 24-bit, aliniat la stânga de SciPy)
        audio = data.astype(np.float32) / 2147483648.0
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32)
    else:
        raise ValueError(f"Format de eșantion nesuportat: {data.dtype}")
    return fs, np.clip(audio, -1.0, 1.0)


def save_wav(path: str, fs: int, audio: np.ndarray, fmt: str):
    if fmt == EXPORT_FORMATS[0]:
        pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(path, fs, pcm)
    else:
        wavfile.write(path, fs, audio.astype(np.float32))


# =============================================================================
#  GUI
# =============================================================================
class EqualizerApp:
    PLOT_W, PLOT_H = 520, 180
    PLOT_DB = 15.0                                # axa Y: ±15 dB

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Egalizator Audio pe 3 Benzi")
        self.root.resizable(False, False)

        self.fs = None
        self.audio = None                         # semnalul original (float32)
        self.processed = None                     # cache: ultimul rezultat procesat
        self.processed_key = None                 # EQSettings folosite pentru cache
        self.busy = False
        self.results = queue.Queue()              # rezultate de la thread-urile de lucru

        # Redare live
        self.stream = None                        # sd.OutputStream activ
        self.play_pos = 0                         # eșantionul curent
        self.zi = None                            # starea internă a filtrelor între blocuri
        self.live = None                          # (sos, gain, clip_mode) folosit de callback

        self._build_vars()
        self._build_ui()
        self._on_settings_changed()
        self.root.after(50, self._poll_results)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------------------------------------------- UI
    def _build_vars(self):
        self.bass = tk.DoubleVar(value=0.0)
        self.mid = tk.DoubleVar(value=0.0)
        self.treble = tk.DoubleVar(value=0.0)
        self.hpf_on = tk.BooleanVar(value=False)
        self.hpf_freq = tk.StringVar(value="40")
        self.lpf_on = tk.BooleanVar(value=False)
        self.lpf_freq = tk.StringVar(value="12000")
        self.order = tk.StringVar(value="4")
        self.clip_mode = tk.StringVar(value=CLIP_MODES[0])
        # Implicit dezactivat: cu headroom, un boost de bas atenuează tot restul
        # semnalului, iar basul rămâne la nivelul inițial (sună ca o scădere).
        self.headroom = tk.BooleanVar(value=False)
        self.export_fmt = tk.StringVar(value=EXPORT_FORMATS[0])
        self.file_info = tk.StringVar(value="Niciun fișier încărcat")
        self.status = tk.StringVar(value="Gata")

        for var in (self.bass, self.mid, self.treble, self.hpf_on, self.hpf_freq,
                    self.lpf_on, self.lpf_freq, self.order, self.clip_mode, self.headroom):
            var.trace_add("write", lambda *_: self._on_settings_changed())

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}
        main = ttk.Frame(self.root, padding=10)
        main.grid(sticky="nsew")

        # --- Fișier
        file_frame = ttk.LabelFrame(main, text="Fișier audio", padding=8)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Button(file_frame, text="Încarcă WAV...", command=self.load_file).grid(row=0, column=0)
        ttk.Label(file_frame, textvariable=self.file_info, width=60).grid(row=0, column=1, padx=10, sticky="w")

        # --- Benzi EQ
        eq_frame = ttk.LabelFrame(main, text="Egalizator (dB)", padding=8)
        eq_frame.grid(row=1, column=0, sticky="ns", **pad)
        bands = (("Bas\n< 250 Hz", self.bass),
                 ("Medii\n250 Hz – 4 kHz", self.mid),
                 ("Înalte\n> 4 kHz", self.treble))
        for col, (label, var) in enumerate(bands):
            ttk.Label(eq_frame, text=label, justify="center").grid(row=0, column=col, padx=10)
            # tk.Scale (nu ttk) pentru că suportă 'resolution' și afișează valoarea.
            tk.Scale(eq_frame, variable=var, from_=GAIN_MAX, to=GAIN_MIN,
                     resolution=GAIN_STEP, orient="vertical", length=200,
                     tickinterval=6).grid(row=1, column=col, padx=10)
        ttk.Button(eq_frame, text="Reset (0 dB)", command=self.reset_bands).grid(
            row=2, column=0, columnspan=3, pady=(8, 0))

        # --- Filtre suplimentare + anti-clipping
        opt = ttk.LabelFrame(main, text="Filtre suplimentare", padding=8)
        opt.grid(row=1, column=1, sticky="nsew", **pad)
        ttk.Checkbutton(opt, text="High-Pass (Low-Cut)", variable=self.hpf_on).grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.hpf_freq, values=HPF_FREQS, width=7,
                     state="readonly").grid(row=0, column=1, padx=4)
        ttk.Label(opt, text="Hz").grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(opt, text="Low-Pass (High-Cut)", variable=self.lpf_on).grid(row=1, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.lpf_freq, values=LPF_FREQS, width=7,
                     state="readonly").grid(row=1, column=1, padx=4)
        ttk.Label(opt, text="Hz").grid(row=1, column=2, sticky="w")
        ttk.Label(opt, text="Ordin Butterworth").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.order, values=FILTER_ORDERS, width=7,
                     state="readonly").grid(row=2, column=1, padx=4, pady=(6, 0))

        ttk.Separator(opt).grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(opt, text="Anti-clipping").grid(row=4, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.clip_mode, values=CLIP_MODES, width=22,
                     state="readonly").grid(row=4, column=1, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(opt, text="Headroom automat (fără distorsiuni la boost)",
                        variable=self.headroom).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Format export").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.export_fmt, values=EXPORT_FORMATS, width=22,
                     state="readonly").grid(row=6, column=1, columnspan=2, sticky="w", padx=4, pady=(6, 0))

        # --- Grafic răspuns în frecvență
        plot_frame = ttk.LabelFrame(main, text="Răspuns în frecvență", padding=8)
        plot_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
        self.canvas = tk.Canvas(plot_frame, width=self.PLOT_W, height=self.PLOT_H,
                                bg="#1e1e24", highlightthickness=0)
        self.canvas.grid()

        # --- Redare / export
        tr = ttk.Frame(main)
        tr.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
        self.btn_play = ttk.Button(tr, text="▶ Play", command=self.play)
        self.btn_play.grid(row=0, column=0, padx=4)
        self.btn_stop = ttk.Button(tr, text="■ Stop", command=self.stop)
        self.btn_stop.grid(row=0, column=1, padx=4)
        self.btn_save = ttk.Button(tr, text="Save Processed Audio...", command=self.save)
        self.btn_save.grid(row=0, column=2, padx=4)
        self.progress = ttk.Progressbar(tr, mode="indeterminate", length=120)
        self.progress.grid(row=0, column=3, padx=10)
        ttk.Label(tr, textvariable=self.status).grid(row=0, column=4, sticky="w")

        if sd is None:
            self.btn_play.state(["disabled"])
            self.btn_stop.state(["disabled"])
            self.status.set("Redare indisponibilă (pip install sounddevice)")
        self._update_buttons()

    # ------------------------------------------------------------ grafic
    def _freq_to_x(self, f, f_min, f_max):
        # Axă logaritmică: percepția frecvenței este logaritmică.
        return (np.log10(f) - np.log10(f_min)) / (np.log10(f_max) - np.log10(f_min)) * self.PLOT_W

    def _db_to_y(self, db):
        return self.PLOT_H / 2 - db / self.PLOT_DB * (self.PLOT_H / 2 - 10)

    def _on_settings_changed(self):
        try:
            settings = self.current_settings()
        except (tk.TclError, ValueError):
            return                                 # valoare intermediară invalidă
        self.draw_response(settings)
        self._update_live_filter(settings)

    def draw_response(self, settings: EQSettings):
        fs = self.fs or 44100
        f_min, f_max = 20.0, min(20000.0, fs / 2 * 0.999)
        c = self.canvas
        c.delete("all")

        # Grilă: linii de frecvență și nivel
        for f in (50, 100, 250, 500, 1000, 2000, 4000, 10000):
            if f_min < f < f_max:
                x = self._freq_to_x(f, f_min, f_max)
                c.create_line(x, 0, x, self.PLOT_H, fill="#33333d")
                c.create_text(x + 2, self.PLOT_H - 2, anchor="sw", fill="#77778a",
                              font=("TkDefaultFont", 7), text=f"{f // 1000}k" if f >= 1000 else str(f))
        for db in (-12, -6, 0, 6, 12):
            y = self._db_to_y(db)
            c.create_line(0, y, self.PLOT_W, y, fill="#55555f" if db == 0 else "#33333d")
            c.create_text(2, y - 1, anchor="sw", fill="#77778a", font=("TkDefaultFont", 7), text=f"{db:+d}")

        freqs = np.logspace(np.log10(f_min), np.log10(f_max), 300)
        mag = np.clip(Equalizer3Band.frequency_response(settings, fs, freqs),
                      -self.PLOT_DB * 1.5, self.PLOT_DB * 1.5)
        pts = []
        for f, m in zip(freqs, mag):
            pts.extend((self._freq_to_x(f, f_min, f_max), self._db_to_y(m)))
        c.create_line(*pts, fill="#4fc3f7", width=2, smooth=True)

    # ---------------------------------------------------------- setări
    def current_settings(self) -> EQSettings:
        """Citește variabilele Tk (doar din thread-ul principal!)."""
        return EQSettings(
            bass_db=round(self.bass.get(), 1),
            mid_db=round(self.mid.get(), 1),
            treble_db=round(self.treble.get(), 1),
            hpf_on=self.hpf_on.get(),
            hpf_freq=float(self.hpf_freq.get()),
            lpf_on=self.lpf_on.get(),
            lpf_freq=float(self.lpf_freq.get()),
            filter_order=int(self.order.get()),
            clip_mode=self.clip_mode.get(),
            auto_headroom=self.headroom.get(),
        )

    def reset_bands(self):
        for var in (self.bass, self.mid, self.treble):
            var.set(0.0)

    # ------------------------------------------------------------ fișier
    def load_file(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[("Fișiere WAV", "*.wav"), ("Toate", "*.*")])
        if not path:
            return
        self.stop()
        self._run_in_thread("load", lambda: (path, *load_wav(path)), "Se încarcă...")

    # ----------------------------------------------------- redare/export
    def play(self):
        """Redare în timp real: semnalul e filtrat bloc cu bloc în callback-ul
        audio, deci mutarea glisoarelor se aude imediat."""
        if sd is None or self.audio is None or self.busy:
            return
        self.stop()
        self.play_pos = 0
        self.zi = None
        channels = 1 if self.audio.ndim == 1 else self.audio.shape[1]
        try:
            self.stream = sd.OutputStream(samplerate=self.fs, channels=channels, dtype="float32",
                                          blocksize=1024, callback=self._audio_callback)
            self.stream.start()
        except Exception as exc:
            self.stream = None
            messagebox.showerror("Eroare redare", str(exc))

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.status.set("Oprit")

    def _update_live_filter(self, settings: EQSettings):
        sos = Equalizer3Band.build_sos(settings, self.fs or 44100, skip_neutral=False)
        # O singură atribuire (atomică) - callback-ul audio citește tuplul întreg.
        self.live = (sos, Equalizer3Band.headroom_gain(settings), settings.clip_mode)

    def _audio_callback(self, outdata, frames, time_info, status):
        """Rulează în thread-ul audio (fără apeluri Tk aici!).
        Filtrarea pe blocuri e echivalentă cu filtrarea întregului semnal doar
        dacă starea internă a filtrelor (zi) e transmisă de la un bloc la altul."""
        sos, gain, clip_mode = self.live
        block = self.audio[self.play_pos:self.play_pos + frames]
        n = len(block)
        self.play_pos += n

        x = block.astype(np.float64) * gain
        # Forma lui zi pentru sosfilt pe axa 0: (secțiuni, 2, canale...)
        zi_shape = (len(sos), 2) + x.shape[1:]
        if self.zi is None or self.zi.shape != zi_shape:
            self.zi = np.zeros(zi_shape)          # s-a schimbat numărul de secțiuni (HPF/LPF)
        if len(sos):
            x, self.zi = signal.sosfilt(sos, x, axis=0, zi=self.zi)
        # Normalizarea are nevoie de vârful întregului semnal, deci live folosim soft clip.
        if clip_mode != CLIP_MODES[2]:
            x = Equalizer3Band.soft_clip(x)

        outdata[:n] = np.clip(x, -1.0, 1.0).reshape(n, -1)
        if n < frames:
            outdata[n:] = 0
            raise sd.CallbackStop

    def save(self):
        if self.audio is None or self.busy:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("Fișiere WAV", "*.wav")],
                                            initialfile="processed.wav")
        if path:
            self._process_then("save", path=path, fmt=self.export_fmt.get())

    def _process_then(self, action, **extra):
        """Procesează (sau refolosește cache-ul) într-un thread separat,
        apoi execută acțiunea în thread-ul principal."""
        settings = self.current_settings()
        if self.processed is not None and self.processed_key == settings:
            self._finish(action, self.processed, settings, extra)
            return
        audio, fs = self.audio, self.fs
        self._run_in_thread(action,
                            lambda: (Equalizer3Band.process(audio, fs, settings), settings, extra),
                            "Se procesează...")

    def _finish(self, action, processed, settings, extra):
        self.processed, self.processed_key = processed, settings
        if action == "save":
            path, fmt, fs = extra["path"], extra["fmt"], self.fs
            # Și scrierea pe disc poate dura pentru fișiere mari.
            self._run_in_thread("saved", lambda: (save_wav(path, fs, processed, fmt), path)[1],
                                "Se salvează...")

    # ----------------------------------------------------------- threading
    def _run_in_thread(self, tag, func, message):
        """Execută func() în fundal. Thread-ul NU atinge widget-urile Tk;
        rezultatul ajunge în coadă și e preluat de _poll_results."""
        self.busy = True
        self._update_buttons()
        self.status.set(message)
        self.progress.start(12)

        def worker():
            try:
                self.results.put((tag, func(), None))
            except Exception as exc:  # raportăm orice eroare în UI
                self.results.put((tag, None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_results(self):
        try:
            while True:
                tag, result, error = self.results.get_nowait()
                self.busy = False
                self.progress.stop()
                if error is not None:
                    self.status.set("Eroare")
                    messagebox.showerror("Eroare", str(error))
                else:
                    self._handle_result(tag, result)
                # După _handle_result, ca self.audio să fie deja setat la încărcare.
                self._update_buttons()
        except queue.Empty:
            pass

        if self.stream is not None:
            if self.stream.active:
                fmt = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
                self.status.set(f"Redare live  {fmt(self.play_pos / self.fs)} / "
                                f"{fmt(len(self.audio) / self.fs)}")
            else:                                 # s-a terminat fișierul
                self.stop()
                self.status.set("Redare terminată")
        self.root.after(50, self._poll_results)

    def _handle_result(self, tag, result):
        if tag == "load":
            path, fs, audio = result
            self.fs, self.audio = fs, audio
            self.processed = self.processed_key = None
            channels = 1 if audio.ndim == 1 else audio.shape[1]
            kind = {1: "mono", 2: "stereo"}.get(channels, f"{channels} canale")
            duration = audio.shape[0] / fs
            name = os.path.basename(path)
            if len(name) > 32:                    # nume lung ar ascunde fs/canalele
                name = name[:29] + "..."
            self.file_info.set(f"{name}  |  {fs} Hz  |  {kind}  |  {duration:.1f} s")
            self.status.set("Fișier încărcat")
            self._on_settings_changed()           # filtrele și graficul depind de fs
        elif tag == "saved":
            self.status.set(f"Salvat: {os.path.basename(result)}")
        else:                                     # save după procesare
            processed, settings, extra = result
            self.status.set("Procesare terminată")
            self._finish(tag, processed, settings, extra)

    def _update_buttons(self):
        has_audio = self.audio is not None and not self.busy
        for btn in (self.btn_save,) + ((self.btn_play,) if sd is not None else ()):
            btn.state(["!disabled"] if has_audio else ["disabled"])

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    EqualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
