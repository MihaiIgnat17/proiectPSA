# Egalizator Audio pe 3 Benzi

Aplicație desktop (Python + tkinter) pentru egalizarea fișierelor `.wav` pe 3 benzi,
cu filtre HPF/LPF opționale, anti-clipping, redare și export.

## Instalare

Necesită Python 3.8+ (tkinter este inclus în instalarea standard de pe python.org).

```bash
pip install -r requirements.txt
# sau
pip install numpy scipy sounddevice
```

`sounddevice` este necesar doar pentru redare. Fără el, aplicația pornește
și poate exporta fișiere, dar butonul **Play** este dezactivat.

## Rulare

```bash
python equalizer.py
```

1. **Încarcă WAV...**: se afișează numele fișierului, rata de eșantionare, numărul de canale și durata.
2. Reglezi glisoarele **Bas / Medii / Înalte** (de la -12 la +12 dB, pas de 0.5 dB). Graficul arată răspunsul în frecvență pe loc.
3. Opțional, activezi **High-Pass** (ex. 40 Hz) și/sau **Low-Pass** (ex. 12 kHz) și alegi ordinul Butterworth.
4. **▶ Play** procesează semnalul în fundal și îl redă. **Save Processed Audio...** îl exportă în WAV.

## Detalii DSP

| Bandă  | Filtru                 | Parametri                         |
|--------|------------------------|-----------------------------------|
| Bas    | Low-shelf (biquad RBJ)  | f0 = 250 Hz                       |
| Medii  | Peaking (biquad RBJ)    | f0 = 1 kHz (√(250·4000)), BW = 4 octave |
| Înalte | High-shelf (biquad RBJ) | f0 = 4 kHz                        |
| HPF    | Butterworth high-pass  | 20–100 Hz, ordin 2/4/6            |
| LPF    | Butterworth low-pass   | 8–16 kHz, ordin 2/4/6             |

- Toate etajele sunt combinate într-o singură matrice **SOS** și aplicate cu
  `scipy.signal.sosfilt(..., axis=0)`, deci mono `(n,)` și stereo `(n, 2)` sunt tratate la fel.
- Pentru benzi s-au folosit filtre shelving și peaking în locul unui crossover
  Butterworth LP/BP/HP: la 0 dB pe toate benzile semnalul trece neschimbat. Suma
  benzilor separate cu Butterworth nu dă un răspuns plat, din cauza fazei.
- Intrarea este convertită în `float32` în intervalul [-1, 1] (8/16/24/32-bit PCM și float).
- Anti-clipping:
  - **Soft clip**: liniar până la 0.9, apoi compresie `tanh`.
  - **Normalizare**: vârful este adus la -0.2 dBFS.
  - **Hard clip**: tăiere simplă.
- Procesarea, încărcarea și salvarea rulează în thread-uri separate. Rezultatele
  ajung în UI printr-o `queue.Queue`, deci interfața nu se blochează.
