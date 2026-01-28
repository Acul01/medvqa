# HPC GPU + JupyterLab - Schnellstart-Anleitung

Einfache Schritt-für-Schritt-Anleitung zur Nutzung von GPU-Ressourcen (H100/A100) auf dem HPC-Cluster der Uni Oldenburg über JupyterLab im lokalen Browser.

---

## Voraussetzungen

- HPC-Account der Uni Oldenburg
- SSH-Zugriff auf den Cluster
- Lokaler Rechner mit Browser
- Projekt bereits auf dem Cluster vorhanden (z.B. `~/medvqa/medvqa`)

---

# Schnellstart: JupyterLab mit GPU starten

## Schritt 1: SSH-Verbindung zum Cluster

```bash
ssh USERNAME@rosa.hpc.uni-oldenburg.de
```

Ersetzen Sie `USERNAME` mit Ihrem HPC-Benutzernamen.

---

## Schritt 2: Slurm-Modul laden (für srun-Befehl)

```bash
module load slurm
```

**Wichtig**: Laden Sie nur das Slurm-Modul, keine anderen Module (besonders nicht Python)!

---

## Schritt 3: GPU-Job starten

```bash
# GPU-Job anfordern
srun --partition=all_gpu.p --gres=gpu:1 --time=04:00:00 --mem=32G --pty bash
```

**Warten Sie**, bis Sie eine neue Eingabeaufforderung auf einem GPU-Knoten sehen (z.B. `[USERNAME@cfdg001 ~]$`).

---

## Schritt 4: Neue bash-Session starten

In der neuen Eingabeaufforderung (auf dem GPU-Knoten):

```bash
# Starten Sie eine neue bash-Session (wichtig für saubere Umgebung)
bash
```

---

## Schritt 5: Module laden und LD_LIBRARY_PATH setzen

```bash
# Laden Sie die benötigten Module
module load hpc-env/13.1
module load Python/3.11.3-GCCcore-13.1.0
module load SQLite/3.41.2-GCCcore-13.1.0
module load libffi/3.4.4-GCCcore-13.1.0
module load bzip2/1.0.8-GCCcore-13.1.0

# WICHTIG: Setzen Sie LD_LIBRARY_PATH auf zen3 (GPU-Knoten verwenden zen3, nicht zen4)
# Das Modul-System setzt LD_LIBRARY_PATH auf zen4, was zu "Illegal instruction" Fehlern führt
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/bzip2/1.0.8-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/libffi/3.4.4-GCCcore-13.1.0/lib64:/cm/shared/uniol/sw/zen3/13.1/SQLite/3.41.2-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib

# Setzen Sie auch PATH auf zen3
export PATH=/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/bin:$PATH

# Setzen Sie PYTHONPATH für user-installierte Pakete
export PYTHONPATH=/user/esba3078/.local/lib/python3.11/site-packages:$PYTHONPATH

# Prüfen Sie Python-Version
python --version  # Sollte 3.11.3 sein

# Prüfen Sie SQLite (wichtig für JupyterLab)
python -c "import sqlite3; print('SQLite Version:', sqlite3.sqlite_version)"  # Sollte 3.41.2 sein
```

---

## Schritt 6: Zu Projektverzeichnis wechseln

```bash
# Wechseln Sie zu Ihrem Projektverzeichnis
cd ~/medvqa/medvqa
# Oder: cd ~/IHR_PROJEKT_VERZEICHNIS
```

---

## Schritt 7: Virtuelle Umgebung aktivieren

```bash
# Aktivieren Sie die virtuelle Umgebung
source medvqa_env/bin/activate
# Oder: source IHR_ENV_NAME/bin/activate

# WICHTIG: Setzen Sie LD_LIBRARY_PATH und PATH erneut (nach venv-Aktivierung)
# Die venv kann diese Variablen überschreiben
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/bzip2/1.0.8-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/libffi/3.4.4-GCCcore-13.1.0/lib64:/cm/shared/uniol/sw/zen3/13.1/SQLite/3.41.2-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib
export PATH=/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/bin:$PATH
export PYTHONPATH=/user/esba3078/.local/lib/python3.11/site-packages:$PYTHONPATH

# Prüfen Sie Python
which python  # Sollte zen3 zeigen
python --version  # Sollte 3.11.3 sein
```

**Hinweis**: Falls die virtuelle Umgebung noch nicht existiert oder Pakete fehlen, installieren Sie sie:

```bash
# Falls venv nicht existiert: Erstellen Sie sie
python -m venv medvqa_env
source medvqa_env/bin/activate

# Installieren Sie die benötigten Pakete
pip install --upgrade pip setuptools wheel
pip install jupyterlab torch transformers pillow pandas tqdm nltk datasets huggingface_hub accelerate tensorboard
```

---

## Schritt 8: GPU prüfen

```bash
# Prüfen Sie, ob GPU verfügbar ist
nvidia-smi
```

Sie sollten die GPU-Details sehen (z.B. NVIDIA A100 oder H100).

---

## Schritt 9: JupyterLab starten

```bash
# Starten Sie JupyterLab
python -m jupyterlab --no-browser --ip=0.0.0.0 --port=8888
```

**Wichtig**: Notieren Sie sich:
- Den **Hostnamen** (z.B. `cfdg001`)
- Den **Token** (z.B. `abc123def456...`)
- Den **Port** (z.B. `8888`)

Die Ausgabe sieht etwa so aus:
```
[I ...] Jupyter Server is running at:
[I ...] http://cfdg001:8888/lab?token=abc123def456...
```

**Lassen Sie dieses Terminal-Fenster geöffnet!**

---

## Schritt 10: SSH-Tunnel erstellen

**Öffnen Sie ein NEUES Terminal auf Ihrem lokalen Rechner** (lassen Sie die JupyterLab-Session offen):

```bash
ssh -L 8888:HOSTNAME:8888 USERNAME@rosa.hpc.uni-oldenburg.de -N
```

Ersetzen Sie:
- `HOSTNAME` mit dem Hostnamen aus Schritt 9 (z.B. `cfdg001`)
- `USERNAME` mit Ihrem HPC-Benutzernamen

**Beispiel**:
```bash
ssh -L 8888:cfdg001:8888 esba3078@rosa.hpc.uni-oldenburg.de -N
```

**Hinweis**: Dieser Befehl bleibt aktiv und zeigt nichts an - das ist normal. Lassen Sie dieses Terminal-Fenster geöffnet.

---

## Schritt 11: JupyterLab im Browser öffnen

Öffnen Sie Ihren Browser und gehen Sie zu:

```
http://localhost:8888/lab?token=TOKEN
```

Ersetzen Sie `TOKEN` mit dem Token aus Schritt 9.

**Beispiel**:
```
http://localhost:8888/lab?token=abc123def456...
```

---

## Schritt 12: Virtuelle Umgebung in JupyterLab aktivieren

Im JupyterLab-Terminal:

```bash
# Laden Sie die Module (falls noch nicht geladen)
module load hpc-env/13.1
module load Python/3.11.3-GCCcore-13.1.0
module load SQLite/3.41.2-GCCcore-13.1.0
module load libffi/3.4.4-GCCcore-13.1.0
module load bzip2/1.0.8-GCCcore-13.1.0

# WICHTIG: Setzen Sie LD_LIBRARY_PATH auf zen3 (GPU-Knoten verwenden zen3, nicht zen4)
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/bzip2/1.0.8-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/libffi/3.4.4-GCCcore-13.1.0/lib64:/cm/shared/uniol/sw/zen3/13.1/SQLite/3.41.2-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib
export PATH=/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/bin:$PATH
export PYTHONPATH=/user/esba3078/.local/lib/python3.11/site-packages:$PYTHONPATH

# Aktivieren Sie die virtuelle Umgebung
source medvqa_env/bin/activate

# Prüfen Sie Python
python --version
which python
```

**Testen Sie die Pakete**:

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Schnellstart-Checkliste

- [ ] SSH-Verbindung zum Cluster hergestellt
- [ ] Slurm-Modul geladen (`module load slurm`)
- [ ] GPU-Job gestartet (`srun`)
- [ ] Neue bash-Session gestartet (`bash`)
- [ ] Module geladen (`hpc-env/13.1`, `Python/3.11.3-GCCcore-13.1.0`, `SQLite/3.41.2-GCCcore-13.1.0`, `libffi/3.4.4-GCCcore-13.1.0`, `bzip2/1.0.8-GCCcore-13.1.0`)
- [ ] Zu Projektverzeichnis gewechselt
- [ ] Virtuelle Umgebung aktiviert
- [ ] GPU verfügbar (`nvidia-smi`)
- [ ] JupyterLab gestartet (Token notiert)
- [ ] SSH-Tunnel erstellt
- [ ] JupyterLab im Browser geöffnet
- [ ] Virtuelle Umgebung in JupyterLab aktiviert

---

# Nützliche Befehle

## Job-Status prüfen
```bash
squeue -u $USER
```

## Job beenden
```bash
# Im ursprünglichen Terminal (nicht im GPU-Job)
scancel JOB_ID
```

## JupyterLab beenden
- Im Terminal wo JupyterLab läuft: `Ctrl+C` (zweimal)

## SSH-Tunnel beenden
- Im Terminal wo der Tunnel läuft: `Ctrl+C`

## Verfügbare GPUs prüfen
```bash
sinfo -o "%20N %10c %10m %25f %10G"
```

## Spezifische GPU anfordern
```bash
# H100 GPU
srun --partition=all_gpu.p --gres=gpu:H100:1 --time=04:00:00 --mem=32G --pty bash

# A100 GPU
srun --partition=all_gpu.p --gres=gpu:A100:1 --time=04:00:00 --mem=32G --pty bash
```

## Git-Modul laden (falls benötigt)
```bash
module load git/2.40.1-GCCcore-13.1.0-nodocs
```

## Training vom Checkpoint fortsetzen
```bash
# Finden Sie den neuesten Checkpoint
ls -td ~/medvqa/medvqa/outputs/IHR_OUTPUT_DIR/checkpoint-* | head -1

# Setzen Sie das Training fort (verwenden Sie die gleichen Parameter wie beim ursprünglichen Training)
python train.py \
    --model llava \
    --dataset vqarad \
    --data_root ~/medvqa/medvqa/data \
    --output_dir ~/medvqa/medvqa/outputs/IHR_OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --max_length 800 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --save_steps 25 \
    --eval_steps 25 \
    --val_split test \
    --dataloader_num_workers 1 \
    --resume_from_checkpoint /user/esba3078/medvqa/medvqa/outputs/IHR_OUTPUT_DIR/checkpoint-1300
```

**Wichtig**: 
- Verwenden Sie die **gleichen Parameter** wie beim ursprünglichen Training
- Geben Sie den **vollständigen Pfad** zum Checkpoint an (nicht nur `True`)
- Falls der neueste Checkpoint beschädigt ist, verwenden Sie einen früheren Checkpoint

---

# Troubleshooting

## Problem: "srun: command not found"
- **Lösung**: Laden Sie das Slurm-Modul: `module load slurm`

## Problem: "Illegal instruction" Fehler beim srun-Befehl
- **Ursache**: Module wurden vor dem `srun`-Befehl geladen (auf Login-Knoten zen4), aber der GPU-Job läuft auf einem anderen Knoten (zen3)
- **Lösung**: Laden Sie KEINE Module (außer Slurm) vor dem `srun`-Befehl. Laden Sie Module erst im GPU-Job nach `bash`

## Problem: "Illegal instruction" Fehler bei Python/JupyterLab
- **Ursache**: Das Modul-System setzt `LD_LIBRARY_PATH` auf zen4-Pfade, aber GPU-Knoten verwenden zen3. Das Python-Binary zeigt auf zen3, aber lädt zen4-Bibliotheken.
- **Lösung**: 
  1. Setzen Sie `LD_LIBRARY_PATH` explizit auf zen3:
     ```bash
     export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib:$LD_LIBRARY_PATH
     export PATH=/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/bin:$PATH
     ```
  2. Prüfen Sie Python: `python --version` (sollte jetzt funktionieren)
  3. Falls das Problem weiterhin besteht, starten Sie eine neue bash-Session: `bash` und wiederholen Sie Schritt 1

## Problem: "ModuleNotFoundError" in JupyterLab
- **Ursache**: Pakete fehlen in der virtuellen Umgebung
- **Lösung**: Installieren Sie die Pakete im JupyterLab-Terminal:
  ```bash
  module load hpc-env/13.1
  module load Python/3.11.3-GCCcore-13.1.0
  source medvqa_env/bin/activate
  pip install torch transformers pillow pandas tqdm nltk datasets huggingface_hub accelerate tensorboard
  ```

## Problem: "ImportError: undefined symbol: sqlite3_deserialize" oder "No module named 'pysqlite2'"
- **Ursache**: Die System-SQLite-Bibliothek ist zu alt für das Python-Modul. JupyterLab benötigt eine neuere SQLite-Version.
- **Lösung**: 
  1. Laden Sie das SQLite-Modul:
     ```bash
     module load SQLite/3.41.2-GCCcore-13.1.0
     ```
  2. Setzen Sie `LD_LIBRARY_PATH` mit der SQLite-Bibliothek:
     ```bash
     unset LD_LIBRARY_PATH
     export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/SQLite/3.41.2-GCCcore-13.1.0/lib:/cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib
     ```
  3. Prüfen Sie SQLite: `python -c "import sqlite3; print(sqlite3.sqlite_version)"` (sollte 3.41.2 sein)
  4. Versuchen Sie erneut, JupyterLab zu starten

## Problem: "ImportError: libffi.so.8: cannot open shared object file"
- **Ursache**: Die `libffi`-Bibliothek fehlt in `LD_LIBRARY_PATH`. PyTorch und andere Pakete benötigen `libffi` für `ctypes`.
- **Lösung**: 
  1. Laden Sie das libffi-Modul:
     ```bash
     module load libffi/3.4.4-GCCcore-13.1.0
     ```
  2. Fügen Sie libffi zu `LD_LIBRARY_PATH` hinzu:
     ```bash
     export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/libffi/3.4.4-GCCcore-13.1.0/lib64:$LD_LIBRARY_PATH
     ```
  3. Prüfen Sie, ob libffi gefunden wird: `ldd /cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so | grep ffi`
  4. Versuchen Sie erneut, PyTorch zu importieren: `python -c "import torch; print(torch.__version__)"`

## Problem: "ImportError: libbz2.so.1.0: cannot open shared object file"
- **Ursache**: Die `bzip2`-Bibliothek fehlt in `LD_LIBRARY_PATH`. Wird benötigt für `datasets` und andere Pakete.
- **Lösung**: 
  1. Laden Sie das bzip2-Modul:
     ```bash
     module load bzip2/1.0.8-GCCcore-13.1.0
     ```
  2. Fügen Sie bzip2 zu `LD_LIBRARY_PATH` hinzu:
     ```bash
     export LD_LIBRARY_PATH=/cm/shared/uniol/sw/zen3/13.1/bzip2/1.0.8-GCCcore-13.1.0/lib:$LD_LIBRARY_PATH
     ```
  3. Prüfen Sie, ob bzip2 gefunden wird: `ldd /cm/shared/uniol/sw/zen3/13.1/Python/3.11.3-GCCcore-13.1.0/lib/python3.11/lib-dynload/_bz2.cpython-311-x86_64-linux-gnu.so | grep bz2`
  4. Versuchen Sie erneut, das Training zu starten

## Problem: "Connection refused" beim SSH-Tunnel
- Prüfen Sie, ob JupyterLab noch läuft: `jupyter lab list` auf dem Cluster
- Prüfen Sie, ob der SSH-Tunnel noch aktiv ist
- Stellen Sie sicher, dass Sie den richtigen Hostnamen verwenden

## Problem: "Disk quota exceeded"
- Prüfen Sie Ihre Speichernutzung: `du -sh ~/*`
- Löschen Sie alte Checkpoints oder temporäre Dateien
- Prüfen Sie Checkpoints: `du -sh ~/medvqa/medvqa/outputs/*/checkpoint-* | sort -h`

---

# Weitere Ressourcen

- **HPC Wiki**: https://hpcwiki.uni-oldenburg.de
- **HPC Support**: hpcsupport@uol.de
- **Mattermost Gruppe**: Für Community-Support

---

**Viel Erfolg!** 🚀
