# -*- coding: utf-8 -*-
# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


import shutil
from platformdirs import user_config_dir
from PySide6.QtWidgets import QApplication


import gui.landing as landing
from engines.engine import StateManager, load_config

APP_NAME = "CALISTO"


def user_config_path():
    cfg_dir = Path(user_config_dir(APP_NAME))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.yaml"


def ensure_user_config_exists():
    dst = user_config_path()
    if dst.exists():
        return dst

    default = Path(__file__).resolve().parent / "config.yaml"
    with default.open("rb") as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)
    return dst


def main():
    config_path = ensure_user_config_exists()
    config = load_config(config_path)
    print(f"Using config file: {config_path}")
    state = None
    state_manager = StateManager(state)
    state_manager.set_state("config", config)
    state_manager.set_state("config_setup", "Default")
    MTGui = QApplication(sys.argv)
    window = landing.MainWindow(state_manager)
    window.show()
    MTGui.exec()


if __name__ == "__main__":
    main()
