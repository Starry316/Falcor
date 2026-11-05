@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo_neural.ps1" -Model "%~dp0build/neural-assets/blue_gray_pile_fullray_lod0.mpmat" -OptionsJson "%~dp0build/neural-assets/blue_gray_pile_appearance.options.json" %*

