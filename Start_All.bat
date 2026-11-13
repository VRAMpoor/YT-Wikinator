@echo off
:: One start-up path for everything: the Control Center finds the LLM engine,
:: the model and ytc.exe, starts them hidden, waits until they are really
:: ready and opens the web UI. Problems are shown in its log pane and in
:: logs\control_center.log, logs\llama-server.log and logs\ytc-server.log.
call "%~dp0Launch_ControlCenter.bat" %*
