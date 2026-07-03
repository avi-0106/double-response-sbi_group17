# double-response-sbi_group17

## Running the Project (Docker)

1. Make sure Docker Desktop is installed and running.
2. From the repo root:
```bash
   Set-Content -Path ..\.env -Value "KERAS_BACKEND=jax"
   cd ..
   cd docker
   docker compose up --build
```
   (drop `--build` on later runs unless `requirements.txt` changed)
3. Copy the `http://127.0.0.1:8888/lab?token=...` URL printed in the terminal into your browser.
4. To stop: `Ctrl+C`, then `docker compose down`.
