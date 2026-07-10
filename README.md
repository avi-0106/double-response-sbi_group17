## Running the Project (Docker)

### First-time setup
1. Install Docker Desktop and make sure it's running.
2. Clone the repo and go into it:
```bash
   git clone https://github.com/avi-0106/double-response-sbi_group17.git
   cd double-response-sbi_group17
```
3. Create the `.env` file. It's not included in the repo on purpose (keeps secrets/env settings out of git), so everyone has to make it once:

   Git Bash:
```bash
   echo "KERAS_BACKEND=jax" > .env
```
   PowerShell:
```powershell
   Set-Content -Path .env -Value "KERAS_BACKEND=jax"
```

### Every time you want to work
Make sure you open the DOCKER application in your system
```bash
cd docker
docker compose up --build
```
Drop `--build` after the first run, unless `requirements.txt` changed.

Copy the `http://127.0.0.1:8888/lab?token=...` URL from the terminal into your browser. That's JupyterLab, running locally on your own machine, nothing gets uploaded anywhere.

You don't need VS Code for any of this. Inside JupyterLab you can edit files, run notebooks, and even open a terminal (Launcher tab > Terminal) for running Python directly. If you prefer editing in VS Code or another editor instead, that works too, any changes you save show up automatically in JupyterLab.

### To stop
`Ctrl+C` in the terminal running Docker, then:
```bash
docker compose down
```
