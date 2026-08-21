## About this project:

Evans, N. J., Dutilh, G., Wagenmakers, E.-J., & van der Maas, H. L. (2020). Double responding: A new constraint for models of speeded decision making. *Cognitive Psychology*, *121*, 101292. https://doi.org/10.1016/j.cogpsych.2020.101292

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

## Project File Structure

```
├── .dockerignore
├── .env
├── .git/
├── .gitignore
├── .Trash-0/
│   ├── files/
│   │   ├── pipeline_walkthrough.ipynb
│   │   ├── using_the_trained_model 1.ipynb
│   │   └── using_the_trained_model.ipynb
│   └── info/
│       ├── pipeline_walkthrough.ipynb.trashinfo
│       ├── using_the_trained_model 1.ipynb.trashinfo
│       └── using_the_trained_model.ipynb.trashinfo
├── data/
│   └── .gitkeep
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── LICENSE
├── notebooks/
│   ├── exploration/
│   │   ├── calibration_check.ipynb
│   │   └── corrected_model_check.ipynb
│   └── final/
│       ├── .ipynb_checkpoints/
│       │   ├── experiment-checkpoint.ipynb
│       │   ├── full_project-checkpoint.ipynb
│       │   └── full_project_task8-checkpoint.ipynb
│       ├── double_response_rdm.ipynb
│       ├── history_backup.json
│       ├── loss_curve.png
│       ├── posterior_pair_plot.png
│       └── simulated_data.csv
├── README.md
├── report/
│   └── Report.pdf
├── requirements.txt
├── results/
│   ├── trained_model.keras
│   └── trained_model_v2.keras
├── src/
│   ├── adapter.py
│   ├── bf_simulator.py
│   ├── config.yml
│   ├── load_model.py
│   ├── priors.py
│   ├── simulator.py
│   └── workflow.py
└── TODO.md
```
